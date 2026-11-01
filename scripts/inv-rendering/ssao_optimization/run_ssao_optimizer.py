"""Recover SSAO parameters by differentiating a rendered AO image.

This is the driver half of the SSAOOptimizer sample. The C++ side owns the
scene, the render graph and ASSAOPass; this script owns the loss and the
optimization loop, because the framework for them is PyTorch.

ASSAOPass is a differentiable op: it renders AO from the four parameters and,
given dL/dAO, returns dL/dparam. It has no notion of a loss. `ASSAOFunction`
wraps it as a torch.autograd.Function, so the loss is ordinary torch code. The
images cross between Falcor and torch as zero-copy CUDA views of shared
buffers; only the parameters, the gradient and the scalar loss pass through the
host.

The script exposes a generator-based protocol that the GUI can drive
frame-by-frame, so the window stays responsive during optimization:

    import run_ssao_optimizer as opt
    cfg = {**opt.OPTIONS, 'iterations': 75}
    gen = opt.optimize(cfg)
    for record in gen:      # called once per frame by SSAOOptimizer
        falcor.report_iteration(record)

`optimize()` yields one dict per iteration, containing:
    it       - iteration index (0 … iterations)
    loss     - L2 loss value
    params   - current parameter vector (shader units, 4 floats)
    grad     - analytic gradient vector (shader units, 4 floats)
    done     - True on the last iteration

`gradient_check()` compares the reverse pass against central differences of
the loss. The headless run calls it before optimizing, and the GUI calls it from
its results phase.

The optimizer is `torch.optim.Adam`, following the pattern in
scripts/inv-rendering/bsdf_optimization/run_bsdf_optimizer.py. Two properties of
the problem make it the right choice and made the previous hand-written
steepest descent the wrong one:

  * The four parameters live on scales that differ by two orders of magnitude.
    intensity and power are O(1); radius is uiRadius/1600 (0.0156 at the default
    25) and bias is uiBias/1000 (0.003 at the default 3). A single step size in
    linear units is either invisible for radius and bias or destabilising for
    intensity and power, and the previous version's answer — dividing each by the
    parameter's own magnitude to make the directions comparable — threw away the
    relative magnitudes that carry the conditioning information. Adam's per-
    parameter second moment does the same normalisation without discarding it.

  * (intensity, power) is close to a flat valley. `ao = 1 - intensity * (1 - ao^power)`
    is almost stationary along `intensity * (1 - r^power) = const`, so the loss has
    one very shallow and one very steep direction crossing at each iterate.
    Steepest descent zig-zags in a valley like that and needs a step count in the
    thousands; Adam's momentum carries it along the floor.

There is no longer a line search. It was a net loss: it cost two full GPU round
trips per trial and up to 27 trials per iteration, and because the loss is a
float32 sum over two million atomic contributions, its acceptance test
(`trial_loss < loss`, strict) started failing randomly once the true improvement
fell under the accumulation noise — at which point the search gave up and the run
reported "converged" while still far from the answer.

When run headless (``--script`` + ``--headless``), the entry point calls
``main()`` which drives the generator to completion and prints a summary.

Headless usage:

    SSAOOptimizer.exe --headless \\
        --script scripts/ssao_optimizer/run_ssao_optimizer.py \\
        [--iterations N] [--lr LR]
        [--target-mode ssao|ssao_plain|assao|all]
        [--skip-gradient-check]
"""

import argparse
import math
import sys
import time

import torch
import falcor


# ---------------------------------------------------------------------------
# Default optimization options.  The GUI reads this dict and builds widgets
# for every entry; the user can edit them before starting a run.
# ---------------------------------------------------------------------------

OPTIONS = {
    "iterations":         200,
    "lr":                 1e-2,
    "skip_gradient_check": False,
    "target_mode":        "ssao",   # ssao | ssao_plain | assao | all
    # Adam's other hyper-parameters. Exposed here so a run's settings are
    # recorded with its results, and so a sweep can vary them without editing
    # the loop.
    "adam_beta1":         0.9,
    "adam_beta2":         0.999,
    "adam_eps":           1e-6,
}

# The learnable parameters, in the order ASSAOPass' gradient buffer uses.
#
# Two of the seven render parameters are absent, for different reasons.
#
# The fade pair (fadeMul/fadeAdd). They only set the distance past which AO is
# lerped back to 1, so they contribute nothing outside that band and trade off
# against each other inside it. On Kroken at the default fade the band starts at
# linearZ = 3000 while most of the frame sits far short of it, where saturate's
# derivative is exactly zero.
#
# mipLevelFactor. This one is not a small gradient but an identically zero
# one over the range the experiment uses. Its only effect is on the HZB mip level,
#
#     mipLevel = log2(mipLevelFactor * scale * samplePos)
#
# and that is clamped to [0, mipLevels - 1]. At mipLevelFactor = 0.5 the
# argument spans [0.0139, 0.458], so log2 spans [-6.17, -1.13] and every tap is
# clamped to mip 0. Adam would not cope with that gracefully: a zero gradient is
# ambiguous between "this parameter is at a stationary point" and "this parameter
# has no effect", and Adam's answer -- divide by sqrt(v), which decays toward
# zero -- would keep taking unit-sized steps in a direction that does nothing.
# It is fixed at its configured value instead.
PARAM_NAMES = [
    "intensity",
    "power",
    "radius",
    "bias",
]

# Target renderer indices, matching SSAOOptimizer::TargetMode.
TARGET_SSAO       = 0
TARGET_SSAO_PLAIN = 1
TARGET_ASSAO      = 2


# ---------------------------------------------------------------------------
# ASSAOPass as a torch op
# ---------------------------------------------------------------------------

class _Views:
    """The sample's shared image buffers as [height, width] CUDA tensors.

    Buffer.to_torch() is a zero-copy view: the tensor reads and writes the
    buffer's memory, so it follows whatever Falcor last wrote there. Building
    the views once, on the calling thread, also creates every CUDA mapping
    before autograd's worker thread runs ASSAOFunction.backward.
    """

    def __init__(self):
        width, height = list(falcor.frame_dim())
        shape = [height, width]
        self.ao   = falcor.ao_map_buffer().to_torch(shape, falcor.float32)
        self.ref  = falcor.reference_buffer().to_torch(shape, falcor.float32)
        self.dldi = falcor.dL_dI_buffer().to_torch(shape, falcor.float32)


class ASSAOFunction(torch.autograd.Function):
    """AO = ASSAO(params), differentiable in the four learnable parameters.

    forward() renders AO at `params` and returns a copy, because the view is
    overwritten by the next forward while autograd may still hold this result.
    backward() writes dL/dAO into the pass' dL/dI buffer and returns dL/dparam.
    falcor.forward() and falcor.backward() order themselves against the CUDA
    stream, so neither side needs an explicit sync.
    """

    @staticmethod
    def forward(ctx, params, views):
        ctx.params = params.tolist()
        ctx.views = views
        falcor.set_params(ctx.params)
        falcor.forward()
        return views.ao.clone()

    @staticmethod
    def backward(ctx, grad_ao):
        # The reverse pass differentiates at whatever the pass holds, which is
        # not necessarily still the point this forward ran at.
        falcor.set_params(ctx.params)
        ctx.views.dldi.copy_(grad_ao)
        grad = falcor.backward()
        return torch.tensor(grad, dtype=torch.float32), None


def l2_loss(ao: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """L = 0.5 * mean((ao - ref)^2). The 0.5 makes dL/dao the residual / N."""
    return 0.5 * torch.mean((ao - ref) ** 2)


def evaluate(params, views=None):
    """Loss and dL/dparam at `params` (shader units). Returns (loss, grad)."""
    views = views if views is not None else _Views()
    p = torch.tensor(params, dtype=torch.float32, requires_grad=True)
    loss = l2_loss(ASSAOFunction.apply(p, views), views.ref)
    loss.backward()
    return float(loss.detach()), p.grad.tolist()


def compute_loss(params, views=None) -> float:
    """The loss alone at `params`: a forward with no reverse pass."""
    views = views if views is not None else _Views()
    with torch.no_grad():
        p = torch.tensor(params, dtype=torch.float32)
        return float(l2_loss(ASSAOFunction.apply(p, views), views.ref))


# ---------------------------------------------------------------------------
# Gradient check
# ---------------------------------------------------------------------------

def gradient_check(params, rel_step: float = 1e-3) -> dict:
    """Compare the reverse pass against central differences of the loss.

    The step is relative, h = rel_step * |param|, because the four parameters
    live on scales that differ by orders of magnitude: an absolute h fine for
    intensity is a no-op in float32 for a radius of 0.03. Leaves the pass at
    `params`.

    Returns {"rows": [...], "worst": float}, one row per parameter with keys
    name, analytic, numeric, rel_err, frozen. A row is frozen when its analytic
    gradient is zero: its numeric slope is then float32 noise in the loss, so
    its relative error is 1 by construction and is left out of `worst`.
    """
    params = [float(x) for x in params]
    views = _Views()
    _, analytic = evaluate(params, views)

    rows, worst = [], 0.0
    for i, name in enumerate(PARAM_NAMES):
        h = rel_step * max(abs(params[i]), 1e-6)
        plus, minus = list(params), list(params)
        plus[i] += h
        minus[i] -= h
        numeric = (compute_loss(plus, views) - compute_loss(minus, views)) / (2.0 * h)

        a = float(analytic[i])
        rel_err = abs(a - numeric) / max(abs(a), abs(numeric), 1e-12)
        frozen = abs(a) <= 1e-12
        # A NaN must stay visible in the headline rather than lose to max().
        if not frozen and (math.isnan(rel_err) or rel_err > worst):
            worst = rel_err
        rows.append({"name": name, "analytic": a, "numeric": numeric,
                     "rel_err": rel_err, "frozen": frozen})

    falcor.set_params(params)
    return {"rows": rows, "worst": worst}


def format_gradient_check(report: dict, rel_step: float) -> str:
    """The gradient check as the table the headless log prints."""
    lines = [f"gradient check (relative step {rel_step:e})",
             f"{'param':<22}{'analytic':<18}{'numeric':<18}rel.err"]
    frozen = 0
    for r in report["rows"]:
        err = "(frozen)" if r["frozen"] else f"{r['rel_err']:e}"
        frozen += r["frozen"]
        lines.append(f"{r['name']:<22}{r['analytic']:<18e}{r['numeric']:<18e}{err}")
    lines.append(f"worst relative error: {report['worst']:e}")
    if frozen:
        lines.append(f"{frozen} direction(s) frozen (zero analytic gradient), excluded above")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core optimization generator
# ---------------------------------------------------------------------------

def optimize(cfg: dict):
    """Descend from a guess and *yield* one record per iteration.

    The caller drives execution by pulling from this generator one step at a
    time.  Each yield suspends the loop so the C++ frame pump can render a
    progress frame, then the next call resumes from where it left off.

    Yields dicts with keys:
        it        - int,   iteration index (0 … iterations)
        loss      - float, L2 loss at the current parameters
        params    - list[float], current parameters in shader units
        grad      - list[float], analytic gradient in shader units
        done      - bool,  True on the final yield

    cfg keys used:
        iterations, lr, adam_beta1, adam_beta2, adam_eps
    """
    iterations = int(cfg.get("iterations", OPTIONS["iterations"]))
    lr         = float(cfg.get("lr", OPTIONS["lr"]))
    beta1      = float(cfg.get("adam_beta1", OPTIONS["adam_beta1"]))
    beta2      = float(cfg.get("adam_beta2", OPTIONS["adam_beta2"]))
    eps        = float(cfg.get("adam_eps", OPTIONS["adam_eps"]))

    n = len(PARAM_NAMES)

    # Scale each parameter to O(1) before handing it to Adam, and unscale on the
    # way into the pass.
    #
    # Adam's update is `lr * m_hat / (sqrt(v_hat) + eps)`, which is scale-free: a
    # coordinate whose gradient is consistently 1e-8 takes the same step as one
    # whose gradient is consistently 1e-2. That is the property that makes Adam
    # worth using here, but it only helps if the *parameters* are commensurate.
    # They are not — intensity and power are O(1) while radius is uiRadius/1600
    # (0.0156 at the default 25) and bias is uiBias/1000 (0.003 at the default 3),
    # a spread of two orders of magnitude. Left unscaled, a step of lr = 1e-2 is
    # 60% of radius' entire value and 0.5% of intensity's, so radius oscillates
    # across its range while intensity crawls. Measured on Kroken: radius stalls
    # at ~40% relative error and never settles, and intensity needs thousands of
    # iterations to cover a 0.3 gap.
    #
    # Dividing by the initial magnitude makes all four O(1), which is the space
    # Adam is designed for. It is not the log reparameterization the earlier
    # version used: that changed the *shape* of the objective and needed a chain
    # rule through it, whereas a constant per-coordinate divisor only rescales the
    # axes. The gradient transform is exact — if x = s * u then dL/du = s * dL/dx —
    # and autograd applies it through the multiplication below.
    init = torch.tensor(falcor.get_init_params(), dtype=torch.float32)
    scale = init.abs().clamp_min(1e-9)

    # The optimizer steps `u`; the pass is fed `scale * u`.
    u = torch.nn.Parameter((init / scale).clone().detach())

    optimizer = torch.optim.Adam(
        [{"params": u, "lr": lr}],
        betas=(beta1, beta2),
        eps=eps,
    )

    views = _Views()

    for it in range(iterations + 1):
        optimizer.zero_grad(set_to_none=True)

        # params is a non-leaf in the graph, so its gradient, the one in shader
        # units the record reports, has to be retained explicitly.
        params = u * scale
        params.retain_grad()

        loss = l2_loss(ASSAOFunction.apply(params, views), views.ref)
        loss.backward()

        record = {
            "it":     it,
            "loss":   float(loss.detach()),
            "params": params.detach().tolist(),
            "grad":   params.grad.tolist(),
            "done":   (it == iterations),
        }

        if it == iterations:
            yield record
            return

        optimizer.step()

        yield record


# ---------------------------------------------------------------------------
# Result reporter
# ---------------------------------------------------------------------------

def report(truth: torch.Tensor, recovered: torch.Tensor, names) -> float:
    """Print the recovered parameters next to the truth. Returns max rel err."""
    print("\n=== result ===")
    print(f"{'param':>22}  {'truth':>14}  {'recovered':>14}  {'rel err':>10}")
    worst = 0.0
    for i, name in enumerate(names):
        t = float(truth[i])
        r = float(recovered[i])
        rel = abs(r - t) / max(abs(t), 1e-12)
        worst = max(worst, rel)
        print(f"{name:>22}  {t:>14.6e}  {r:>14.6e}  {rel:>10.3e}")
    print(f"\nworst relative parameter error: {worst:.6e}")
    return worst


# ---------------------------------------------------------------------------
# Headless entry point
# ---------------------------------------------------------------------------

def _run_mode(mode: int, cfg: dict) -> tuple:
    """Run one target-mode experiment.  Returns (worst_rel, initial_loss, residual)."""
    device = torch.device("cpu")

    falcor.set_target_mode(mode)
    falcor.capture_target()
    falcor.sync()

    truth = torch.tensor(falcor.get_target_params(), dtype=torch.float32, device=device)
    init = falcor.get_init_params()

    if not cfg.get("skip_gradient_check", False):
        print("\n=== gradient check ===")
        gc_report = gradient_check(init, 1e-3)
        print(format_gradient_check(gc_report, 1e-3) + "\n")
        worst_gc = gc_report["worst"]
        ok = worst_gc < 1e-2
        print(f"gradient check verdict: {'OK' if ok else 'SUSPECT'} "
              f"(worst relative error {worst_gc:.3e})")

    initial_loss = compute_loss(init)
    falcor.set_params(init)

    print(f"\n{'iter':>6}  {'loss':>13}  {'param max err':>13}  "
          f"{'rel err':>10}  {'ms/iter':>8}")
    recovered_params = torch.tensor(falcor.get_init_params(),
                                     dtype=torch.float32, device=device)

    log_every = int(cfg.get("log_every", 10))
    iter_times_ms = []
    prev_t = time.perf_counter()

    for rec in optimize(cfg):
        it     = rec["it"]
        loss   = rec["loss"]
        params = torch.tensor(rec["params"], dtype=torch.float32, device=device)
        recovered_params = params
        residual = loss

        now = time.perf_counter()
        # The first sample covers the generator's startup (building the parameter
        # tensor and the Adam state), so it is reported separately from the
        # steady-state figure rather than averaged into it.
        iter_times_ms.append((now - prev_t) * 1000.0)
        prev_t = now

        param_err = (params - truth).abs()
        denom     = truth.abs().clamp_min(1e-12)
        rel_err   = float((param_err / denom).max())

        if it % log_every == 0 or rec.get("done"):
            shown_ms = iter_times_ms[-1] if it == 0 else None
            suffix = f"  {shown_ms:>8.1f}" if shown_ms is not None else " " * 10
            print(f"{it:>6}  {loss:>13.6e}  {float(param_err.max()):>13.6e}  "
                  f"{rel_err:>10.3e}{suffix}")

    worst_rel = report(truth, recovered_params, list(falcor.param_names()))
    print(f"initial loss:  {initial_loss:.6e}")
    print(f"residual loss at recovered parameters: {residual:.6e}")

    # Steady-state per-iteration cost: the median over every iteration after the
    # first, which is what a wall-clock estimate of a longer run should use.
    if len(iter_times_ms) > 1:
        steady = sorted(iter_times_ms[1:])
        median_ms = steady[len(steady) // 2]
        print(f"first iteration wall time: {iter_times_ms[0]:.1f} ms "
              f"(includes optimizer setup)")
        print(f"steady-state median iteration: {median_ms:.1f} ms")
    return worst_rel, initial_loss, residual


def main() -> int:
    # The AO images are shared with torch as CUDA tensors.
    if not torch.cuda.is_available():
        print("error: CUDA is not available to torch; the AO images are shared "
              "as CUDA tensors.", file=sys.stderr)
        return 1
    print(f"torch {torch.__version__}, images on {torch.cuda.get_device_name(0)}, "
          f"parameters on cpu")

    num_params = falcor.num_params()
    names = list(falcor.param_names())
    assert num_params == len(PARAM_NAMES), \
        f"shader has {num_params} params, script expects {len(PARAM_NAMES)}"
    print(f"optimizing {num_params} parameters: {', '.join(names)}")

    frame_dim   = list(falcor.frame_dim())
    pixel_count = frame_dim[0] * frame_dim[1]
    print(f"AO image {frame_dim[0]}x{frame_dim[1]} ({pixel_count} pixels)")

    if pixel_count == 0:
        print("error: the AO image has zero pixels; did the graph run?",
              file=sys.stderr)
        return 1

    cfg = {**OPTIONS}
    cfg.update(_parse_headless_args(_script_argv()))

    target_mode = cfg.get("target_mode", "ssao")
    modes = (
        [TARGET_SSAO, TARGET_SSAO_PLAIN, TARGET_ASSAO]
        if target_mode == "all"
        else [{"ssao": TARGET_SSAO, "ssao_plain": TARGET_SSAO_PLAIN,
               "assao": TARGET_ASSAO}[target_mode]]
    )

    failures  = 0
    summaries = []
    for mode in modes:
        print(f"\n{'#' * 72}\n# target renderer: {list(falcor.target_modes())[mode]}\n{'#' * 72}")
        try:
            worst_rel, initial_loss, residual = _run_mode(mode, cfg)
        except Exception as e:
            print(f"ERROR in mode {mode}: {e}")
            failures += 1
            continue
        summaries.append((mode, worst_rel, initial_loss, residual))

    if len(summaries) > 1:
        print("\n=== summary ===")
        mode_names = list(falcor.target_modes())
        print(f"{'target':>28}  {'param max rel err':>18}  "
              f"{'initial loss':>13}  {'residual':>13}")
        for mode, worst_rel, initial_loss, residual in summaries:
            print(f"{mode_names[mode]:>28}  {worst_rel:>18.3e}  "
                  f"{initial_loss:>13.6e}  {residual:>13.6e}")

    return 1 if failures else 0


# ---------------------------------------------------------------------------
# Argument parsing helpers
# ---------------------------------------------------------------------------

def _parse_headless_args(argv: list) -> dict:
    """Parse command-line overrides and return a partial cfg dict."""
    parser = argparse.ArgumentParser(prog="run_ssao_optimizer", add_help=False)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--log-every",  type=int)
    parser.add_argument("--lr",         type=float)
    parser.add_argument("--target-mode",
                        choices=["ssao", "ssao_plain", "assao", "all"])
    parser.add_argument("--skip-gradient-check", action="store_true", default=None)
    args, _ = parser.parse_known_args(argv)

    out = {}
    if args.iterations is not None:           out["iterations"]          = args.iterations
    if args.log_every is not None:            out["log_every"]            = args.log_every
    if args.lr is not None:                   out["lr"]                   = args.lr
    if args.target_mode is not None:          out["target_mode"]          = args.target_mode
    if args.skip_gradient_check:              out["skip_gradient_check"]  = True
    return out


def _script_argv() -> list:
    """The arguments this script should parse.

    Under the sample, Scripting::runScriptFromFile evaluates this file with a
    bare pybind11::exec: sys.argv is the executable's, and the sample's own
    flags are in it, so parsing sys.argv there would misread --script/--headless
    as optimizer options. The sample publishes the leftovers as the global
    ``falcor_argv``, which is what we parse when it is present. Standalone,
    there is no such global and sys.argv is the real thing.
    """
    injected = globals().get("falcor_argv")
    if injected is not None:
        return list(injected)
    return sys.argv[1:]


# ---------------------------------------------------------------------------
# Entry-point guard
# ---------------------------------------------------------------------------

def _is_entry_point() -> tuple:
    """Whether this file is being run as the entry point.

    The sample executes it through Scripting::runScriptFromFile, which is a bare
    pybind11::exec and therefore never binds __name__ — so the usual
    ``if __name__ == "__main__"`` guard would be false, and the script would
    define its functions and exit having done nothing. A missing __name__ means
    the same thing here as the string "__main__" does standalone.

    Returns (is_entry, owns_process).  ``owns_process`` decides whether sys.exit
    may be called: under pybind11::exec a SystemExit would surface as a C++
    exception in the sample.
    """
    name = globals().get("__name__")
    return (name is None or name == "__main__"), name == "__main__"


_RUN_AS_ENTRY, _OWNS_PROCESS = _is_entry_point()

if _RUN_AS_ENTRY:
    _status = main()
    if _OWNS_PROCESS:
        sys.exit(_status)
