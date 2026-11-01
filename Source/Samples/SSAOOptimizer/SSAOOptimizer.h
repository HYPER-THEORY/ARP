/// Copyright (C) 2026-2027 ARP (Auto Render Pipeline). All Rights Reserved.

#pragma once

#include "Falcor.h"
#include "Core/SampleApp.h"
#include "Core/Pass/FullScreenPass.h"
#include "RenderGraph/RenderGraph.h"

#include "SSAOPass/ASSAOPass.h"
#include "SSAOPass/SSAOPass.h"

#include <imgui.h>

#include <array>
#include <filesystem>
#include <string>
#include <vector>

using namespace Falcor;

/** Recovers SSAO parameters by differentiating a rendered AO image.

    The experiment:

      1. Render an AO image with SSAOPass at a chosen parameter set (the ground
         truth), and freeze it as the target.
      2. Restart from a perturbed guess and iterate: evaluate AO at the current
         guess with ASSAOPass, measure the L2 loss against the target in
         PyTorch, hand its dL/dAO back to ASSAOPass' reverse pass for the
         gradient, and step an optimizer.
      3. Report whether the four learnable parameters come back.

    The scene is Kroken, loaded the same way the Kroken sample loads it, because
    its depth complexity is what makes the occlusion term actually sensitive to
    radius and bias rather than to a handful of silhouettes.

    Which renderer produces the target matters, and the sample exposes both:

      - targetMode = SSAO (default). SSAOPass renders the target, with
        `useDiffSurrogateHZB` on so that its forward is the same smooth function
        ASSAOPass differentiates. Recovery is then exact up to float rounding.
      - targetMode = SSAOPlain. SSAOPass renders the target the way the
        interactive graph does — point-sampled HZB, hard fade early-out. This is
        what the parameters were "really" rendered with, but it is not the
        function the gradient is taken of, so the loss bottoms out at a non-zero
        residual and the parameters recover only approximately. The sample
        reports that floor and the bias it causes instead of hiding them.
      - targetMode = ASSAO. ASSAOPass' own forward is the target. Exact, and the
        simplest thing to check the machinery against.

    Driving
    -------
    The loss and the optimization loop live in Python (scripts/ssao_optimizer/
    run_ssao_optimizer.py) because the user-facing framework for them is
    PyTorch, and the loop uses torch.optim.Adam. ASSAOPass only renders AO and
    turns dL/dAO into dL/dparam; the script wraps that pair as a
    torch.autograd.Function. The C++ side owns the scene, the graph and the
    pass, and exposes them through script bindings: `forward()` renders AO at
    the current parameters, `backward()` takes dL/dAO and returns the gradient.

    The AO image, the target and dL/dAO cross between the two through
    ResourceBindFlags::Shared buffers, which PyTorch maps as CUDA tensors in
    place. forward() and backward() copy between those buffers and ASSAOPass'
    textures on the device and order the two APIs with shared-fence waits, so
    no image passes through the host. Only the four parameters, the four
    gradients and the scalar loss do, and backward()'s read of the gradient is
    the one host wait per iteration.

    GUI flow
    --------
    The UI is a three-phase state machine:

      Setup   — user loads a script, sets target/init parameters, presses Start.
      Running — the Python generator drives optimization step-by-step; the GUI
                shows a progress bar and the background shows the error map from
                the last step. No interactive controls are available.
      Results — optimization finished; user can scrub an iteration slider,
                inspect the loss curve, the per-parameter table, and choose what
                the background shows (AO, reference, error).

    The generator protocol: the loaded script must export an `optimize(cfg)`
    generator that yields one dict per iteration with keys:
        it, loss, params, grad, converged, done
    and an `OPTIONS` dict used to populate the setup controls.

    Displaying a past iteration
    ---------------------------
    Nothing per-iteration is cached. The graph runs every frame, and
    ASSAOPass::execute() unconditionally forwards at the pass' current
    parameters, so writing `mHistory[viewIt].params` into the pass and then
    running the graph *is* the AO image for that iteration, in the graph's aoMap
    texture that the error map reads. The temporal jitter is pinned by
    `lockedFrame` and the geometry is already frozen, so the recomputed image is
    bit-identical to the one that iteration actually rendered. Caching instead
    would cost 8 MB per iteration at 1080p.
*/
class SSAOOptimizer : public SampleApp
{
public:
    SSAOOptimizer(const SampleAppConfig& config);
    ~SSAOOptimizer();

    void setCommandLine(int argc, char** argv);

    void onLoad(RenderContext* pRenderContext) override;
    void onFrameRender(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo) override;
    void onGuiRender(Gui* pGui) override;
    bool onKeyEvent(const KeyboardEvent& keyEvent) override;
    bool onMouseEvent(const MouseEvent& mouseEvent) override;

private:
    // ------------------------------------------------------------------
    // Enums and constants
    // ------------------------------------------------------------------

    enum class TargetMode
    {
        SSAO,
        SSAOPlain,
        ASSAO
    };
    static const Gui::DropdownList kTargetModeList;

    enum class Phase
    {
        Setup,    ///< Waiting for user to load script and press Start.
        Running,  ///< Generator is live; step every frame.
        Results,  ///< Optimization finished; show result UI.
    };

    enum class BackgroundView : uint32_t
    {
        AO        = 0,   ///< View.ps.slang viewMode 0.
        Reference = 1,   ///< View.ps.slang viewMode 1.
        Error     = 2,   ///< View.ps.slang viewMode 2.
    };
    static const Gui::DropdownList kBackgroundViewList;

    /// The seven render parameters in the order their shader constant buffer
    /// declares them.
    enum class Param : uint32_t
    {
        Intensity = 0,
        Power,
        Radius,
        Bias,
        FadeMul,
        FadeAdd,
        MipLevelFactor,
        Count
    };
    static constexpr uint32_t kRenderParamCount = uint32_t(Param::Count);

    /// The four learnable parameters, a prefix of Param: the fade pair and
    /// mipLevelFactor are excluded.
    ///
    /// `fadeMul`/`fadeAdd` only set the distance past which AO is lerped back to
    /// 1. Outside that band they contribute nothing, and inside it they trade off
    /// against each other (any fadeDistance/fadeRadius pair with the same ratio
    /// draws the same image), so a gradient there is either zero or ambiguous.
    /// On Kroken at the default fade the band starts at linearZ = 3000 while most
    /// of the frame sits well short of that, where saturate's derivative is zero.
    ///
    /// `mipLevelFactor` is worse than ambiguous: its gradient is identically
    /// zero over (0, 2.2]. The mip level it feeds into is
    ///
    ///     mipLevel = log2(mipLevelFactor * scale * samplePos)
    ///
    /// and aTrilinearSample() clamps that to [0, mipLevels - 1]. With scale in
    /// {1/3, 2/3, 1} and samplePos in {1/12 ... 11/12}, the argument spans
    /// [0.0139, 0.458] at the default 0.5, so log2 spans [-6.17, -1.13] and every
    /// tap clamps to mip 0. A direction whose derivative is identically zero is a
    /// column of zeros in the gradient, which an optimizer cannot tell from a
    /// parameter that is merely lagging.
    ///
    /// Both stay constant buffer values the forward applies; only the reverse
    /// pass ignores them.
    enum class LearnableParam : uint32_t
    {
        Intensity = 0,
        Power,
        Radius,
        Bias,
        Count
    };
    static constexpr uint32_t kParamCount = uint32_t(LearnableParam::Count);

    /// Render-parameter index for each learnable parameter, in learnable order.
    static constexpr std::array<uint32_t, kParamCount> kLearnableToRender = {
        uint32_t(Param::Intensity),
        uint32_t(Param::Power),
        uint32_t(Param::Radius),
        uint32_t(Param::Bias),
    };

    /// Human-readable names in gradient-buffer order.
    static const std::array<const char*, kParamCount> kParamNames;

    /// The seven render parameters the optimizer works with, in shader units.
    struct ShaderParams
    {
        float intensity = 0.5f;
        float power = 2.0f;
        float radius = 100.0f / 1600.0f;
        float bias = 3.0f / 1000.0f;
        float fadeMul = 1.0f / 5000.0f;
        float fadeAdd = -8000.0f / 5000.0f;
        float mipLevelFactor = 0.5f;

        std::array<float, kRenderParamCount> toArray() const;
        static ShaderParams fromArray(const std::array<float, kRenderParamCount>& values);

        std::array<float, kParamCount> toLearnableArray() const;
        void setLearnable(uint32_t i, float v);
    };

    /// Setup parameters in the units SSAOPass' panel edits.
    struct UIParams
    {
        float intensity = 0.5f;
        float power = 2.0f;
        float radius = 25.0f;
        float bias = 3.0f;
        float fadeDistance = 8000.0f;
        float fadeRadius = 5000.0f;
        float mipLevelFactor = 0.5f;
    };

    // ------------------------------------------------------------------
    // Per-iteration record (mirrors the Python yield dict)
    // ------------------------------------------------------------------

    struct IterRecord
    {
        int                it        = 0;
        float              loss      = 0.f;
        std::vector<float> params;   ///< Shader units, 4 floats.
        std::vector<float> grad;     ///< Shader units, 4 floats.
        bool               converged = false;
        bool               done      = false;
    };

    // ------------------------------------------------------------------
    // Optimizer config (read from script OPTIONS dict + user edits)
    // ------------------------------------------------------------------

    struct OptimizerConfig
    {
        int   iterations = 200;
        float lr         = 0.01f;

        /// Read from the script's OPTIONS and handed straight back to it. The
        /// script uses it to decide whether to run its own gradient check, which
        /// it only does when it drives itself; the GUI has a gradient check of
        /// its own in the results phase and never consults this.
        bool  skipGradientCheck = false;
    };

    /// One row of the gradient-check report: the analytic slope the run
    /// recorded, the central difference recomputed now, and their disagreement.
    struct GradCheckRow
    {
        std::string name;
        double      analytic = 0.0;
        double      numeric  = 0.0;
        double      relErr   = 0.0;
        /// The run zeroed this direction (its analytic gradient was below the
        /// script's freeze threshold), so a relErr of 1 is the finite-difference
        /// floor showing through, not a disagreement.
        bool        frozen   = false;
    };

    struct GradCheckResult
    {
        bool                     valid = false;
        int                      it    = -1;
        double                   worst = 0.0;
        std::vector<GradCheckRow> rows;
    };

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    void parseCommandLine();
    static std::filesystem::path resolveScriptPath(const std::filesystem::path& given);

    /// Parse "i,p,r,b" into the four learnable UI parameters. Returns false and
    /// leaves `out` untouched on anything else.
    static bool parseParamsArg(const std::string& text, UIParams& out);

    void loadScene(RenderContext* pRenderContext);
    void buildRenderGraph();
    void registerPasses();
    void registerOptimizerBindings();

    void runGraph(RenderContext* pRenderContext);
    ref<Texture> getAOMap() const;
    uint2 getFrameDim() const;

    /// Render the target with the selected renderer and copy it into
    /// mpReferenceBuffer.
    void captureTarget(RenderContext* pRenderContext);

    /// Run ASSAOPass' forward at the current parameters and copy its AO texture
    /// into mpAOBuffer. Ends with a device-side wait, so CUDA work queued after
    /// this sees the new image without blocking the host.
    void forward(RenderContext* pRenderContext);

    /// Copy dL/dAO from mpDLdIBuffer into mpDLdITexture, run ASSAOPass'
    /// backward on it, and read the four gradients back. Starts with a
    /// device-side wait on the CUDA stream that wrote mpDLdIBuffer.
    std::vector<float> backward(RenderContext* pRenderContext);

    std::vector<float> getParams() const;
    void setParams(const std::vector<float>& values);
    std::vector<float> getParamsUI() const;
    void setParamsUI(const std::vector<float>& values);
    std::vector<float> getTargetParams() const;
    void setTargetParams(const std::vector<float>& values);
    std::vector<float> getInitParams() const;
    void setInitParams(const std::vector<float>& values);

    void setTargetMode(uint32_t mode);
    uint32_t getTargetMode() const { return uint32_t(mTargetMode); }

    /// Gradient check at one recorded iteration: central differences of the loss
    /// at that iteration's parameters, against the analytic gradient there. The
    /// loss lives in the script, so this calls the script's gradient_check() and
    /// converts its rows for display. Both sides are in shader units, so no
    /// chain rule is needed to make them comparable.
    GradCheckResult gradientCheckAt(const IterRecord& rec, float relativeStep);

    std::vector<float> getLastGradient() const { return mLastGradient; }
    static std::vector<std::string> getParamNames();

    std::vector<float> readBackBuffer(
        RenderContext* pRenderContext, const ref<Buffer>& pBuffer, size_t floatCount);

    /// Read several device buffers with one copy pass and one flush. `out`
    /// receives the concatenation, in the order the buffers were given.
    ///
    /// This is the form to prefer: each submit(true) drains the whole queue, so the
    /// number of flushes, not the bytes moved, is what an evaluation costs.
    void readBackBuffers(
        RenderContext* pRenderContext,
        const std::vector<ref<Buffer>>& buffers,
        const std::vector<size_t>& floatCounts,
        std::vector<float>& out);

    // ---- Script / generator plumbing ----

    /// Load a script file and read its OPTIONS dict into mConfig.
    bool loadScript(const std::filesystem::path& path);

    /// Start the optimization run: capture target, call optimize(cfg),
    /// store the resulting generator object.
    void startOptimization();

    /// Advance the generator by one step and store the record.
    /// Returns false when the generator is exhausted.
    bool advanceGenerator();

    /// Build a Python dict from mConfig to pass to optimize().
    pybind11::dict buildConfigDict() const;

    // ---- Parameter conversion (UI units <-> shader units) ----

    /// Fold the setup sliders (UI units) into mTargetParams / mInitParams.
    /// The init guess only exposes the four learnable parameters, so its fade
    /// pair is taken from the target: a fade mismatch between the forward and
    /// the reference puts an irreducible floor under the loss that no optimizer
    /// can descend past, and the fade is not learnable to begin with.
    void applySetupParamsToShaderUnits();

    static ShaderParams uiToShaderParams(const UIParams& ui, bool radiusInWorldSpace);
    static UIParams shaderToUIParams(const ShaderParams& shader, bool radiusInWorldSpace);
    static Properties makeUIProperties(const UIParams& ui);
    static UIParams readUIProperties(const Properties& props);

    ShaderParams getASSAOParams() const;
    void setASSAOParams(const ShaderParams& params);
    static void setSSAOParams(SSAOPass& pass, const ShaderParams& params);

    /// Write the parameters of the iteration being displayed into the pass, so
    /// that the next graph run renders exactly that iteration's AO.
    void applyDisplayParams();

    /// Index into mHistory of the iteration the UI is showing, or -1.
    int displayedIndex() const;

    // ---- GUI sections ----

    void renderSetupUI(Gui::Window& w);
    void renderRunningUI(Gui::Window& w);
    void renderResultsUI(Gui::Window& w);
    void renderLossCurve(Gui::Widgets& w);
    void renderParamTable(Gui::Widgets& w);
    void renderGradCheckUI(Gui::Widgets& w);

    // ---- Background rendering ----

    /// Run the view pass in whichever mode the current phase/view selects and
    /// present it on the frame buffer.
    void renderBackground(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo);

    // ---- View-pass shading ----

    void compileViewPass();

    // ---- Texture <-> buffer interop ----

    /// Size the shared buffers and the dL/dAO texture to ASSAOPass' frame.
    /// Reallocates only when the size changes, since the CUDA mappings, and the
    /// tensors the script built on them, do not survive a new buffer.
    void createInteropResources();
    void compileCopyPasses();
    void copyTextureToBuffer(RenderContext* pRenderContext, const ref<Texture>& pSrc, const ref<Buffer>& pDst);
    void copyBufferToTexture(RenderContext* pRenderContext, const ref<Buffer>& pSrc, const ref<Texture>& pDst);

    // ------------------------------------------------------------------
    // Scene / graph
    // ------------------------------------------------------------------

    ref<Scene>       mpScene;
    ref<RenderGraph> mpGraph;

    ref<SSAOPass>  mpRefAOPass;
    ref<ASSAOPass> mpASSAOPass;

    TargetMode mTargetMode = TargetMode::SSAO;

    ShaderParams mTargetParams;
    ShaderParams mInitParams;

    std::vector<float> mLastGradient;
    float              mLastLoss = 0.f;

    ref<Buffer> mpStagingBuffer;

    /// The images PyTorch reads and writes, frameDim.x * frameDim.y floats in
    /// row-major pixel order. ResourceBindFlags::Shared, so Buffer.to_torch()
    /// maps them as CUDA tensors in place.
    ref<Buffer> mpAOBuffer;          ///< ASSAOPass' AO, copied by forward().
    ref<Buffer> mpReferenceBuffer;   ///< The target, copied by captureTarget().
    ref<Buffer> mpDLdIBuffer;        ///< dL/dAO, written by the script.

    /// mpDLdIBuffer as the texture ASSAOPass::backward() takes.
    ref<Texture> mpDLdITexture;

    ref<ComputePass> mpTextureToBufferPass;
    ref<ComputePass> mpBufferToTexturePass;

    // ------------------------------------------------------------------
    // Background display
    // ------------------------------------------------------------------

    ref<FullScreenPass> mpViewPass;   ///< Backgrounds: AO / reference / error.

    /// "Full red" threshold for the error map. A fixed value rather than a
    /// per-iteration normalization, so colour is comparable across iterations —
    /// which is the point of watching the map while the run converges.
    ///
    /// 0.25 measured on Kroken: at the initial guess |AO - reference| has a
    /// median of 0.05 and a 99th percentile of 0.29, so this saturates about
    /// 1.6% of pixels. Lower and the map clips the regions that carry the
    /// signal; much higher and everything reads as near-white.
    float mErrorMapMaxVal = 0.25f;

    // ------------------------------------------------------------------
    // GUI state machine
    // ------------------------------------------------------------------

    Phase          mPhase          = Phase::Setup;
    BackgroundView mBackgroundView = BackgroundView::AO;

    // ---- Setup phase ----

    std::filesystem::path mScriptPath;
    char                  mScriptPathBuf[2048] = {};
    OptimizerConfig       mConfig;

    /// Target and init parameters in the units SSAOPass' own UI edits, so the
    /// sliders read like the pass the experiment mimics (radius ~50, bias ~1).
    /// Converted to shader units once, on Start.
    UIParams mTargetUI;
    UIParams mInitUI;

    /// Set when --target-params / --init-params supplied the pair, so loadScene()
    /// knows not to overwrite them with the compiled-in defaults.
    bool mTargetUIFromCommandLine = false;
    bool mInitUIFromCommandLine   = false;

    // ---- Running phase ----

    pybind11::object mGenerator;    ///< Python generator from optimize(cfg).
    bool             mGeneratorDone = false;
    int              mCurrentIt     = 0;
    int              mTotalIts      = 0;

    /// Per-iteration records accumulated during the run.
    std::vector<IterRecord> mHistory;

    // ---- Results phase ----

    int  mViewIt   = 0;    ///< Currently displayed iteration (scrub slider).
    bool mLogScale = true; ///< Loss curve y-axis; the loss spans orders of magnitude.

    GradCheckResult mGradCheck;

    /// Relative step for the numeric side of the gradient check: h is this times
    /// |param|, so one setting spans five parameters whose magnitudes differ by
    /// orders of magnitude.
    ///
    /// 0.01 measured on Kroken, where intensity then agrees to 0.8% and bias to
    /// 2.5%. Smaller steps sit under the float32 noise of a loss accumulated
    /// over two million pixels through atomics; larger ones hit truncation
    /// error. radius is unreliable at every step (27% at best) because it picks
    /// an HZB mip through floor(log2), so the loss is only piecewise smooth in
    /// it and a central difference straddles a kink — its analytic gradient is
    /// stable to six digits across a 1000x range of h, and that is the side to
    /// believe.
    float mGradCheckRelStep = 0.01f;

    // ------------------------------------------------------------------
    // Headless / command-line mode (unchanged from original)
    // ------------------------------------------------------------------

    std::vector<std::string> mScriptArgs;
    bool                     mHeadless  = false;
    uint32_t                 mFrameCount = 0;
    bool                     mScriptRun  = false;

    int    mArgc = 0;
    char** mArgv = nullptr;

    static constexpr const char* kKrokenScenePath = "kroken/camera-1.pbrt";
    static constexpr const char* kEnvMapPath      = "env/Studio_Soft.hdr";

    /// Shader path for the background view pixel shader.
    static constexpr const char* kViewShader = "Samples/SSAOOptimizer/View.ps.slang";

    /// Shader path for the texture <-> buffer copies.
    static constexpr const char* kCopyShader = "Samples/SSAOOptimizer/TextureBufferCopy.cs.slang";
};
