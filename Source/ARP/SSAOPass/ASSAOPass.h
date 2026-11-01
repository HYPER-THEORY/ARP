/// Copyright (C) 2026-2027 ARP (Auto Render Pipeline). All Rights Reserved.

#pragma once

#include "Falcor.h"
#include "RenderGraph/RenderPass.h"

using namespace Falcor;

/** Reverse-mode companion of SSAOPass.

    ASSAOPass evaluates the differentiable SSAO integral from ASSAO.cs.slang as a
    differentiable image operator:

      - forward() renders AO at the current parameters into the graph's `aoMap`
        output;
      - backward() takes dL/dAO as a texture and writes dL/dparam for the four
        learnable parameters into getParamGrads().

    The pass has no notion of a loss or a target. Whatever consumes the AO image
    computes the loss and hands back its derivative, which is what lets the pass
    sit inside a larger differentiable pipeline. SSAOOptimizer does this in
    PyTorch.

    backward() is called by the caller rather than from execute(): its input is
    produced downstream of this pass, and wiring it back into the graph would
    make a cycle.

    Only the AO image is differentiated. The G-buffer, scene depth and HZB are
    treated as constants, which is what makes this a screen-space backward pass:
    the geometry is fixed, the four learnable SSAO parameters are not. The AO
    computation mirrors SSAOPass' differentiable mode, including its HZB sampling
    and early-outs.

    Parameter units
    ---------------
    Properties uses the same UI-facing keys and units as SSAOPass. The values
    are folded into shader units only when the constant buffer is bound.

    Interop
    -------
    The gradient buffer is created with ResourceBindFlags::Shared, so on a
    CUDA-enabled build it surfaces to Python as a zero-copy CUDA tensor through
    Buffer.to_torch(). The AO image and dL/dAO are plain textures; a caller that
    wants them in torch copies them through shared buffers of its own, as
    SSAOOptimizer does.
*/
class ASSAOPass : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(ASSAOPass, "ASSAOPass", "Differentiable SSAO (reverse-mode companion of SSAOPass)");

    static ref<ASSAOPass> create(ref<Device> pDevice, const Properties& props);

    ASSAOPass(ref<Device> pDevice, const Properties& props);

    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void compile(RenderContext* pRenderContext, const CompileData& compileData) override;
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;

    virtual void setProperties(const Properties& props) override;
    virtual Properties getProperties() const override;

    virtual void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;

    /// dL/dparam, four floats in intensity/power/radius/bias order. Filled by
    /// backward(). Shared with CUDA.
    ref<Buffer> getParamGrads() const { return mpParamGrads; }

    /// Dispatch the primal pass at the current parameters.
    void forward(RenderContext* pRenderContext);

    /// The reverse pass. `pdLdI` holds dL/dAO per pixel, sized like the AO output,
    /// read from its first channel; the result, sum_pixels dLdI * dAO/dparam,
    /// overwrites getParamGrads(), so no clear is needed between calls.
    ///
    /// The integral is re-evaluated at the current parameters, so they and the
    /// temporal jitter must be the ones the forward that produced `pdLdI` ran
    /// at. Returns without dispatching while the pass has no inputs bound yet,
    /// as forward() does; a missing or wrongly sized `pdLdI` throws.
    void backward(RenderContext* pRenderContext, const ref<Texture>& pdLdI);

private:
    static constexpr uint32_t kGradientCount = 4;

    /// Threads per reversing group, i.e. the shared-memory reduction width. Keep
    /// in step with kGroupDimX / kGroupDimY in ASSAO.cs.slang, which size the
    /// per-group partial buffer this pass allocates.
    static constexpr uint32_t kGroupDimX = 8;
    static constexpr uint32_t kGroupDimY = 8;
    static constexpr uint32_t kGroupSize = kGroupDimX * kGroupDimY;

    void compilePasses();
    void createBuffers();
    void bindShaderData(const ShaderVar& var);

    /// The two dispatches of backward(): one gradient per thread group into the
    /// partial buffer, then their sum into getParamGrads().
    void dispatchBackward(RenderContext* pRenderContext, const ref<Texture>& pdLdI);
    void dispatchReduction(RenderContext* pRenderContext);

    uint32_t mFrameIndex = 0;
    uint2 mFrameDim = { 0, 0 };

    /// Thread groups per axis in the reverse pass, i.e. ceil(mFrameDim / kGroupDim).
    /// Sizes the partial buffer and bounds the reduction's stride.
    uint2 mGroupCount = { 0, 0 };

    float mIntensity = 0.5f;
    float mPower = 2.0f;
    float mRadius = 100.0f;
    float mBias = 3.0f;
    float mFadeDistance = 8000.0f;
    float mFadeRadius = 5000.0f;
    float mMipLevelFactor = 0.5f;

    bool mRadiusInWorldSpace = false;
    bool mLockedFrame = true;

    ref<Scene> mpScene;

    ref<ComputePass> mpForwardPass;
    ref<ComputePass> mpBackwardPass;
    ref<ComputePass> mpReducePass;

    ref<Buffer> mpParamGrads;
    ref<Buffer> mpGradPartials;

    ref<Texture> mpRandomNormalMap;

    ref<Sampler> mpRandomNormalSampler;

    /// Geometry bound by the graph, so forward()/backward() can be driven
    /// directly without re-running the graph.
    ref<Texture> mpGBufferA;
    ref<Texture> mpSceneDepth;
    ref<Texture> mpZFurthest;
    ref<Texture> mpAOMapOutput;
};
