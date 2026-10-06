/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived from
 #    this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 # LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#include "Falcor.h"
#include "TAAPass.h"

namespace
{
const std::string kProgramFile = "Arctic/TAAPass/TAAPass.cs.slang";

const ChannelList kInputs = {
    { "colorIn", "gColor", "Current-frame HDR color (RGBA32Float)", true, ResourceFormat::Unknown },
    { "motionVecs", "gMotionVecs", "Per-pixel motion vector (RG16Float)", true, ResourceFormat::Unknown },
    { "sceneDepth", "gSceneDepth", "Scene depth (device Z, 0=near) for velocity dilation", true, ResourceFormat::Unknown },
};
const ChannelList kOutputs = {
    { "colorOut", "gColorOut", "Temporally resolved HDR color (RGBA32Float)", false, ResourceFormat::RGBA32Float },
};

const char kEnabled[] = "enabled";
const char kCurrentFrameWeight[] = "currentFrameWeight";
const char kExposureScale[] = "exposureScale";
} // namespace

TAAPass::TAAPass(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    Sampler::Desc samplerDesc;
    samplerDesc.setFilterMode(TextureFilteringMode::Linear, TextureFilteringMode::Linear, TextureFilteringMode::Linear);
    samplerDesc.setAddressingMode(TextureAddressingMode::Clamp, TextureAddressingMode::Clamp, TextureAddressingMode::Clamp);
    mpLinearSampler = mpDevice->createSampler(samplerDesc);

    for (const auto& [key, value] : props)
    {
        if (key == kEnabled) mEnabled = value;
        else if (key == kCurrentFrameWeight) mCurrentFrameWeight = value;
        else if (key == kExposureScale) mExposureScale = value;
        else logWarning("Unknown property '{}' in a TAAPass properties.", key);
    }
}

Properties TAAPass::getProperties() const
{
    Properties props;
    props[kEnabled] = mEnabled;
    props[kCurrentFrameWeight] = mCurrentFrameWeight;
    props[kExposureScale] = mExposureScale;
    return props;
}

RenderPassReflection TAAPass::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    const uint2 sz = compileData.defaultTexDims;
    addRenderPassInputs(reflector, kInputs, ResourceBindFlags::ShaderResource, uint2(0));
    addRenderPassOutputs(reflector, kOutputs, ResourceBindFlags::UnorderedAccess, sz);
    return reflector;
}

void TAAPass::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
    mpPass.reset();
    mpPrevColor.reset();
    mHasHistory = false;
    mDirty = true;
}

void TAAPass::rebuildPass()
{
    if (!mpScene) return;
    if (!mpPass)
    {
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kProgramFile).csEntry("csMain");
        desc.addTypeConformances(mpScene->getTypeConformances());
        mpPass = ComputePass::create(mpDevice, desc, mpScene->getSceneDefines());
    }
    mDirty = false;
}

void TAAPass::allocatePrevColor(const Texture* pColorOut)
{
    bool allocate = mpPrevColor == nullptr;
    allocate = allocate || (mpPrevColor->getWidth() != pColorOut->getWidth());
    allocate = allocate || (mpPrevColor->getHeight() != pColorOut->getHeight());
    // The history must be updated with a blit (SRV->RTV), which requires the
    // destination to be a render target. CopyResource would need an exact
    // format match with the (RGBA32Float) graph output, so we match the format
    // too and use blit for the per-frame history copy.
    allocate = allocate || (mpPrevColor->getFormat() != pColorOut->getFormat());

    if (allocate)
    {
        mpPrevColor = mpDevice->createTexture2D(
            pColorOut->getWidth(),
            pColorOut->getHeight(),
            pColorOut->getFormat(),
            1,
            1,
            nullptr,
            ResourceBindFlags::RenderTarget | ResourceBindFlags::ShaderResource
        );
        mHasHistory = false; // first frame after (re)alloc has no valid history
    }
}

void TAAPass::renderUI(Gui::Widgets& widget)
{
    // NOTE: the on/off toggle lives in the host's "Temporal AA" UI group
    // (pushed into this pass via setEnabled() every frame). Do NOT add a second
    // "Enable" checkbox here — it would be overwritten each frame and appear
    // un-uncheckable.
    mDirty |= widget.var("Current Frame Weight", mCurrentFrameWeight, 0.02f, 0.3f, 0.001f);
    widget.tooltip("Baseline temporal blend factor (UE CurrentFrameWeight). Lower = more stable, slower to converge.");
    mDirty |= widget.var("Exposure Scale", mExposureScale, 0.01f, 100.0f, 0.01f);
    widget.tooltip(
        "Scales luma before the HDR weighting (UE FrameExposureScale * OneOverPreExposure). The weight is "
        "1/(Y*scale + 1), so this sets where the firefly-suppression knee sits: roughly 1 / (scene mid-grey "
        "radiance). Too low and the HDR weighting does nothing; too high and highlights lose energy.",
        true
    );
}

void TAAPass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    auto pColorIn = renderData.getTexture("colorIn");
    auto pColorOut = renderData.getTexture("colorOut");
    auto pMotionVecs = renderData.getTexture("motionVecs");
    auto pSceneDepth = renderData.getTexture("sceneDepth");
    if (!pColorIn || !pColorOut || !pMotionVecs || !pSceneDepth) return;

    // Pass-through when disabled: no history update so toggling back on resumes
    // from a (stale) history; the next first-frame reset will clean it up anyway.
    if (!mEnabled)
    {
        pRenderContext->blit(pColorIn->getSRV(), pColorOut->getRTV());
        return;
    }

    if (mDirty) rebuildPass();
    if (!mpPass) return;

    allocatePrevColor(pColorOut.get());

    const uint32_t w = pColorOut->getWidth();
    const uint32_t h = pColorOut->getHeight();

    // This frame's sub-pixel sample offset, relative to the UN-jittered pixel
    // grid, in pixels with +x right and +y down. Falcor's jitterX/jitterY are
    // sub-pixel offsets already divided by the render dimensions, and the jitter
    // is applied as a clip-space translation of +2*jitter, so a world point's
    // position moves by (+jitterX*W, -jitterY*H) pixels when jitter is applied.
    // The sample rasterized AT a pixel centre therefore represents the scene at
    // (centre + (-jitterX*W, +jitterY*H)) in the un-jittered grid -- that is the
    // offset the reconstruction filter needs.
    //
    // Cross-check: this is exactly the negation of the (+jitterX, -jitterY) term
    // MotionVectorPass adds to move the history lookup onto the output pixel
    // centre, so the two passes agree on the convention by construction.
    float2 jitterPixels = float2(0.f, 0.f);
    if (mpScene)
    {
        const auto cam = mpScene->getCamera();
        jitterPixels = float2(-cam->getJitterX() * float(w), cam->getJitterY() * float(h));
    }

    auto var = mpPass->getVars()->getRootVar();
    var["gColor"] = pColorIn;
    var["gMotionVecs"] = pMotionVecs;
    var["gSceneDepth"] = pSceneDepth;
    var["gPrevColor"] = mpPrevColor;
    var["gLinearSampler"] = mpLinearSampler;
    var["gColorOut"] = pColorOut;

    auto cb = var["PerFrameCB"];
    cb["gFrameDim"] = uint2(w, h);
    cb["gJitterPixels"] = jitterPixels;
    cb["gCurrentFrameWeight"] = mCurrentFrameWeight;
    cb["gExposureScale"] = mExposureScale;
    cb["gHasHistory"] = mHasHistory ? 1u : 0u;
    // UE drives bCameraCut off real view cuts only, never off camera speed. Here
    // the only intrinsic cut is "no valid history yet" (first frame / resize /
    // setScene); requestCameraCut() covers explicit cuts.
    cb["gCameraCut"] = mHasHistory ? 0u : 1u;

    mpPass->execute(pRenderContext, uint3(w, h, 1));

    // Advance history: copy the just-written output into the history texture.
    pRenderContext->blit(pColorOut->getSRV(), mpPrevColor->getRTV());
    mHasHistory = true;
}
