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
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#include "Falcor.h"
#include "RenderGraph/RenderPassStandardFlags.h"
#include "ShadowProjectionPass.h"
#include "Scene/Lights/Light.h"

namespace
{
const std::string kShadowProjectionProgramFile = "Arctic/ShadowProjectionPass/ShadowProjectionPass.3d.slang";

const ChannelList kInputs = {
    { "sceneDepth", "gSceneDepth", "GBuffer scene depth (D32Float)", true, ResourceFormat::Unknown },
    { "shadowDepth", "gShadowDepth", "Light-space linearized shadow depth (R32Float)", true, ResourceFormat::Unknown },
    { "gBufferA", "gGBufferA", "GBuffer world normal (RGB10A2Unorm, N*0.5+0.5)", true, ResourceFormat::Unknown },
    { "gBufferB", "gGBufferB", "GBuffer metallic/spec/rough/ShadingModelID", true, ResourceFormat::Unknown },
    { "gBufferD", "gGBufferD", "GBuffer custom data (.a = UE material Opacity, drives SSS thickness)", true, ResourceFormat::Unknown },
};
const ChannelList kOutputs = {
    { "shadowMask", "gShadowMask", "Screen shadow (RG32Float: .r surface, .g SSS transmission)", false, ResourceFormat::RG32Float },
};

const char kConstantDepthBias[] = "constantDepthBias";
const char kReceiverBias[] = "receiverBias";
const char kSoftTransitionScale[] = "softTransitionScale";
const char kShadowSharpen[] = "shadowSharpen";
const char kFilterMode[] = "filterMode";
const char kPCFKernelSize[] = "pcfKernelSize";
const char kFilterRadiusWorld[] = "filterRadiusWorld";
const char kDiskTapCount[] = "diskTapCount";
const char kDensityDepthScale[] = "densityDepthScale";
} // namespace

ShadowProjectionPass::ShadowProjectionPass(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    parseProperties(props);

    // Point sampler with BORDER addressing and a border value far beyond any
    // depth ShadowPass can write.
    //
    // The addressing mode is load-bearing. The shadowmap border sits right on
    // real geometry — with clamp-to-edge, filter taps that reach past the
    // frustum would smear that boundary geometry outward and cast a spurious
    // band of shadow along the scene silhouette. Because the frustum encloses
    // the whole scene, "outside the frustum" genuinely means "no occluder", and
    // a huge border depth makes calcOcclusion4's unwritten test report exactly
    // that. This also lets the Gather4 path skip manual texel clamping.
    Sampler::Desc samplerDesc;
    samplerDesc.setFilterMode(TextureFilteringMode::Point, TextureFilteringMode::Point, TextureFilteringMode::Point);
    samplerDesc.setAddressingMode(TextureAddressingMode::Border, TextureAddressingMode::Border, TextureAddressingMode::Border);
    samplerDesc.setBorderColor(float4(1e9f));
    mpShadowSampler = mpDevice->createSampler(samplerDesc);
}

void ShadowProjectionPass::parseProperties(const Properties& props)
{
    for (const auto& [key, value] : props)
    {
        if (key == kConstantDepthBias) mConstantDepthBias = value;
        else if (key == kReceiverBias) mReceiverBias = value;
        else if (key == kSoftTransitionScale) mSoftTransitionScale = value;
        else if (key == kShadowSharpen) mShadowSharpen = value;
        else if (key == kFilterMode) mFilterMode = FilterMode(uint32_t(value));
        else if (key == kPCFKernelSize) mPCFKernelSize = value;
        else if (key == kFilterRadiusWorld) mFilterRadiusWorld = value;
        else if (key == kDiskTapCount) mDiskTapCount = value;
        else if (key == kDensityDepthScale) mDensityDepthScale = value;
    }
}

Properties ShadowProjectionPass::getProperties() const
{
    Properties props;
    props[kConstantDepthBias] = mConstantDepthBias;
    props[kReceiverBias] = mReceiverBias;
    props[kSoftTransitionScale] = mSoftTransitionScale;
    props[kShadowSharpen] = mShadowSharpen;
    props[kFilterMode] = uint32_t(mFilterMode);
    props[kPCFKernelSize] = mPCFKernelSize;
    props[kFilterRadiusWorld] = mFilterRadiusWorld;
    props[kDiskTapCount] = mDiskTapCount;
    props[kDensityDepthScale] = mDensityDepthScale;
    return props;
}

RenderPassReflection ShadowProjectionPass::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    // Inputs use auto dimensions (0,0) so they inherit the size of whatever
    // they are connected to (GBuffer depth = screen size, shadow depth = shadow res).
    addRenderPassInputs(reflector, kInputs, ResourceBindFlags::ShaderResource, uint2(0));
    const uint2 sz = compileData.defaultTexDims;
    addRenderPassOutputs(reflector, kOutputs, ResourceBindFlags::UnorderedAccess, sz);
    return reflector;
}

void ShadowProjectionPass::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mUpdateFlagsConnection = {};
    mUpdateFlags = IScene::UpdateFlags::None;
    mpScene = pScene;
    pPass.reset();
    mFrameIndex = 0;
    if (mpScene)
    {
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kShadowProjectionProgramFile).csEntry("csMain");
        desc.addTypeConformances(mpScene->getTypeConformances());
        pPass = ComputePass::create(mpDevice, desc, mpScene->getSceneDefines());
        mUpdateFlagsConnection = mpScene->getUpdateFlagsSignal().connect([&](IScene::UpdateFlags flags) { mUpdateFlags |= flags; });
    }
}

void ShadowProjectionPass::renderUI(Gui::Widgets& widget)
{
    Gui::DropdownList modeList;
    modeList.push_back({uint32_t(FilterMode::AxisAlignedPCF), "Axis-aligned PCF (Gather4)"});
    modeList.push_back({uint32_t(FilterMode::SobolDisk), "Sobol disk (rotated, TAA)"});
    uint32_t mode = uint32_t(mFilterMode);
    if (widget.dropdown("Filter mode", modeList, mode))
    {
        mFilterMode = FilterMode(mode);
        mOptionsChanged = true;
    }
    widget.tooltip(
        "Axis-aligned PCF: texel-for-texel port of UE ManualPCF, accelerated with Gather4 (4 taps per\n"
        "texture op). Temporally stable on its own.\n\n"
        "Sobol disk: sparse golden-angle disk, re-rotated per pixel and per frame from UE's Sobol\n"
        "sequence, each tap a bilinear 2x2 PCF. Much wider and smoother penumbra for fewer taps, but\n"
        "it needs TAAPass to resolve — with TAA off you will see per-pixel noise.");

    if (mFilterMode == FilterMode::AxisAlignedPCF)
    {
        Gui::DropdownList kernelList;
        kernelList.push_back({1u, "1x1"});
        kernelList.push_back({3u, "3x3"});
        kernelList.push_back({5u, "5x5"});
        kernelList.push_back({7u, "7x7"});
        if (widget.dropdown("PCF kernel", kernelList, mPCFKernelSize)) mOptionsChanged = true;
        widget.tooltip(
            "Kernel width in shadowmap texels. Costs ((N+1)/2)^2 Gather4 ops: 7x7 = 16.\n"
            "1x1 is UE's Manual1x1PCF (bilinear 2x2 PCF), not a single unfiltered tap.");
    }
    else
    {
        mOptionsChanged |= widget.var("Filter radius (m)", mFilterRadiusWorld, 0.005f, 5.f, 0.005f);
        widget.tooltip(
            "Disk radius in WORLD meters, so the penumbra width does not change when the shadow\n"
            "resolution or the fitted light frustum changes.\n\n"
            "For reference, a physically-sized sun penumbra is about 0.0093 * (occluder distance),\n"
            "i.e. far below one shadowmap texel for typical scene scales. The useful\n"
            "range here is therefore set by hiding texel stair-stepping, not by the light's angular\n"
            "size. Much larger and contact shadows wash out, and the\n"
            "taps become too sparse for the tap count to resolve.");

        Gui::DropdownList tapList;
        tapList.push_back({4u, "4 taps"});
        tapList.push_back({8u, "8 taps"});
        tapList.push_back({12u, "12 taps"});
        tapList.push_back({16u, "16 taps"});
        tapList.push_back({24u, "24 taps"});
        if (widget.dropdown("Disk taps", tapList, mDiskTapCount)) mOptionsChanged = true;
    }

    widget.separator();

    // Ranges in METERS (shadowmap stores light-space linear depth in meters).
    // NOTE: UE's equivalents are dimensionless, normalized over the cascade
    // depth range, so UE's numeric defaults are not transferable here.
    mOptionsChanged |= widget.var("Constant bias (m)", mConstantDepthBias, 0.f, 1.f, 0.001f);
    widget.tooltip("Should match ShadowPass's constant bias. Only used to nudge the receiver on the foliage transmission path.");
    mOptionsChanged |= widget.var("Receiver bias (m)", mReceiverBias, 0.f, 5.f, 0.01f);
    mOptionsChanged |= widget.var("Soft transition scale (1/m)", mSoftTransitionScale, 0.1f, 200.f, 0.1f);
    mOptionsChanged |= widget.var("Shadow sharpen", mShadowSharpen, 0.f, 8.f, 0.05f);
    widget.tooltip("UE ShadowSharpen: expands contrast about 0.5 after filtering. 1 = off.");
    // Scales subsurfaceDensityFromOpacity output into DensityMulConstant
    // (UE ProjectionDepthBiasParameters.w). Higher -> foliage transmits less
    // through stacked leaves.
    mOptionsChanged |= widget.var("Density depth scale (1/m)", mDensityDepthScale, 0.f, 2000.f, 1.f);
}

void ShadowProjectionPass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (mOptionsChanged)
    {
        auto& dict = renderData.getDictionary();
        auto flags = dict.getValue(kRenderPassRefreshFlags, RenderPassRefreshFlags::None);
        dict[Falcor::kRenderPassRefreshFlags] = flags | Falcor::RenderPassRefreshFlags::RenderOptionsChanged;
        mOptionsChanged = false;
    }

    auto pSceneDepth = renderData.getTexture("sceneDepth");
    auto pShadowDepth = renderData.getTexture("shadowDepth");
    auto pGBufferA = renderData.getTexture("gBufferA");
    auto pGBufferB = renderData.getTexture("gBufferB");
    auto pGBufferD = renderData.getTexture("gBufferD");
    auto pShadowMask = renderData.getTexture("shadowMask");
    if (!pSceneDepth || !pShadowDepth || !pGBufferA || !pGBufferB || !pGBufferD || !pShadowMask || !pPass) return;

    // Read the light view parameters ShadowPass published this frame. They are
    // never recomputed here: two independent copies of the frustum fit would
    // silently diverge into a whole-screen shadow offset that is very hard to
    // distinguish from a bias or filter-centering problem.
    mShadowView = renderData.getDictionary().getValue(kShadowViewDataKey, ShadowViewData{});
    if (!mShadowView.valid)
    {
        // No directional light / no valid scene bounds: everything is lit.
        pRenderContext->clearUAV(pShadowMask->getUAV().get(), float4(1.f));
        return;
    }

    auto var = pPass->getVars()->getRootVar();
    // Bind the scene parameter block. The shader uses gScene.camera.data.invViewProj
    // to reconstruct world-space positions from GBuffer depth.
    if (mpScene) mpScene->bindShaderData(var["gScene"]);
    var["gSceneDepth"] = pSceneDepth;
    var["gShadowDepth"] = pShadowDepth;
    var["gGBufferA"] = pGBufferA;
    var["gGBufferB"] = pGBufferB;
    var["gGBufferD"] = pGBufferD;
    var["gShadowMask"] = pShadowMask;
    var["gShadowSampler"] = mpShadowSampler;

    const uint2 shadowDim = uint2(pShadowDepth->getWidth(), pShadowDepth->getHeight());

    var["PerFrameCB"]["gLightViewProj"] = mShadowView.lightViewProj;
    var["PerFrameCB"]["gLightPos"] = mShadowView.lightPos;
    var["PerFrameCB"]["gLightDir"] = mShadowView.lightDir;
    var["PerFrameCB"]["gMaxSubjectDepth"] = mShadowView.maxSubjectDepth;
    var["PerFrameCB"]["gConstantDepthBias"] = mConstantDepthBias;
    var["PerFrameCB"]["gReceiverBias"] = mReceiverBias;
    var["PerFrameCB"]["gSoftTransitionScale"] = mSoftTransitionScale;
    var["PerFrameCB"]["gDensityDepthScale"] = mDensityDepthScale;
    var["PerFrameCB"]["gShadowSharpen"] = mShadowSharpen;
    var["PerFrameCB"]["gFilterRadiusWorld"] = mFilterRadiusWorld;
    // World meters -> shadowmap UV. The frustum spans 2*halfExtent world units
    // across the full [0,1] UV range, and the two axes differ whenever the
    // fitted frustum is non-square, which is what keeps the disk circular in
    // world space rather than in UV space.
    var["PerFrameCB"]["gShadowUVPerMeter"] = 0.5f / mShadowView.orthoHalfExtent;
    var["PerFrameCB"]["gFrameDim"] = uint2(pShadowMask->getWidth(), pShadowMask->getHeight());
    var["PerFrameCB"]["gShadowDim"] = shadowDim;
    var["PerFrameCB"]["gInvShadowDim"] = 1.f / float2(shadowDim);
    var["PerFrameCB"]["gFrameIndex"] = mFrameIndex;

    pPass->addDefine("SHADOW_FILTER_MODE", std::to_string(uint32_t(mFilterMode)));
    pPass->addDefine("PCF_KERNEL_SIZE", std::to_string(mPCFKernelSize));
    pPass->addDefine("DISK_TAP_COUNT", std::to_string(mDiskTapCount));

    pPass->execute(pRenderContext, uint3(pShadowMask->getWidth(), pShadowMask->getHeight(), 1));

    ++mFrameIndex;
}
