/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the copyright
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
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#include "Falcor.h"
#include "GTAOPass.h"

namespace
{
const std::string kSearchProgramFile = "Arctic/GTAOPass/GTAOPass.cs.slang";
const std::string kSpatialProgramFile = "Arctic/GTAOPass/GTAOSpatialFilter.cs.slang";
const std::string kTemporalProgramFile = "Arctic/GTAOPass/GTAOTemporalFilter.cs.slang";
const std::string kUpsampleProgramFile = "Arctic/GTAOPass/GTAOUpsample.cs.slang";

const ChannelList kInputs = {
    { "gBufferA", "gGBufferA", "World normal (encoded N*0.5+0.5) + per-object", true, ResourceFormat::Unknown },
    { "sceneDepth", "gSceneDepth", "GBuffer scene depth (D32Float)", true, ResourceFormat::Unknown },
    { "motionVecs", "gMotionVecs", "Screen-space motion vectors (RG16Float)", true, ResourceFormat::Unknown },
};
const ChannelList kOutputs = {
    { "aoMap", "gAOMap", "Full-res AO (R8Unorm), 0=occluded 1=open", false, ResourceFormat::R8Unorm },
};

// UE's temporal sequence (PostProcessAmbientOcclusion.cpp). The rotations are
// scaled by PI/360, not PI/180, so the effective angles are half the listed
// values: {30, 150, 90, 120, 60, 0} degrees — a 6-frame low-discrepancy sweep of
// [0, 180), which is all that is needed since a slice and its opposite are the
// same slice. The spatial offset advances only once per full rotation cycle, so
// the two sequences stay decorrelated over 24 frames.
const float kTemporalRotations[6] = { 60.f, 300.f, 180.f, 240.f, 120.f, 0.f };
const float kSpatialOffsets[4] = { 0.f, 0.5f, 0.25f, 0.75f };

const char kEffectRadius[] = "effectRadius";
const char kThickness[] = "thickness";
const char kNumAngles[] = "numAngles";
const char kPower[] = "power";
const char kIntensity[] = "intensity";
const char kFadeDistance[] = "fadeDistance";
const char kFadeRadius[] = "fadeRadius";
const char kMaxDistance[] = "maxDistance";
const char kSpatialFilterWeight[] = "spatialFilterWeight";
const char kTemporalBlendWeight[] = "temporalBlendWeight";
const char kEnableTemporal[] = "enableTemporal";
const char kOutputStage[] = "outputStage";
} // namespace

GTAOPass::GTAOPass(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    Sampler::Desc samplerDesc;
    samplerDesc.setFilterMode(TextureFilteringMode::Linear, TextureFilteringMode::Linear, TextureFilteringMode::Point);
    samplerDesc.setAddressingMode(TextureAddressingMode::Clamp, TextureAddressingMode::Clamp, TextureAddressingMode::Clamp);
    mpLinearSampler = mpDevice->createSampler(samplerDesc);

    for (const auto& [key, value] : props)
    {
        if (key == kEffectRadius) mEffectRadius = value;
        else if (key == kThickness) mThickness = value;
        else if (key == kNumAngles) mNumAngles = value;
        else if (key == kPower) mPower = value;
        else if (key == kIntensity) mIntensity = value;
        else if (key == kFadeDistance) mFadeDistance = value;
        else if (key == kFadeRadius) mFadeRadius = value;
        else if (key == kMaxDistance) mMaxDistance = value;
        else if (key == kSpatialFilterWeight) mSpatialFilterWeight = value;
        else if (key == kTemporalBlendWeight) mTemporalBlendWeight = value;
        else if (key == kEnableTemporal) mEnableTemporal = value;
        else if (key == kOutputStage) mOutputStage = (OutputStage)(uint32_t)value;
        else logWarning("Unknown property '{}' in GTAOPass properties.", key);
    }
}

Properties GTAOPass::getProperties() const
{
    Properties props;
    props[kEffectRadius] = mEffectRadius;
    props[kThickness] = mThickness;
    props[kNumAngles] = mNumAngles;
    props[kPower] = mPower;
    props[kIntensity] = mIntensity;
    props[kFadeDistance] = mFadeDistance;
    props[kFadeRadius] = mFadeRadius;
    props[kMaxDistance] = mMaxDistance;
    props[kSpatialFilterWeight] = mSpatialFilterWeight;
    props[kTemporalBlendWeight] = mTemporalBlendWeight;
    props[kEnableTemporal] = mEnableTemporal;
    props[kOutputStage] = (uint32_t)mOutputStage;
    return props;
}

RenderPassReflection GTAOPass::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    const uint2 sz = compileData.defaultTexDims;
    addRenderPassInputs(reflector, kInputs, ResourceBindFlags::ShaderResource, uint2(0));
    addRenderPassOutputs(reflector, kOutputs, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource, sz);
    return reflector;
}

void GTAOPass::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
    mpSearchPass.reset();
    mpSpatialPass.reset();
    mpTemporalPass.reset();
    mpUpsamplePass.reset();
    mHasHistory = false;
    mFrameIndex = 0;
    mDirty = true;
}

void GTAOPass::rebuildPasses()
{
    if (!mpScene) return;

    auto makePass = [this](const std::string& file) {
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(file).csEntry("csMain");
        desc.addTypeConformances(mpScene->getTypeConformances());
        return ComputePass::create(mpDevice, desc, mpScene->getSceneDefines());
    };

    if (!mpSearchPass) mpSearchPass = makePass(kSearchProgramFile);
    if (!mpSpatialPass) mpSpatialPass = makePass(kSpatialProgramFile);
    if (!mpTemporalPass) mpTemporalPass = makePass(kTemporalProgramFile);
    if (!mpUpsamplePass) mpUpsamplePass = makePass(kUpsampleProgramFile);
    mDirty = false;
}

void GTAOPass::allocateResources(uint32_t fullW, uint32_t fullH)
{
    if (mAllocDim.x == fullW && mAllocDim.y == fullH) return;

    const uint32_t halfW = (fullW + 1) / 2;
    const uint32_t halfH = (fullH + 1) / 2;
    const auto flags = ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource;

    auto makeHalf = [&](const char* name) {
        auto pTex = mpDevice->createTexture2D(halfW, halfH, ResourceFormat::R8Unorm, 1u, 1u, nullptr, flags);
        pTex->setName(name);
        return pTex;
    };

    mpHalfAO = makeHalf("GTAO.halfAO");
    mpHalfAOFiltered = makeHalf("GTAO.halfAOFiltered");
    mpHalfHistory[0] = makeHalf("GTAO.halfHistory0");
    mpHalfHistory[1] = makeHalf("GTAO.halfHistory1");

    mAllocDim = uint2(fullW, fullH);
    mHistoryIdx = 0;
    mHasHistory = false;
}

void GTAOPass::renderUI(Gui::Widgets& widget)
{
    widget.var("Effect Radius (world)", mEffectRadius, 0.05f, 100.f, 0.01f);
    widget.tooltip("World-space AO radius (UE GTAOParams[3].y / AmbientOcclusionRadius).");
    widget.var("Thickness", mThickness, 0.f, 1.f, 0.01f);
    widget.tooltip("UE r.GTAO.ThicknessBlend. Uploaded as clamp(1 - t*t, 0, 0.99). Higher = more occlusion retained "
                   "behind objects.");
    widget.var("Num Angles", mNumAngles, 1u, 16u);
    widget.tooltip("Slice directions per pixel (UE GTAOParams[4].y). Cost is linear in this.");

    if (auto g = widget.group("User adjust", true))
    {
        g.var("Power", mPower, 0.f, 8.f, 0.01f);
        g.tooltip("UE AmbientOcclusionPower. Halved in the shader.");
        g.var("Intensity", mIntensity, 0.f, 1.f, 0.01f);
        g.tooltip("UE AmbientOcclusionIntensity. 0 disables AO entirely (AO := 1).");
    }

    if (auto g = widget.group("Distance fade", true))
    {
        g.var("Fade Distance", mFadeDistance, 1.f, 20000.f, 1.f);
        g.tooltip("AO reaches 1 at this depth (UE AmbientOcclusionFadeDistance).");
        g.var("Fade Radius", mFadeRadius, 1.f, 20000.f, 1.f);
        g.tooltip("Width of the fade band ending at Fade Distance (UE AmbientOcclusionFadeRadius).");
        g.var("Max Distance", mMaxDistance, 1.f, 40000.f, 1.f);
        g.tooltip("Hard early-out (UE ScreenSpaceAOParams[4].w). Keep this >= Fade Distance, "
                  "otherwise it cuts in before the smooth fade has finished and you get a "
                  "visible edge.");
    }

    if (auto g = widget.group("Filters", true))
    {
        g.var("Spatial Filter Weight", mSpatialFilterWeight, 0.f, 200000.f, 100.f);
        g.tooltip("Bilateral depth tolerance of the 5x5 filter (UE hardcodes 20000).");
        g.checkbox("Enable Temporal Filter", mEnableTemporal);
        g.tooltip("Reprojected accumulation. Requires the motionVecs input.");
        if (mEnableTemporal)
        {
            g.var("Temporal Blend Weight", mTemporalBlendWeight, 0.01f, 1.f, 0.01f);
            g.tooltip("New-frame weight (UE GTAOParams[4].x). 1.0 = no accumulation.");
        }

        Gui::DropdownList stageList;
        stageList.push_back({(uint32_t)OutputStage::Search, "0: Horizon search (2/PI-scaled!)"});
        stageList.push_back({(uint32_t)OutputStage::Spatial, "1: + spatial filter"});
        stageList.push_back({(uint32_t)OutputStage::Temporal, "2: + temporal filter"});
        uint32_t stage = (uint32_t)mOutputStage;
        if (g.dropdown("Upsample source", stageList, stage)) mOutputStage = (OutputStage)stage;
        g.tooltip("Which stage the full-res upsample reads, for dumping one stage at a "
                  "time against UE. Stage 0 is NOT a usable AO value: it still carries "
                  "UE's 2/PI, so unoccluded reads 0.6366 there.");
    }
}

void GTAOPass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    auto pGBufferA = renderData.getTexture("gBufferA");
    auto pSceneDepth = renderData.getTexture("sceneDepth");
    auto pMotionVecs = renderData.getTexture("motionVecs");
    auto pAOMap = renderData.getTexture("aoMap");
    if (!pGBufferA || !pSceneDepth || !pAOMap || !mpScene) return;

    if (mDirty) rebuildPasses();
    if (!mpSearchPass || !mpSpatialPass || !mpTemporalPass || !mpUpsamplePass) return;

    const uint32_t fullW = pAOMap->getWidth();
    const uint32_t fullH = pAOMap->getHeight();
    const uint32_t halfW = (fullW + 1) / 2;
    const uint32_t halfH = (fullH + 1) / 2;
    allocateResources(fullW, fullH);

    // The temporal filter needs reprojection; without motion vectors there is
    // nothing to reproject with, so fall through to the spatial result.
    const bool useTemporal = mEnableTemporal && pMotionVecs != nullptr;
    if (mEnableTemporal && !pMotionVecs)
    {
        logWarning("GTAOPass: temporal filter enabled but the 'motionVecs' input is not connected; skipping it.");
    }

    // ---- 1. Horizon search + inner integrate (half res) ----
    {
        auto var = mpSearchPass->getVars()->getRootVar();
        mpScene->bindShaderData(var["gScene"]);
        var["gGBufferA"] = pGBufferA;
        var["gSceneDepth"] = pSceneDepth;
        var["gAOMapHalf"] = mpHalfAO;

        // UE maps the user-facing thickness through 1 - t^2 before upload.
        const float thicknessBlend = std::clamp(1.f - mThickness * mThickness, 0.f, 0.99f);
        const float deltaAngle = (float)M_PI / (float)std::max(mNumAngles, 1u);
        const float temporalAngle = kTemporalRotations[mFrameIndex % 6] * ((float)M_PI / 360.f);
        // saturate(depth * mul + add) == 0 at (fadeDistance - fadeRadius), 1 at fadeDistance.
        const float fadeRadius = std::max(1.f, mFadeRadius);
        const float fadeMul = 1.f / fadeRadius;
        const float fadeAdd = -(mFadeDistance - fadeRadius) * fadeMul;

        auto cb = var["PerFrameCB"];
        cb["gFullFrameDim"] = uint2(fullW, fullH);
        cb["gHalfFrameDim"] = uint2(halfW, halfH);
        cb["gEffectRadius"] = mEffectRadius;
        cb["gThicknessBlend"] = thicknessBlend;
        cb["gNumAngles"] = std::max(mNumAngles, 1u);
        cb["gSinDeltaAngle"] = std::sin(deltaAngle);
        cb["gCosDeltaAngle"] = std::cos(deltaAngle);
        cb["gTemporalCos"] = std::cos(temporalAngle);
        cb["gTemporalSin"] = std::sin(temporalAngle);
        cb["gTemporalOffset"] = kSpatialOffsets[(mFrameIndex / 6) % 4];
        cb["gFadeMul"] = fadeMul;
        cb["gFadeAdd"] = fadeAdd;
        cb["gMaxDistance"] = mMaxDistance;

        mpSearchPass->execute(pRenderContext, uint3(halfW, halfH, 1));
    }
    pRenderContext->uavBarrier(mpHalfAO.get());

    // ---- 2. 5x5 bilateral spatial filter (half res) ----
    {
        auto var = mpSpatialPass->getVars()->getRootVar();
        var["gAOMapHalf"] = mpHalfAO;
        var["gSceneDepth"] = pSceneDepth;
        var["gAOFilteredHalf"] = mpHalfAOFiltered;

        auto cb = var["PerFrameCB"];
        cb["gFullFrameDim"] = uint2(fullW, fullH);
        cb["gHalfFrameDim"] = uint2(halfW, halfH);
        cb["gDownsampleFactor"] = 2;
        cb["gFilterMin"] = -2;
        cb["gFilterMax"] = 2;
        cb["gSpatialFilterWeight"] = mSpatialFilterWeight;
        cb["gPower"] = mPower;
        cb["gIntensity"] = mIntensity;

        // Threadgroup is 16x8 (the LDS staging pattern depends on it).
        mpSpatialPass->execute(pRenderContext, uint3(halfW, halfH, 1));
    }
    pRenderContext->uavBarrier(mpHalfAOFiltered.get());

    // ---- 3. Reprojected temporal filter (half res, ping-pong) ----
    ref<Texture> pUpsampleSrc = mpHalfAOFiltered;
    const bool wantTemporal = useTemporal && mOutputStage == OutputStage::Temporal;
    if (wantTemporal)
    {
        const uint32_t prevIdx = mHistoryIdx;
        const uint32_t curIdx = 1u - mHistoryIdx;

        auto var = mpTemporalPass->getVars()->getRootVar();
        var["gAOFilteredHalf"] = mpHalfAOFiltered;
        var["gHistoryAO"] = mpHalfHistory[prevIdx];
        var["gMotionVecs"] = pMotionVecs;
        var["gAOTemporalHalf"] = mpHalfHistory[curIdx];

        auto cb = var["PerFrameCB"];
        cb["gFullFrameDim"] = uint2(fullW, fullH);
        cb["gHalfFrameDim"] = uint2(halfW, halfH);
        cb["gBlendWeight"] = mTemporalBlendWeight;
        cb["gHasHistory"] = mHasHistory ? 1u : 0u;

        mpTemporalPass->execute(pRenderContext, uint3(halfW, halfH, 1));
        pRenderContext->uavBarrier(mpHalfHistory[curIdx].get());

        pUpsampleSrc = mpHalfHistory[curIdx];
        mHistoryIdx = curIdx;
        mHasHistory = true;
    }
    else
    {
        mHasHistory = false;
    }
    if (mOutputStage == OutputStage::Search) pUpsampleSrc = mpHalfAO;

    // ---- 4. Conservative min-of-4 upsample (full res) ----
    {
        auto var = mpUpsamplePass->getVars()->getRootVar();
        var["gAOMapHalf"] = pUpsampleSrc;
        var["gAOSampler"] = mpLinearSampler;
        var["gAOMap"] = pAOMap;

        auto cb = var["PerFrameCB"];
        cb["gFullFrameDim"] = uint2(fullW, fullH);
        cb["gHalfFrameDim"] = uint2(halfW, halfH);

        mpUpsamplePass->execute(pRenderContext, uint3(fullW, fullH, 1));
    }

    ++mFrameIndex;
}
