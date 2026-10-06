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
#include "SSAOPass.h"

#include <random>

namespace
{
const std::string kSSAOProgramFile = "Arctic/SSAOPass/SSAOPass.cs.slang";
const std::string kSmoothProgramFile = "Arctic/SSAOPass/SSAOSmooth.cs.slang";
const std::string kUpsampleProgramFile = "Arctic/SSAOPass/SSAOUpsample.cs.slang";

const ChannelList kInputs = {
    { "gBufferA", "gGBufferA", "World normal (encoded N*0.5+0.5) + per-object", true, ResourceFormat::Unknown },
    { "sceneDepth", "gSceneDepth", "GBuffer scene depth (D32Float)", true, ResourceFormat::Unknown },
};
const ChannelList kOutputs = {
    { "aoMap", "gAOMap", "Full-res AO (RGBA32Float, r=AO)", false, ResourceFormat::RGBA32Float },
};

const char kRadius[] = "radius";
const char kBias[] = "bias";
const char kPower[] = "power";
const char kIntensity[] = "intensity";
const char kScaleRadiusInWorldSpace[] = "scaleRadiusInWorldSpace";
const char kFadeDistance[] = "fadeDistance";
const char kFadeRadius[] = "fadeRadius";
const char kUpsampleDepthThreshold[] = "upsampleDepthThreshold";
const char kQuality[] = "quality";
} // namespace

SSAOPass::SSAOPass(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    for (const auto& [key, value] : props)
    {
        if (key == kRadius) mRadius = value;
        else if (key == kBias) mBias = value;
        else if (key == kPower) mPower = value;
        else if (key == kIntensity) mIntensity = value;
        else if (key == kScaleRadiusInWorldSpace) mScaleRadiusInWorldSpace = value;
        else if (key == kFadeDistance) mFadeDistance = value;
        else if (key == kFadeRadius) mFadeRadius = value;
        else if (key == kUpsampleDepthThreshold) mUpsampleDepthThreshold = value;
        else if (key == kQuality) mQuality = value;
    }
}

Properties SSAOPass::getProperties() const
{
    Properties props;
    props[kRadius] = mRadius;
    props[kBias] = mBias;
    props[kPower] = mPower;
    props[kIntensity] = mIntensity;
    props[kScaleRadiusInWorldSpace] = mScaleRadiusInWorldSpace;
    props[kFadeDistance] = mFadeDistance;
    props[kFadeRadius] = mFadeRadius;
    props[kUpsampleDepthThreshold] = mUpsampleDepthThreshold;
    props[kQuality] = mQuality;
    return props;
}

RenderPassReflection SSAOPass::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    const uint2 sz = compileData.defaultTexDims;
    addRenderPassInputs(reflector, kInputs, ResourceBindFlags::ShaderResource, uint2(0));
    addRenderPassOutputs(reflector, kOutputs, ResourceBindFlags::UnorderedAccess, sz);
    return reflector;
}

void SSAOPass::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
    mpSSAOPass.reset();
    mpSmoothPass.reset();
    mpUpsamplePass.reset();
    mpRandomTexture.reset();
    mFrameIndex = 0;
    mDirty = true;
}

void SSAOPass::createRandomTexture()
{
    // 4x4 RGBA8 tile of random 2D unit vectors, encoded as v*0.5+0.5 in .rg
    // (matches UE's RandomNormalTexture format that the shader decodes as .rg*2-1).
    const uint32_t kTileSize = 4;
    const uint32_t kPixelCount = kTileSize * kTileSize;
    std::vector<uint8_t> data(kPixelCount * 4);
    std::mt19937 rng(0x515A0u); // fixed seed — deterministic across runs
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (uint32_t i = 0; i < kPixelCount; ++i)
    {
        float x = dist(rng);
        float y = dist(rng);
        float len = std::sqrt(x * x + y * y);
        if (len < 1e-4f) { x = 1.0f; y = 0.0f; len = 1.0f; }
        x /= len; y /= len;
        data[i * 4 + 0] = static_cast<uint8_t>(std::clamp(x * 0.5f + 0.5f, 0.0f, 1.0f) * 255.0f + 0.5f);
        data[i * 4 + 1] = static_cast<uint8_t>(std::clamp(y * 0.5f + 0.5f, 0.0f, 1.0f) * 255.0f + 0.5f);
        data[i * 4 + 2] = 0;
        data[i * 4 + 3] = 255;
    }
    mpRandomTexture = mpDevice->createTexture2D(kTileSize, kTileSize, ResourceFormat::RGBA8Unorm, 1u, 1u,
        data.data(), ResourceBindFlags::ShaderResource);
}

void SSAOPass::rebuildPasses()
{
    if (!mpScene) return;
    if (!mpRandomTexture) createRandomTexture();
    if (!mpSSAOPass)
    {
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kSSAOProgramFile).csEntry("csMain");
        desc.addTypeConformances(mpScene->getTypeConformances());
        // SHADER_QUALITY is a compile-time macro (selects the sample-set table
        // and step count); we pass it as a program define and re-create the
        // pass when the UI changes it (renderUI resets mpSSAOPass).
        DefineList defines = mpScene->getSceneDefines();
        defines.add("SHADER_QUALITY", std::to_string(mQuality));
        mpSSAOPass = ComputePass::create(mpDevice, desc, defines);
    }
    if (!mpSmoothPass)
    {
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kSmoothProgramFile).csEntry("csMain");
        desc.addTypeConformances(mpScene->getTypeConformances());
        mpSmoothPass = ComputePass::create(mpDevice, desc, mpScene->getSceneDefines());
    }
    if (!mpUpsamplePass)
    {
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kUpsampleProgramFile).csEntry("csMain");
        desc.addTypeConformances(mpScene->getTypeConformances());
        mpUpsamplePass = ComputePass::create(mpDevice, desc, mpScene->getSceneDefines());
    }
    mDirty = false;
}

void SSAOPass::renderUI(Gui::Widgets& widget)
{
    Gui::DropdownList qualityList;
    qualityList.push_back({0, "0 - Very Low"});
    qualityList.push_back({1, "1 - Low"});
    qualityList.push_back({2, "2 - Medium"});
    qualityList.push_back({3, "3 - High"});
    qualityList.push_back({4, "4 - Very High"});
    if (widget.dropdown("Quality", qualityList, mQuality))
    {
        widget.tooltip("UE SHADER_QUALITY tier. Changing this recompiles the SSAO shader.");
        mpSSAOPass.reset(); // force recompile with the new SHADER_QUALITY define
        mDirty = true;
    }
    mDirty |= widget.var("Radius", mRadius, 0.1f, 10.f, 0.05f);
    widget.tooltip("AO radius. In world-space mode this is meters; "
                   "in view-space mode it is UE's AORadiusInShader (screen units).");
    mDirty |= widget.var("Power", mPower, 0.1f, 8.f, 0.05f);
    widget.tooltip("Contrast power applied to AO (UE AmbientOcclusionPower). Higher = darker.");
    mDirty |= widget.var("Intensity", mIntensity, 0.f, 4.f, 0.01f);
    widget.tooltip("AO intensity multiplier (UE AmbientOcclusionIntensity).");
    mDirty |= widget.var("Bias", mBias, -0.1f, 0.1f, 0.001f);
    widget.tooltip("View-space position bias along the normal (UE AmbientOcclusionBias).");
    mDirty |= widget.var("Fade Distance", mFadeDistance, 1.f, 20000.f, 1.f);
    widget.tooltip("Depth at which AO starts fading out (UE FadeDistance).");
    mDirty |= widget.var("Fade Radius", mFadeRadius, 1.f, 5000.f, 1.f);
    widget.tooltip("Depth range over which AO ramps to 1 past Fade Distance (UE FadeRadius). "
                   "A soft ramp avoids the visible seam a hard cutoff leaves.");
    mDirty |= widget.var("Upsample Depth Threshold", mUpsampleDepthThreshold, 0.001f, 5.f, 0.005f);
    widget.tooltip("Bilateral upsample depth rejection, in 1/meters (UE's 0.003 in cm). "
                   "Lower = blurrier across depth edges; higher = more crawling at edges.");
    bool worldSpace = (mScaleRadiusInWorldSpace > 0.5f);
    if (widget.checkbox("World-space radius", worldSpace))
    {
        mScaleRadiusInWorldSpace = worldSpace ? 1.0f : 0.0f;
        mDirty = true;
    }
    widget.tooltip("On: radius is constant in world space. Off: radius scales with view-space depth (UE default).");
}

void SSAOPass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    auto pGBufferA = renderData.getTexture("gBufferA");
    auto pSceneDepth = renderData.getTexture("sceneDepth");
    auto pAOMap = renderData.getTexture("aoMap");
    if (!pGBufferA || !pSceneDepth || !pAOMap || !mpScene) return;

    if (mDirty) rebuildPasses();
    if (!mpSSAOPass || !mpSmoothPass || !mpUpsamplePass) return;

    const uint32_t fullW = pAOMap->getWidth();
    const uint32_t fullH = pAOMap->getHeight();
    const uint32_t halfW = (fullW + 1) / 2;
    const uint32_t halfH = (fullH + 1) / 2;

    // Allocate half-res intermediates (internal — not graph outputs).
    if (!mpHalfAO || mpHalfAO->getWidth() != halfW || mpHalfAO->getHeight() != halfH)
    {
        mpHalfAO = mpDevice->createTexture2D(halfW, halfH, ResourceFormat::RGBA32Float, 1u, 1u, nullptr,
            ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource);
        mpHalfAOSmoothed = mpDevice->createTexture2D(halfW, halfH, ResourceFormat::RGBA32Float, 1u, 1u, nullptr,
            ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource);
    }

    // UE ScreenSpaceAOParams[4].xy: AO ramps to 1 over [fadeDistance, fadeDistance+fadeRadius],
    // i.e. saturate(depth * fadeMul + fadeAdd) is 0 at fadeDistance and 1 at the end of the ramp.
    const float fadeRadius = std::max(mFadeRadius, 1e-3f);
    const float fadeMul = 1.0f / fadeRadius;
    const float fadeAdd = -mFadeDistance / fadeRadius;

    // ---- Pass 1: half-res SSAO ----
    {
        auto var = mpSSAOPass->getVars()->getRootVar();
        mpScene->bindShaderData(var["gScene"]);
        var["gGBufferA"] = pGBufferA;
        var["gSceneDepth"] = pSceneDepth;
        var["gRandomTexture"] = mpRandomTexture;
        var["gAOMapHalf"] = mpHalfAO;

        auto cb = var["PerFrameCB"];
        cb["gFullFrameDim"] = uint2(fullW, fullH);
        cb["gHalfFrameDim"] = uint2(halfW, halfH);
        cb["gRadius"] = mRadius;
        cb["gBias"] = mBias;
        cb["gScaleRadiusInWorldSpace"] = mScaleRadiusInWorldSpace;
        cb["gFadeMul"] = fadeMul;
        cb["gFadeAdd"] = fadeAdd;
        cb["gFrameIndex"] = mFrameIndex;

        mpSSAOPass->execute(pRenderContext, uint3(halfW, halfH, 1));
    }

    // UAV barrier: pass 1 writes mpHalfAO as UAV, pass 2 reads it as SRV.
    pRenderContext->uavBarrier(mpHalfAO.get());

    // ---- Pass 2: half-res dither smooth ----
    {
        auto var = mpSmoothPass->getVars()->getRootVar();
        mpScene->bindShaderData(var["gScene"]);
        var["gAOMapHalf"] = mpHalfAO;
        var["gAOMapHalfSmoothed"] = mpHalfAOSmoothed;

        auto cb = var["PerFrameCB"];
        cb["gHalfFrameDim"] = uint2(halfW, halfH);

        mpSmoothPass->execute(pRenderContext, uint3(halfW, halfH, 1));
    }

    // UAV barrier: pass 2 writes mpHalfAOSmoothed as UAV, pass 3 reads it as SRV.
    pRenderContext->uavBarrier(mpHalfAOSmoothed.get());

    // ---- Pass 3: bilateral upsample to full-res ----
    {
        auto uvar = mpUpsamplePass->getVars()->getRootVar();
        mpScene->bindShaderData(uvar["gScene"]);
        uvar["gAOMapHalf"] = mpHalfAOSmoothed;
        uvar["gGBufferA"] = pGBufferA;
        uvar["gSceneDepth"] = pSceneDepth;
        uvar["gAOMap"] = pAOMap;

        auto cb = uvar["PerFrameCB"];
        cb["gFullFrameDim"] = uint2(fullW, fullH);
        cb["gHalfFrameDim"] = uint2(halfW, halfH);
        cb["gPower"] = mPower;
        cb["gIntensity"] = mIntensity;
        cb["gFadeMul"] = fadeMul;
        cb["gFadeAdd"] = fadeAdd;
        cb["gDepthSimilarityScale"] = mUpsampleDepthThreshold;

        mpUpsamplePass->execute(pRenderContext, uint3(fullW, fullH, 1));
    }

    ++mFrameIndex;
}
