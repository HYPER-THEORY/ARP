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
#include "AmbientPass.h"

namespace
{
const std::string kProgramFile = "Arctic/AmbientPass/AmbientPass.3d.slang";

const ChannelList kInputs = {
    { "gBufferA", "gGBufferA", "World normal + per-object", true, ResourceFormat::Unknown },
    { "gBufferB", "gGBufferB", "Metallic/Specular/Roughness/ShadingModelID", true, ResourceFormat::Unknown },
    { "gBufferC", "gGBufferC", "Base color + AO", true, ResourceFormat::Unknown },
    { "gBufferD", "gGBufferD", "Custom data (subsurface for foliage)", true, ResourceFormat::Unknown },
    { "sceneDepth", "gSceneDepth", "GBuffer scene depth", true, ResourceFormat::Unknown },
    { "aoMap", "gAOMap", "Screen-space AO in .r", true, ResourceFormat::Unknown },
};
const ChannelList kOutputs = {
    { "ambient", "gAmbient", "Sky direct (single-bounce ambient IBL, RGBA32Float HDR)", false, ResourceFormat::RGBA32Float },
};

const char kTint[] = "tint";
const char kIntensity[] = "intensity";
const char kUseFixedColor[] = "useFixedColor";
const char kAmbientColor[] = "ambientColor";
const char kDiffuseMip[] = "diffuseMip";
const char kSpecularMipBoost[] = "specularMipBoost";
const char kApplyAOToSpecular[] = "applyAOToSpecular";
const char kApplyAOToDiffuse[] = "applyAOToDiffuse";
} // namespace

AmbientPass::AmbientPass(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    for (const auto& [key, value] : props)
    {
        if (key == kTint) mTint = value;
        else if (key == kIntensity) mIntensity = value;
        else if (key == kUseFixedColor) mUseFixedColor = value;
        else if (key == kAmbientColor) mAmbientColor = value;
        else if (key == kDiffuseMip) mDiffuseMip = value;
        else if (key == kSpecularMipBoost) mSpecularMipBoost = value;
        else if (key == kApplyAOToSpecular) mApplyAOToSpecular = value;
        else if (key == kApplyAOToDiffuse) mApplyAOToDiffuse = value;
    }
}

Properties AmbientPass::getProperties() const
{
    Properties props;
    props[kTint] = mTint;
    props[kIntensity] = mIntensity;
    props[kUseFixedColor] = mUseFixedColor;
    props[kAmbientColor] = mAmbientColor;
    props[kDiffuseMip] = mDiffuseMip;
    props[kSpecularMipBoost] = mSpecularMipBoost;
    props[kApplyAOToSpecular] = mApplyAOToSpecular;
    props[kApplyAOToDiffuse] = mApplyAOToDiffuse;
    return props;
}

RenderPassReflection AmbientPass::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    const uint2 sz = compileData.defaultTexDims;
    addRenderPassInputs(reflector, kInputs, ResourceBindFlags::ShaderResource, uint2(0));
    addRenderPassOutputs(reflector, kOutputs, ResourceBindFlags::UnorderedAccess, sz);
    return reflector;
}

void AmbientPass::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
    mpPass.reset();
    mDirty = true;
}

void AmbientPass::rebuildPass()
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

void AmbientPass::renderUI(Gui::Widgets& widget)
{
    mDirty |= widget.var("Tint", mTint, 0.f, 4.f, 0.01f);
    mDirty |= widget.var("Intensity", mIntensity, 0.f, 16.f, 0.01f);
    mDirty |= widget.checkbox("Use Fixed Color", mUseFixedColor);
    widget.tooltip("When on, ambient IBL uses Ambient Color instead of the envMap.");
    if (mUseFixedColor)
    {
        mDirty |= widget.var("Ambient Color", mAmbientColor, 0.f, 4.f, 0.01f);
    }
    else
    {
        mDirty |= widget.var("Diffuse Mip", mDiffuseMip, 0.f, 12.f, 0.1f);
        widget.tooltip("Absolute mip level for diffuse IBL lookup (UE AmbientCubemapMipAdjust.z).");
        mDirty |= widget.var("Specular Mip Boost", mSpecularMipBoost, -4.f, 4.f, 0.1f);
        widget.tooltip("Subtracted from roughness-derived specular mip (higher = blurrier reflections).");
    }
    mDirty |= widget.checkbox("Apply AO to Diffuse", mApplyAOToDiffuse);
    mDirty |= widget.checkbox("Apply AO to Specular", mApplyAOToSpecular);
}

void AmbientPass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    auto pAmbient = renderData.getTexture("ambient");
    if (!pAmbient || !mpScene) return;

    if (mDirty) rebuildPass();
    if (!mpPass) return;

    // Compute envmap mip count from the env map texture dimensions.
    // AmbientCubemapMipAdjust.w = MipCount.
    float mipCount = 1.f;
    if (mpScene->getEnvMap() && mpScene->getEnvMap()->getEnvMap())
    {
        uint32_t w = mpScene->getEnvMap()->getEnvMap()->getWidth();
        // ComputeCubemapMipFromRoughness's MipCount is a *cube face* mip count
        // (CubemapCommon.ush — "e.g. 10 for x 512x512"), but our env map is a
        // lat-long. A lat-long of width W resolves the same angle per texel as a
        // cube face of W/4 (360/W == 90/(W/4)), so the equivalent count is
        // log2(W/4) + 1 == log2(W) - 1. Using log2(W) + 1 here would land two mips
        // coarser at every roughness, i.e. 4x the angular blur UE would pick.
        mipCount = std::max(1.f, std::log2(float(w)) - 1.f);
    }

    auto var = mpPass->getVars()->getRootVar();
    mpScene->bindShaderData(var["gScene"]);
    var["gGBufferA"] = renderData.getTexture("gBufferA");
    var["gGBufferB"] = renderData.getTexture("gBufferB");
    var["gGBufferC"] = renderData.getTexture("gBufferC");
    var["gGBufferD"] = renderData.getTexture("gBufferD");
    var["gSceneDepth"] = renderData.getTexture("sceneDepth");
    var["gAOMap"] = renderData.getTexture("aoMap");
    var["gAmbient"] = pAmbient;

    auto cb = var["PerFrameCB"];
    cb["gFrameDim"] = uint2(pAmbient->getWidth(), pAmbient->getHeight());
    cb["gTint"] = mTint;
    cb["gIntensity"] = mIntensity;
    cb["gAmbientColor"] = mAmbientColor;
    cb["gUseFixedColor"] = mUseFixedColor ? 1u : 0u;
    cb["gDiffuseMip"] = mDiffuseMip;
    cb["gSpecularMipBoost"] = mSpecularMipBoost;
    cb["gMipCount"] = mipCount;
    cb["gApplyAOToDiffuse"] = mApplyAOToDiffuse ? 1u : 0u;
    cb["gApplyAOToSpecular"] = mApplyAOToSpecular ? 1u : 0u;

    mpPass->execute(pRenderContext, uint3(pAmbient->getWidth(), pAmbient->getHeight(), 1));
}
