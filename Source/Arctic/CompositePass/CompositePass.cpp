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
#include "CompositePass.h"

namespace
{
const std::string kProgramFile = "Arctic/CompositePass/CompositePass.cs.slang";

const ChannelList kInputs = {
    // clang-format off
    { "sceneDepth",  "gSceneDepth",  "GBuffer scene depth (background test)",   true, ResourceFormat::Unknown },
    { "skyboxColor", "gSkyboxColor", "SkyboxPass.skyboxColor (background, HDR)", true, ResourceFormat::Unknown },
    { "diColor",     "gDIColor",     "DIPass.color (sun direct lighting, HDR)", true, ResourceFormat::Unknown },
    { "ambient",     "gAmbient",     "AmbientPass.ambient (sky direct, HDR)",   true, ResourceFormat::Unknown },
    { "indirect",    "gIndirect",    "IndirectPass.indirect (>= 2 bounce, HDR)", true, ResourceFormat::Unknown },
    { "emissive",    "gEmissive",    "GBufferPass.emissive (HDR)",              true, ResourceFormat::Unknown },
    // clang-format on
};
const ChannelList kOutputs = {
    { "color", "gColor", "Composited HDR color (RGBA32Float)", false, ResourceFormat::RGBA32Float },
};

const char kEnableDirect[] = "enableDirect";
const char kDirectIntensity[] = "directIntensity";
const char kEnableAmbient[] = "enableAmbient";
const char kAmbientIntensity[] = "ambientIntensity";
const char kEnableIndirect[] = "enableIndirect";
const char kIndirectIntensity[] = "indirectIntensity";
const char kEnableSkybox[] = "enableSkybox";
const char kSkyboxIntensity[] = "skyboxIntensity";
const char kEnableEmissive[] = "enableEmissive";
const char kEmissiveIntensity[] = "emissiveIntensity";
} // namespace

CompositePass::CompositePass(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    for (const auto& [key, value] : props)
    {
        if (key == kEnableDirect) mEnableDirect = value;
        else if (key == kDirectIntensity) mDirectIntensity = value;
        else if (key == kEnableAmbient) mEnableAmbient = value;
        else if (key == kAmbientIntensity) mAmbientIntensity = value;
        else if (key == kEnableIndirect) mEnableIndirect = value;
        else if (key == kIndirectIntensity) mIndirectIntensity = value;
        else if (key == kEnableSkybox) mEnableSkybox = value;
        else if (key == kSkyboxIntensity) mSkyboxIntensity = value;
        else if (key == kEnableEmissive) mEnableEmissive = value;
        else if (key == kEmissiveIntensity) mEmissiveIntensity = value;
    }

    // No scene dependency: this pass only reads screen-space textures, so the
    // program can be built once here.
    ProgramDesc desc;
    desc.addShaderLibrary(kProgramFile).csEntry("csMain");
    mpPass = ComputePass::create(mpDevice, desc);
}

Properties CompositePass::getProperties() const
{
    Properties props;
    props[kEnableDirect] = mEnableDirect;
    props[kDirectIntensity] = mDirectIntensity;
    props[kEnableAmbient] = mEnableAmbient;
    props[kAmbientIntensity] = mAmbientIntensity;
    props[kEnableIndirect] = mEnableIndirect;
    props[kIndirectIntensity] = mIndirectIntensity;
    props[kEnableSkybox] = mEnableSkybox;
    props[kSkyboxIntensity] = mSkyboxIntensity;
    props[kEnableEmissive] = mEnableEmissive;
    props[kEmissiveIntensity] = mEmissiveIntensity;
    return props;
}

RenderPassReflection CompositePass::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    const uint2 sz = compileData.defaultTexDims;
    addRenderPassInputs(reflector, kInputs, ResourceBindFlags::ShaderResource, uint2(0));
    addRenderPassOutputs(reflector, kOutputs, ResourceBindFlags::UnorderedAccess, sz);
    return reflector;
}

void CompositePass::renderUI(Gui::Widgets& widget)
{
    widget.checkbox("Direct", mEnableDirect);
    if (mEnableDirect) widget.var("Direct Intensity", mDirectIntensity, 0.f, 8.f, 0.01f);

    widget.checkbox("Ambient", mEnableAmbient);
    if (mEnableAmbient) widget.var("Ambient Intensity", mAmbientIntensity, 0.f, 8.f, 0.01f);

    widget.checkbox("Indirect", mEnableIndirect);
    if (mEnableIndirect) widget.var("Indirect Intensity", mIndirectIntensity, 0.f, 8.f, 0.01f);

    widget.checkbox("Emissive", mEnableEmissive);
    if (mEnableEmissive) widget.var("Emissive Intensity", mEmissiveIntensity, 0.f, 8.f, 0.01f);

    widget.separator();
    widget.checkbox("Skybox", mEnableSkybox);
    if (mEnableSkybox) widget.var("Skybox Intensity", mSkyboxIntensity, 0.f, 8.f, 0.01f);
}

void CompositePass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    auto pColor = renderData.getTexture("color");
    if (!pColor || !mpPass) return;

    auto pSceneDepth = renderData.getTexture("sceneDepth");
    auto pDIColor = renderData.getTexture("diColor");
    auto pAmbient = renderData.getTexture("ambient");
    auto pSkybox = renderData.getTexture("skyboxColor");
    auto pIndirect = renderData.getTexture("indirect");
    auto pEmissive = renderData.getTexture("emissive");
    if (!pSceneDepth || !pDIColor) return;

    // Every optional input still needs a valid descriptor bound, so fall back to
    // diColor (always present) and let the gHas* flags gate the reads.
    auto var = mpPass->getVars()->getRootVar();
    var["gSceneDepth"] = pSceneDepth;
    var["gDIColor"] = pDIColor;
    var["gAmbient"] = pAmbient ? pAmbient : pDIColor;
    var["gSkyboxColor"] = pSkybox ? pSkybox : pDIColor;
    var["gIndirect"] = pIndirect ? pIndirect : pDIColor;
    var["gEmissive"] = pEmissive ? pEmissive : pDIColor;
    var["gColor"] = pColor;

    auto cb = var["PerFrameCB"];
    cb["gFrameDim"] = uint2(pColor->getWidth(), pColor->getHeight());
    cb["gDirectIntensity"] = mEnableDirect ? mDirectIntensity : 0.f;
    cb["gAmbientIntensity"] = (pAmbient && mEnableAmbient) ? mAmbientIntensity : 0.f;
    cb["gIndirectIntensity"] = (pIndirect && mEnableIndirect) ? mIndirectIntensity : 0.f;
    cb["gSkyboxIntensity"] = (pSkybox && mEnableSkybox) ? mSkyboxIntensity : 0.f;
    cb["gEmissiveIntensity"] = (pEmissive && mEnableEmissive) ? mEmissiveIntensity : 0.f;

    mpPass->execute(pRenderContext, uint3(pColor->getWidth(), pColor->getHeight(), 1));
}
