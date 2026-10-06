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
#include "SkyboxPass.h"

namespace
{
const std::string kSkyboxProgramFile = "Arctic/SkyboxPass/SkyboxPass.3d.slang";

const ChannelList kInputs = {
    { "sceneDepth", "gSceneDepth", "GBuffer scene depth", true, ResourceFormat::Unknown },
};
const ChannelList kOutputs = {
    { "skyboxColor", "gSkyboxColor", "Background color (RGBA32Float, 0 on geometry)", false, ResourceFormat::RGBA32Float },
};

const char kLod[] = "lod";
const char kSunAngularRadius[] = "sunAngularRadius";
const char kSunHaloMultiplier[] = "sunHaloMultiplier";
} // namespace

SkyboxPass::SkyboxPass(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    for (const auto& [key, value] : props)
    {
        if (key == kLod) mLod = value;
        else if (key == kSunAngularRadius) mSunAngularRadius = value;
        else if (key == kSunHaloMultiplier) mSunHaloMultiplier = value;
    }
}

void SkyboxPass::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
    pPass.reset();
    if (mpScene)
    {
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kSkyboxProgramFile).csEntry("csMain");
        desc.addTypeConformances(mpScene->getTypeConformances());
        pPass = ComputePass::create(mpDevice, desc, mpScene->getSceneDefines());
    }
}

Properties SkyboxPass::getProperties() const
{
    Properties props;
    props[kLod] = mLod;
    props[kSunAngularRadius] = mSunAngularRadius;
    props[kSunHaloMultiplier] = mSunHaloMultiplier;
    return props;
}

RenderPassReflection SkyboxPass::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    const uint2 sz = compileData.defaultTexDims;
    addRenderPassInputs(reflector, kInputs, ResourceBindFlags::ShaderResource, uint2(0));
    addRenderPassOutputs(reflector, kOutputs, ResourceBindFlags::UnorderedAccess, sz);
    return reflector;
}

void SkyboxPass::renderUI(Gui::Widgets& widget)
{
    widget.var("EnvMap LOD", mLod, 0.f, 12.f, 0.1f);
    widget.var("Sun Angular Radius (rad)", mSunAngularRadius, 0.f, 0.5f, 0.0005f);
    widget.var("Sun Halo Multiplier", mSunHaloMultiplier, 1.f, 20.f, 0.1f);
}

void SkyboxPass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    auto pSceneDepth = renderData.getTexture("sceneDepth");
    auto pSkyboxColor = renderData.getTexture("skyboxColor");
    if (!pSceneDepth || !pSkyboxColor || !pPass) return;

    auto var = pPass->getVars()->getRootVar();
    // Bind the scene parameter block (contains camera + envMap). Without this,
    // gScene.envMap / gScene.camera in the compute shader read default-empty
    // resources and the sky renders as pure black.
    mpScene->bindShaderData(var["gScene"]);
    var["gSceneDepth"] = pSceneDepth;
    var["gSkyboxColor"] = pSkyboxColor;
    var["PerFrameCB"]["gFrameDim"] = uint2(pSkyboxColor->getWidth(), pSkyboxColor->getHeight());
    var["PerFrameCB"]["gLod"] = mLod;
    var["PerFrameCB"]["gSunAngularRadius"] = mSunAngularRadius;
    var["PerFrameCB"]["gSunHaloMultiplier"] = mSunHaloMultiplier;

    pPass->execute(pRenderContext, uint3(pSkyboxColor->getWidth(), pSkyboxColor->getHeight(), 1));
}
