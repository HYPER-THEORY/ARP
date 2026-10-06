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
#include "DIPass.h"
#include "Scene/Lights/Light.h"

namespace
{
const std::string kDIPassProgramFile = "Arctic/DIPass/DIPass.3d.slang";

const ChannelList kInputs = {
    { "gBufferA", "gGBufferA", "World normal + per-object", true, ResourceFormat::Unknown },
    { "gBufferB", "gGBufferB", "Metallic/Specular/Roughness/ShadingModelID", true, ResourceFormat::Unknown },
    { "gBufferC", "gGBufferC", "Base color + AO", true, ResourceFormat::Unknown },
    { "gBufferD", "gGBufferD", "Custom data (sqrt(subsurfaceColor) + opacity)", true, ResourceFormat::Unknown },
    { "gBufferE", "gGBufferE", "Precomputed shadow factors (unused: no baked lighting)", true, ResourceFormat::Unknown },
    { "sceneDepth", "gSceneDepth", "GBuffer scene depth", true, ResourceFormat::Unknown },
    { "shadowMask", "gShadowMask", "Shadow visibility (RG32Float: .r surface, .g SSS transmission)", true, ResourceFormat::Unknown },
};
const ChannelList kOutputs = {
    { "color", "gColor", "Sun direct lighting, 0 on background (RGBA32Float)", false, ResourceFormat::RGBA32Float },
};

const char kLightIntensityScale[] = "lightIntensityScale";
const char kMinRoughness[] = "minRoughness";
const char kSubsurfaceTransmittanceDistance[] = "subsurfaceTransmittanceDistance";
} // namespace

DIPass::DIPass(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    for (const auto& [key, value] : props)
    {
        if (key == kLightIntensityScale) mLightIntensityScale = value;
        else if (key == kMinRoughness) mMinRoughness = value;
        else if (key == kSubsurfaceTransmittanceDistance) mSubsurfaceTransmittanceDistance = value;
    }
}

Properties DIPass::getProperties() const
{
    Properties props;
    props[kLightIntensityScale] = mLightIntensityScale;
    props[kMinRoughness] = mMinRoughness;
    props[kSubsurfaceTransmittanceDistance] = mSubsurfaceTransmittanceDistance;
    return props;
}

RenderPassReflection DIPass::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    const uint2 sz = compileData.defaultTexDims;
    addRenderPassInputs(reflector, kInputs, ResourceBindFlags::ShaderResource, uint2(0));
    addRenderPassOutputs(reflector, kOutputs, ResourceBindFlags::UnorderedAccess, sz);
    return reflector;
}

void DIPass::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
    pPass.reset();
    if (mpScene)
    {
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kDIPassProgramFile).csEntry("csMain");
        desc.addTypeConformances(mpScene->getTypeConformances());
        pPass = ComputePass::create(mpDevice, desc, mpScene->getSceneDefines());
    }
}

void DIPass::renderUI(Gui::Widgets& widget)
{
    widget.var("Light intensity scale", mLightIntensityScale, 0.f, 100.f, 0.01f);

    widget.var("Min roughness", mMinRoughness, 0.f, 1.f, 0.001f);
    widget.tooltip(
        "Lower bound applied to GBuffer roughness, matching UE's View.MinRoughness "
        "(CapsuleLightIntegrate.ush). At roughness 0 the GGX lobe collapses to a "
        "delta and the highlight disappears. UE's C++ default is not in ref/unrealengine "
        "(shaders only); 0.02 is the value UE ships.",
        true
    );

    widget.var("Subsurface transmittance distance", mSubsurfaceTransmittanceDistance, 0.001f, 2.f, 0.001f);
    widget.tooltip(
        "Only affects SHADINGMODELID_SUBSURFACE. UE's "
        "View.SubSurfaceColorAsTransmittanceAtDistanceInMeters: the normalized distance at "
        "which SubsurfaceColor is interpreted as a transmittance, used to derive the "
        "Beer-Lambert extinction (ShadingModels.ush). Smaller = denser = darker "
        "transmitted color.",
        true
    );
}

void DIPass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    auto pColor = renderData.getTexture("color");
    if (!pColor || !pPass) return;

    // Find first DirectionalLight and upload its parameters.
    float3 lightDir = float3(0.f, -1.f, 0.f);
    float3 lightIntensity = float3(1.f);
    bool hasLight = false;
    if (mpScene)
    {
        const auto& lights = mpScene->getLights();
        for (const auto& l : lights)
        {
            if (!l || l->getType() != LightType::Directional) continue;
            auto pDir = dynamic_cast<DirectionalLight*>(l.get());
            if (!pDir) continue;
            lightDir = normalize(pDir->getWorldDirection());
            lightIntensity = pDir->getIntensity() * mLightIntensityScale;
            hasLight = true;
            break;
        }
    }

    auto var = pPass->getVars()->getRootVar();
    // Bind the scene parameter block (camera + materials + lights). The DI
    // shader reads gScene.camera.data.invViewProj / getPosition() to
    // reconstruct world positions and view direction.
    if (mpScene) mpScene->bindShaderData(var["gScene"]);
    var["gGBufferA"] = renderData.getTexture("gBufferA");
    var["gGBufferB"] = renderData.getTexture("gBufferB");
    var["gGBufferC"] = renderData.getTexture("gBufferC");
    var["gGBufferD"] = renderData.getTexture("gBufferD");
    var["gGBufferE"] = renderData.getTexture("gBufferE");
    var["gSceneDepth"] = renderData.getTexture("sceneDepth");
    var["gShadowMask"] = renderData.getTexture("shadowMask");
    var["gColor"] = pColor;

    var["PerFrameCB"]["gFrameDim"] = uint2(pColor->getWidth(), pColor->getHeight());
    var["PerFrameCB"]["gLightDir"] = lightDir;
    var["PerFrameCB"]["gLightIntensity"] = lightIntensity;
    var["PerFrameCB"]["gHasLight"] = hasLight ? 1u : 0u;
    var["PerFrameCB"]["gMinRoughness"] = mMinRoughness;
    var["PerFrameCB"]["gSubsurfaceTransmittanceDistance"] = mSubsurfaceTransmittanceDistance;

    pPass->execute(pRenderContext, uint3(pColor->getWidth(), pColor->getHeight(), 1));
}
