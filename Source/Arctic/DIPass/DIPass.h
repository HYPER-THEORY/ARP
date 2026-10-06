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
#pragma once
#include "Falcor.h"
#include "RenderGraph/RenderPass.h"
#include "RenderGraph/RenderPassHelpers.h"

using namespace Falcor;

/** Fullscreen direct lighting pass.

    Reads the UE-layout GBuffer (GBufferA-E) + scene depth + ShadowProjectionPass
    ShadowMask, and computes direct lighting from the first DirectionalLight in
    the scene.

    Shading model dispatch follows UE ShadingModels.ush:IntegrateBxDF:
      - SHADINGMODELID_UNLIT             → no lighting at all (UE skips the
        deferred light pass for these pixels). Their radiance is the emissive
        term, which GBufferPass writes to its own MRT and CompositePass adds.
      - SHADINGMODELID_DEFAULT_LIT       → DefaultLitBxDF
        (GGX specular + Lambert diffuse, UE dielectric Specular)
      - SHADINGMODELID_SUBSURFACE        → SubsurfaceBxDF
        (DefaultLit + wrap/in-scatter back-scatter, Beer-Lambert transmitted color)
      - SHADINGMODELID_TWOSIDED_FOLIAGE  → TwoSidedBxDF
        (DefaultLit + wrap-lighting transmission with GGX scatter)

    The shader is self-contained: it does not call into Falcor's material
    system. All material parameters come from the GBuffer.

    Deliberately absent, because UE's default legacy (non-Substrate) path also
    omits them: specular multiple-scattering energy conservation, Chan rough
    diffuse. Absent because Arctic has no such feature: anisotropy, area lights,
    contact shadows, light functions, baked shadowmasks, multiple lights.
*/
class DIPass : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(DIPass, "DIPass", "Fullscreen direct lighting pass (UE BRDF, ShadowMask-modulated).");

    static ref<DIPass> create(ref<Device> pDevice, const Properties& props) { return make_ref<DIPass>(pDevice, props); }

    DIPass(ref<Device> pDevice, const Properties& props);

    RenderPassReflection reflect(const CompileData& compileData) override;
    void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    void renderUI(Gui::Widgets& widget) override;
    void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;
    Properties getProperties() const override;

private:
    ref<Scene> mpScene;
    ref<ComputePass> pPass;
    float mLightIntensityScale = 1.f;
    /// UE View.MinRoughness. Its C++ default is not in ref/unrealengine (shaders only).
    float mMinRoughness = 0.02f;
    /// UE View.SubSurfaceColorAsTransmittanceAtDistanceInMeters. Same caveat.
    float mSubsurfaceTransmittanceDistance = 0.15f;
};
