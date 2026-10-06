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
#pragma once
#include "Falcor.h"
#include "RenderGraph/RenderPass.h"
#include "RenderGraph/RenderPassHelpers.h"

using namespace Falcor;

/** Lighting composite pass.

    The single place where the deferred lighting terms are summed. Each upstream
    pass now emits only its own contribution:

      - SkyboxPass.skyboxColor  background radiance (0 on geometry)
      - DIPass.color            sun direct lighting  (0 on background)
      - AmbientPass.ambient     sky direct lighting, i.e. single-bounce ambient IBL
      - IndirectPass.indirect   canopy multiple scattering (>= 2 bounces)

    Geometry pixels get direct + ambient + indirect; background pixels get the
    skybox alone. Splitting the sum into independent terms means each can be
    toggled and scaled independently here, and each is separately inspectable in
    DebugViewPass without the composite having already folded it in.
*/
class CompositePass : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(CompositePass, "CompositePass", "Composites skybox + direct + ambient + indirect into the final HDR color.");

    static ref<CompositePass> create(ref<Device> pDevice, const Properties& props) { return make_ref<CompositePass>(pDevice, props); }

    CompositePass(ref<Device> pDevice, const Properties& props);

    RenderPassReflection reflect(const CompileData& compileData) override;
    void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    void renderUI(Gui::Widgets& widget) override;
    Properties getProperties() const override;

private:
    ref<ComputePass> mpPass;

    bool mEnableDirect = true;
    float mDirectIntensity = 1.f;
    bool mEnableAmbient = true;
    float mAmbientIntensity = 1.f;
    // Canopy multiple scattering (>= 2 bounces). Deliberately not multiplied by
    // AO anywhere in the chain: AO models occlusion of the uncollided sky, which
    // this term has by construction already scattered out of, so applying it
    // would darken the very contribution meant to lift the shadow side.
    //
    // 1.0 because IndirectApproxPass outputs a physical radiance: it applies the
    // leaf's two-sided BSDF itself, so its result needs no further scaling.
    bool mEnableIndirect = true;
    float mIndirectIntensity = 1.f;
    bool mEnableSkybox = true;
    float mSkyboxIntensity = 1.f;
    // Self-illumination from the emissive MRT. UE's base pass writes
    // this directly to SceneColor, so it is deliberately unshadowed and
    // un-occluded, and it is the only radiance an Unlit pixel receives.
    bool mEnableEmissive = true;
    float mEmissiveIntensity = 1.f;
};
