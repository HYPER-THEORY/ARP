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

/** Ambient direct lighting pass.

    Ports UnrealEngine's AmbientCubemapComposite.usf to a Falcor compute pass.
    Uses Falcor's EnvMap (lat-long) for IBL lookups and screen-space AO for
    occlusion.

    Computes the sky's direct (single-bounce) contribution only.

    Outputs:
      - ambient (RGBA32Float): the ambient contribution alone, 0 on background
*/
class AmbientPass : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(AmbientPass, "AmbientPass", "Ambient IBL pass (UE AmbientCubemapComposite port).");

    static ref<AmbientPass> create(ref<Device> pDevice, const Properties& props) { return make_ref<AmbientPass>(pDevice, props); }

    AmbientPass(ref<Device> pDevice, const Properties& props);

    RenderPassReflection reflect(const CompileData& compileData) override;
    void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    void renderUI(Gui::Widgets& widget) override;
    void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;
    Properties getProperties() const override;

private:
    void rebuildPass();

    ref<Scene> mpScene;
    ref<ComputePass> mpPass;

    // UE AmbientCubemapColor equivalent: tint * intensity.
    float3 mTint = float3(1.f, 1.f, 1.f);
    float mIntensity = 1.f;
    bool  mUseFixedColor = false; // use a fixed color instead of the envMap for ambient IBL.
    float3 mAmbientColor = float3(1.f, 1.f, 1.f);
    // UE AmbientCubemapMipAdjust: { mul, add, diffuseMip, mipCount }.
    // mipCount is computed from the env map dimensions at runtime; we keep
    // a user-facing "diffuse mip" + "specular mip boost" pair.
    float mDiffuseMip = 0.f;        // AbsoluteDiffuseMip
    float mSpecularMipBoost = 0.f;  // subtracted from ComputeCubemapMipFromRoughness result
    // Apply AO also to specular (UE multiplies final OutColor by AmbientOcclusion,
    // which dampens both diffuse and specular).
    bool mApplyAOToSpecular = true;
    bool mApplyAOToDiffuse = true;
    bool mDirty = true;
};
