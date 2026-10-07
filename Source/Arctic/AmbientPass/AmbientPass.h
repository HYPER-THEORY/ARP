/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
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

    Two environment-derived resources back the IBL, each matching UE's own:

      - A cosine-convolved irradiance map (IrradianceConvolve.cs.slang), used for
        the diffuse lookup. Independent of roughness -- UE's AmbientCubemapMipAdjust.z
        is a single absolute mip for every pixel. Falcor's lat-long mip chain is a
        box-filtered blit, not a cosine kernel, so it cannot be used here.

      - UE's PreintegratedGF split-sum table (PreIntegratedGF.cs.slang), used for the
        specular F*G term in place of an analytic fit.

    Both are per-instance and rebuilt only when their inputs change; see
    updateIrradiance()/updatePreIntegratedGF().

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

    /** Rebuild the cosine-convolved irradiance map if the environment map it was
        built from changed.

        The convolution is done in the env map's local space (rotation is applied at
        lookup time, exactly as EnvMap::eval does), so rotating or re-tinting the
        environment does not invalidate it -- only a different texture or a different
        resolution does.
    */
    void updateIrradiance(RenderContext* pRenderContext);

    /** Build UE's PreintegratedGF split-sum table once and keep it resident. It is a
        pure function of (NoV, roughness), so there is nothing to invalidate.
    */
    void updatePreIntegratedGF(RenderContext* pRenderContext);

    ref<Scene> mpScene;
    ref<ComputePass> mpPass;

    // Cosine-convolved irradiance (lat-long). See updateIrradiance().
    ref<ComputePass> mpIrradiancePass;
    ref<Texture> mpIrradiance;
    ref<Sampler> mpIrradianceSampler;
    ref<Texture> mpIrradianceSrc;       ///< Env map the current irradiance was built from.
    uint2 mIrradianceSrcDim = uint2(0);

    // UE PreintegratedGF split-sum table (128x32 RG16Unorm). See updatePreIntegratedGF().
    ref<ComputePass> mpPreIntegratedGFPass;
    ref<Texture> mpPreIntegratedGF;
    ref<Sampler> mpPreIntegratedGFSampler;

    // UE AmbientCubemapColor equivalent: tint * intensity.
    float3 mTint = float3(1.f, 1.f, 1.f);
    float mIntensity = 1.f;
    bool  mUseFixedColor = false; // use a fixed color instead of the envMap for ambient IBL.
    float3 mAmbientColor = float3(1.f, 1.f, 1.f);
    bool mDirty = true;
};
