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
#include "../ShadowPass/ShadowViewData.h"

using namespace Falcor;

/** Fullscreen compute pass that projects the shadow depth texture onto the
    scene and produces a screen-sized shadow mask, porting UE's
    ShadowProjectionPixelShader.usf / ShadowFilteringCommon.ush.

    Inputs: GBufferPass depth (scene depth), ShadowPass shadow depth,
    GBufferA (world normal), GBufferB (shading model id), GBufferD (foliage density).
    Output: RG32Float shadow visibility in [0,1]:
      .r = surface shadow, .g = SSS transmission shadow (foliage, density-driven,
      UE ShadowProjectionPixelShader.usf bSubsurface path).

    The light view/projection matrix is NOT recomputed here — it is read from
    the RenderData dictionary as a ShadowViewData published by ShadowPass, so
    both passes are guaranteed to use the identical transform.

    Two filter modes are available, mirroring UE's own split between ManualPCF
    and DirectionalPCSS (Gather4 requires axis-aligned 2x2 footprints, which a
    rotated kernel cannot provide):
      - AxisAlignedPCF: Gather4-accelerated NxN bilinear PCF, texel-for-texel
        equivalent to UE ManualPCF. Temporally stable without TAA.
      - SobolDisk: sparse golden-angle disk whose phase and radial jitter come
        from UE's Sobol sequence, re-rotated per pixel and per frame. Wider,
        smoother penumbra for far fewer taps, but relies on TAAPass to resolve.
*/
class ShadowProjectionPass : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(ShadowProjectionPass, "ShadowProjectionPass", "Projects shadow depth into a screen-sized ShadowMask via PCF.");

    static ref<ShadowProjectionPass> create(ref<Device> pDevice, const Properties& props) { return make_ref<ShadowProjectionPass>(pDevice, props); }

    ShadowProjectionPass(ref<Device> pDevice, const Properties& props);

    /// Shadowmap filtering kernel. Values are serialized via getProperties(),
    /// so do not renumber.
    enum class FilterMode : uint32_t
    {
        AxisAlignedPCF = 0, ///< UE ManualPCF, Gather4-accelerated.
        SobolDisk = 1,      ///< Sobol-rotated golden-angle disk, TAA-resolved.
    };

    RenderPassReflection reflect(const CompileData& compileData) override;
    void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    void renderUI(Gui::Widgets& widget) override;
    void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;
    Properties getProperties() const override;

private:
    void parseProperties(const Properties& props);

    ref<Scene> mpScene;
    ref<ComputePass> pPass;
    ref<Sampler> mpShadowSampler;
    sigs::Connection mUpdateFlagsConnection;
    IScene::UpdateFlags mUpdateFlags = IScene::UpdateFlags::None;

    /// Light view parameters published by ShadowPass. Never recomputed here.
    ShadowViewData mShadowView;

    // Constant depth bias in METERS, used only to nudge the receiver toward the
    // light on the subsurface path (UE ProjectionDepthBiasParameters.x). Should
    // match ShadowPass's constant bias. Note UE's own bias parameters are
    // dimensionless (normalized over the cascade depth range), so its numeric
    // defaults do not transfer — see ShadowPass.h.
    float mConstantDepthBias = 0.01f;

    // Receiver bias and soft transition scale (UE ProjectionDepthBiasParameters.z /
    // SoftTransitionScale.z). SoftTransitionScale is in 1/meters: a depth
    // difference of `1/Scale` meters spans the full transition from unshadowed
    // to fully shadowed.
    float mReceiverBias = 1.0f;
    float mSoftTransitionScale = 10.0f;

    // UE ShadowSharpen (ShadowProjectionPixelShader.usf). 1 = identity.
    float mShadowSharpen = 1.0f;

    FilterMode mFilterMode = FilterMode::SobolDisk;

    // AxisAlignedPCF: kernel width in shadowmap texels. 1, 3, 5, 7.
    uint32_t mPCFKernelSize = 5;

    // SobolDisk: filter radius in WORLD METERS and tap count. Meters (rather
    // than texels) keeps the penumbra width stable when the shadow resolution
    // or the fitted frustum changes.
    float mFilterRadiusWorld = 0.06f;
    uint32_t mDiskTapCount = 12;

    // Scales subsurfaceDensityFromOpacity output into DensityMulConstant
    // (UE ProjectionDepthBiasParameters.w). Tunable via UI.
    float mDensityDepthScale = 100.f;

    /// Drives the temporal component of the Sobol rotation.
    uint32_t mFrameIndex = 0;

    bool mOptionsChanged = false;
};
