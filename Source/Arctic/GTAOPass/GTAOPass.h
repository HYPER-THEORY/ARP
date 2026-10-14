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

/** Ground-Truth Ambient Occlusion pass.

    Ports UnrealEngine's GTAO chain (Private/PostProcessAmbientOcclusion.usf) as
    four compute passes matching UE's pass boundaries one-for-one, so each stage
    can be dumped and compared against UE in isolation:

      1. GTAO.cs.slang               half res GTAOCombinedPSandCS (HZB-accelerated)
      2. GTAOSpatialFilter.cs.slang  half res GTAOSpatialFilterCS
      3. GTAOTemporalFilter.cs.slang half res GTAOTemporalFilterPSandCS
      4. GTAOUpsample.cs.slang       full res GTAOUpsamplePSAndCS

    Because UE splits the 2/PI normalization across stages 1 and 2,
    the stage-1 output is NOT a usable AO value on its own — a
    fully unoccluded surface reads 2/PI there and only becomes 1.0 after stage 2.

    Input:  gBufferA (world normal), sceneDepth, motionVecs,
            zFurthest (HZB furthest mip chain from HierarchicalZPass)
    Output: aoMap (R8Unorm, full res)

    HZB: UE's shipped horizon search (SearchForLargestAngleDual_HZB) escalates
    through the furthest HZB mips so each tap is a conservative depth over a
    2^mip footprint. The mip staircase is a fixed per-quality-tier schedule
    matching UE's GTAO_BIASMIPLEVEL. UE's loop runs i=1..NumSteps (1-based);
    the i==0 branch in UE's source is dead code and never executes:
      tap i=1  -> mip BIASMIPLEVEL
      tap i=2  -> mip BIASMIPLEVEL + 1
      tap i=3  -> mip BIASMIPLEVEL        (resets to base; no branch for i==3)
      tap i>=4 -> mip BIASMIPLEVEL + 2
    BIASMIPLEVEL per quality tier: Q0=2, Q1=1, Q2=0, Q3=0, Q4=0.
    When zFurthest is not connected, falls back to direct full-res depth Load.
*/
class GTAOPass : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(GTAOPass, "GTAOPass", "Ground-Truth Ambient Occlusion (UE-aligned: search -> spatial -> temporal -> upsample).");

    static ref<GTAOPass> create(ref<Device> pDevice, const Properties& props) { return make_ref<GTAOPass>(pDevice, props); }

    GTAOPass(ref<Device> pDevice, const Properties& props);

    RenderPassReflection reflect(const CompileData& compileData) override;
    void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    void renderUI(Gui::Widgets& widget) override;
    void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;
    Properties getProperties() const override;

private:
    void rebuildPasses();
    void allocateResources(uint32_t fullW, uint32_t fullH);

    ref<Scene> mpScene;
    ref<ComputePass> mpSearchPass;    // 1. horizon search + inner integrate (HZB)
    ref<ComputePass> mpSpatialPass;   // 2. 5x5 bilateral
    ref<ComputePass> mpTemporalPass;  // 3. reprojected temporal accumulation
    ref<ComputePass> mpUpsamplePass;  // 4. conservative min-of-4 upsample
    ref<Sampler> mpLinearSampler;
    ref<Sampler> mpHZBSampler;        // point-clamp for HZB reads

    // Half-res intermediates (not graph outputs). AO is R8Unorm like UE's.
    ref<Texture> mpHalfAO;         // stage 1 output (2/PI-scaled)
    ref<Texture> mpHalfAOFiltered; // stage 2 output
    ref<Texture> mpHalfHistory[2]; // stage 3 ping-pong; [mHistoryIdx] is current
    uint32_t mHistoryIdx = 0;
    uint32_t mFrameIndex = 0;
    uint2 mAllocDim = uint2(0);

    // --- UE GTAOParams / ScreenSpaceAOParams equivalents ---
    float mEffectRadius = 1.0f;   // GTAOParams[3].y — world-space radius
    float mThickness = 0.5f;      // r.GTAO.ThicknessBlend; mapped to 1-t^2 on upload
    uint32_t mNumAngles = 2;      // GTAOParams[4].y — UE default
    float mPower = 2.0f;          // ScreenSpaceAOParams[0].x
    float mIntensity = 1.0f;      // ScreenSpaceAOParams[0].w
    // Fade: lerp(AO, 1, saturate(depth*fadeMul+fadeAdd)); mirrors UE ScreenSpaceAOParams[4].xy.
    float mFadeDistance = 5000.f; // AmbientOcclusionFadeDistance (cm)
    float mFadeRadius = 1000.f;   // AmbientOcclusionFadeRadius (cm)
    // MaxDistance: early-out threshold; depths beyond this write AO=1 immediately.
    // Corresponds to UE ScreenSpaceAOParams[4].w (AmbientOcclusionFadeDistance in UE's GTAO).
    // Kept as a separate knob so fade and hard-clip can be tuned independently.
    float mMaxDistance = 5000.f;
    float mSpatialFilterWeight = 20000.f;
    float mTemporalBlendWeight = 0.1f;
    bool mEnableTemporal = true;

    enum class OutputStage : uint32_t { Search = 0, Spatial = 1, Temporal = 2 };
    OutputStage mOutputStage = OutputStage::Temporal;

    bool mDirty = true;
    bool mHasHistory = false;
};
