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

/** Classic (unintegrated) SSAO pass.

    Ports UnrealEngine's MainPSandCS + WedgeWithNormal
    (Private/PostProcessAmbientOcclusion.usf) to a Falcor compute pass. Runs
    at half resolution over the GBuffer depth and world normal, smooths the
    4x4 dither pattern, then bilateral-upsamples AO to full resolution.

    Outputs:
      - aoMap (RGBA32Float, full-res): ambient occlusion in [0,1] (r channel)

    Algorithm parameters (exposed via GUI) mirror UE's ScreenSpaceAOParams:
      - radius                     (view/world-space AO radius)
      - power                      (AmbientOcclusionPower — contrast)
      - intensity                  (AmbientOcclusionIntensity)
      - scaleRadiusInWorldSpace    (0=view-space, 1=world-space constant)
      - fadeDistance / fadeRadius  (soft distance fade-out to AO=1)
      - upsampleDepthThreshold     (bilateral upsample depth rejection, 1/meters)
      - quality                    (UE SHADER_QUALITY 0-4, recompiles the shader)
*/
class SSAOPass : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(SSAOPass, "SSAOPass", "Classic UE-style SSAO (half-res + dither smooth + bilateral upsample).");

    static ref<SSAOPass> create(ref<Device> pDevice, const Properties& props) { return make_ref<SSAOPass>(pDevice, props); }

    SSAOPass(ref<Device> pDevice, const Properties& props);

    RenderPassReflection reflect(const CompileData& compileData) override;
    void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    void renderUI(Gui::Widgets& widget) override;
    void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;
    Properties getProperties() const override;

private:
    void rebuildPasses();
    void createRandomTexture();

    ref<Scene> mpScene;
    ref<ComputePass> mpSSAOPass;     // half-res SSAO
    ref<ComputePass> mpSmoothPass;   // half-res dither smooth
    ref<ComputePass> mpUpsamplePass; // bilateral upsample to full-res

    // Internal half-res intermediates (not graph outputs).
    ref<Texture> mpHalfAO;
    ref<Texture> mpHalfAOSmoothed;
    // 4x4 RGBA8 random rotation tile (UE RandomNormalTexture equivalent).
    ref<Texture> mpRandomTexture;
    uint32_t mFrameIndex = 0;

    // UE ScreenSpaceAOParams equivalent.
    // mRadius is UE's AORadiusInShader — a small normalized radius (not meters):
    // the per-sample screen-space offset is ~ mRadius / tan(halfFovY) (view-space
    // scaling mode, depth-independent screen radius).
    float mRadius = 1.0f;                 // AO radius (meters in world-space mode, UE AORadiusInShader)
    float mBias = 0.0f;                    // AmbientOcclusionBias (reserved hook)
    float mPower = 1.5f;                   // AmbientOcclusionPower (contrast)
    float mIntensity = 1.0f;              // AmbientOcclusionIntensity
    float mScaleRadiusInWorldSpace = 1.0f; // 0=view-space scaling (constant screen radius), 1=world-space constant (meters)
    // UE FadeDistance/FadeRadius: AO ramps to 1 (unoccluded) over
    // [mFadeDistance, mFadeDistance + mFadeRadius] instead of being cut off hard.
    float mFadeDistance = 5000.f;
    float mFadeRadius = 500.f;
    // UE's ComputeDepthSimilarity tweak constant. UE uses 0.003 in centimeters;
    // Falcor works in meters, so the equivalent is ~100x larger — leaving it at
    // 0.003 would make the bilateral weight ~1 everywhere, i.e. no edge rejection.
    float mUpsampleDepthThreshold = 0.3f;
    uint32_t mQuality = 4;                 // UE SHADER_QUALITY 0-4
    bool mDirty = true;
};
