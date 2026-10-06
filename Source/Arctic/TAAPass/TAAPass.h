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
 #    contributors may be used to endorse or promote products derived from
 #    this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 # LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#pragma once
#include "Falcor.h"
#include "RenderGraph/RenderPass.h"
#include "RenderGraph/RenderPassHelpers.h"

using namespace Falcor;

/** Temporal anti-aliasing pass (port of UE TemporalAA.usf TemporalAASample).

    Runs on HDR color before tonemapping. Maintains a full-res history texture
    matching colorOut's format (RGBA32Float), advanced with a blit each frame.
    History is reset on the first frame and on resize. When disabled, passes
    colorIn through unchanged.

    Mirrors UE's main TAA permutation (TAA_PASS_CONFIG 0 / TAA_QUALITY_HIGH):
    5-tap plus reconstruction filter with jitter-aware Gaussian weights, plus
    min/max clamp box in YCgCo, nearest-depth velocity dilation over a +-2 px X,
    Catmull-Rom history resampling, and an HDR-weighted final blend.

    Inputs:
      - colorIn    (RGBA32Float HDR): current-frame color
      - motionVecs (RG16Float): MotionVectorPass output (current->previous UV)
      - sceneDepth (device Z, 0=near): GBufferPass depth, for velocity dilation
    Output:
      - colorOut   (RGBA32Float HDR): temporally resolved color
*/
class TAAPass : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(TAAPass, "TAAPass", "Temporal AA (UE TemporalAA port, HDR pre-tonemap).");

    static ref<TAAPass> create(ref<Device> pDevice, const Properties& props) { return make_ref<TAAPass>(pDevice, props); }

    TAAPass(ref<Device> pDevice, const Properties& props);

    RenderPassReflection reflect(const CompileData& compileData) override;
    void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    void renderUI(Gui::Widgets& widget) override;
    void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;
    Properties getProperties() const override;

    void setEnabled(bool enabled) { mEnabled = enabled; }
    bool getEnabled() const { return mEnabled; }

    /// Drop the history, forcing a full reseed from the current frame. Call this
    /// on a genuine view cut (level load, camera teleport, resolution change).
    void requestCameraCut() { mHasHistory = false; }

private:
    void rebuildPass();
    void allocatePrevColor(const Texture* pColorOut);

    ref<Scene> mpScene;
    ref<ComputePass> mpPass;
    ref<Sampler> mpLinearSampler;

    ref<Texture> mpPrevColor; // history (matches colorOut format, full-res)
    bool mHasHistory = false;
    bool mEnabled = true;

    // NOTE: there is deliberately NO camera-motion-based cut heuristic here. UE
    // never resets TAA because the camera moved fast -- it relies on the motion
    // vectors, the off-screen test and the neighborhood clamp. A pose-delta
    // threshold silently disables TAA during fast fly-throughs. Use
    // requestCameraCut() for real cuts instead.

    // UE parameters.
    float mCurrentFrameWeight = 0.1f; // UE CurrentFrameWeight (baseline blend)
    float mExposureScale = 1.0f;      // UE FrameExposureScale * OneOverPreExposure

    bool mDirty = true;
};
