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
#include "Core/SampleApp.h"
#include "Core/AssetResolver.h"
#include "RenderGraph/RenderGraph.h"
#include "RenderGraph/RenderPassStandardFlags.h"
#include "Scene/SceneBuilder.h"
#include "Scene/Scene.h"
#include "Scene/Camera/Camera.h"
#include "Scene/Lights/Light.h"
#include "Scene/Material/StandardMaterial.h"
#include "GBufferPass/GBufferPass.h"
#include "ShadowPass/ShadowPass.h"
#include "ShadowProjectionPass/ShadowProjectionPass.h"
#include "SkyboxPass/SkyboxPass.h"
#include "DIPass/DIPass.h"
#include "AmbientPass/AmbientPass.h"
#include "CompositePass/CompositePass.h"
#include "GTAOPass/GTAOPass.h"
#include "MotionVectorPass/MotionVectorPass.h"
#include "TAAPass/TAAPass.h"
#include "DebugViewPass/DebugViewPass.h"
#include "Utils/SampleGenerators/HaltonSamplePattern.h"

using namespace Falcor;

/** Kroken sample: pbrt-v4 interior scene rendered through the Arctic deferred pipeline.

    The scene is imported directly from `assets/kroken/camera-1.pbrt` by the PBRTImporter
    plugin, which is compiled as a plugin DLL and loaded by SampleApp::run() before
    onLoad() runs. The importer's `PBRTImporter:usePBRTMaterials` option is explicitly
    set to false so every pbrt material becomes a StandardMaterial (Metal-Rough), which
    is what the Arctic GBufferPass can encode.

    The render graph mirrors Wheatfield's deferred graph with IndirectApproxPass removed:
      GBuffer -> Skybox / Shadow / ShadowProjection / GTAO -> DIPass / Ambient -> Composite
      -> MotionVector -> TAA -> ToneMapper -> DebugView

    IndirectApproxPass is deliberately absent. CompositePass' `indirect` input is
    optional, and CompositePass::execute() forces the indirect intensity to 0 when that
    input is unconnected, so the graph stays valid without it.

    Lighting note: lights.pbrt contains only an `infinite` light and two area lights --
    there is no DirectionalLight, and DIPass, ShadowPass and ShadowProjectionPass all key
    off the first DirectionalLight in the scene. Without one they would be inert and the
    image would be almost black. This sample therefore adds a synthetic DirectionalLight
    to stand in for the window light. The PBRT importer is left untouched: it drops the
    `float scale` on both the infinite light and the area lights, and that is accepted
    here rather than compensated for.
*/
class Kroken : public SampleApp
{
public:
    Kroken(const SampleAppConfig& config);
    ~Kroken();

    /// Command line is captured here rather than read from the MSVC-only __argc/__argv,
    /// so this sample builds under the Linux/GCC CI job too.
    void setCommandLine(int argc, char** argv);

    void onLoad(RenderContext* pRenderContext) override;
    void onResize(uint32_t width, uint32_t height) override;
    void onFrameRender(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo) override;
    void onGuiRender(Gui* pGui) override;
    bool onKeyEvent(const KeyboardEvent& keyEvent) override;
    bool onMouseEvent(const MouseEvent& mouseEvent) override;

private:
    void parseCommandLine();
    void loadScene(RenderContext* pRenderContext);
    void buildRenderGraph();
    void syncSunLight();

    // F11 screenshot. SampleApp's own F12 path is unusable here: the target FBO is
    // RGBA16Float (see runMain) and Falcor's PNG writer feeds the raw half-float bytes to
    // FreeImage as integers, so F12 produces garbage. This blits through an
    // RGBA8UnormSrgb intermediate first, which both converts to 8-bit and applies the
    // linear->sRGB encoding the FP16 swapchain otherwise leaves to the display.
    void saveScreenshot(RenderContext* pRenderContext, const ref<Texture>& pSrc);

    ref<Scene> mpScene;
    ref<RenderGraph> mpGraph;
    ref<DirectionalLight> mpSunLight;

    float3 mSunDirection = float3(0.866f, -0.5f, 0);
    float mSunIntensity = 7.5f;

    // Scene-derived scale. Arctic's passes default to values tuned for a scene tens of
    // units across (Wheatfield is 50 m of ground); Kroken is an interior spanning several
    // hundred units, so the world-space bias and filter radii are scaled from the scene
    // bounds at load time rather than left at their Wheatfield-era defaults.
    float mSceneRadius = 1.f;
    float mShadowTexelWorld = 1.f;

    uint32_t mShadowResolution = 2048;
    float mShadowConstantDepthBias = 0.01f;
    float mShadowSlopeDepthBias = 0.5f;
    float mShadowMaxSlopeDepthBias = 2.0f;
    float mShadowProjectionFilterRadiusWorld = 0.06f;
    // Width of the depth comparison band, in shadowmap texels. UE computes this
    // as r.Shadow.CSMDepthBias worth of world texels (10 by default) in
    // FProjectedShadowInfo::ComputeTransitionSize, and for CSM there is no
    // separate soft-transition-scale knob: it is derived from the same depth bias
    // the shadow pass uses. Matched here for the same reason — the band has to be
    // wide enough to absorb the sub-texel depth error that would otherwise show up
    // as acne.
    float mShadowProjectionSoftTransitionTexels = 10.f;

    bool mTAAEnabled = true;

    // Auto-capture for headless validation.
    bool mAutoCapture = false;
    uint32_t mFrameCount = 0;
    uint32_t mCaptureFrame = 256;
    uint32_t mCaptureBufferIdx = 0;
    uint32_t mCaptureChannelIdx = 0;
    std::string mCaptureBufferName = "final";
    std::string mCaptureChannelName = "RGB";

    bool mScreenshotRequested = false;

    int mArgc = 0;
    char** mArgv = nullptr;

    // Asset path, relative to the assets/ search path added in onLoad().
    static constexpr const char* kKrokenScenePath = "kroken/camera-1.pbrt";
    static constexpr const char* kEnvMapPath = "env/Studio_Soft.hdr";
};
