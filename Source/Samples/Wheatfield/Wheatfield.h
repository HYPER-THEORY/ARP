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
#include "Core/SampleApp.h"
#include "Core/AssetResolver.h"
#include "RenderGraph/RenderGraph.h"
#include "RenderGraph/RenderPassStandardFlags.h"
#include "Scene/SceneBuilder.h"
#include "Scene/TriangleMesh.h"
#include "Scene/Scene.h"
#include "Scene/Camera/Camera.h"
#include "Scene/Lights/EnvMap.h"
#include "Scene/Lights/Light.h"
#include "Scene/Transform.h"
#include "Scene/Material/StandardMaterial.h"
#include "Materials/Bush/BushMaterial.h"
#include "GBufferPass/GBufferPass.h"
#include "ShadowPass/ShadowPass.h"
#include "ShadowProjectionPass/ShadowProjectionPass.h"
#include "SkyboxPass/SkyboxPass.h"
#include "DIPass/DIPass.h"
#include "AmbientPass/AmbientPass.h"
#include "CompositePass/CompositePass.h"
#include "GTAOPass/GTAOPass.h"
#include "SSAOPass/SSAOPass.h"
#include "MotionVectorPass/MotionVectorPass.h"
#include "TAAPass/TAAPass.h"
#include "DebugViewPass/DebugViewPass.h"

using namespace Falcor;

class Wheatfield : public SampleApp
{
public:
    enum class AOMethod { SSAO, GTAO };

    Wheatfield(const SampleAppConfig& config);
    ~Wheatfield();

    void onLoad(RenderContext* pRenderContext) override;
    void onResize(uint32_t width, uint32_t height) override;
    void onFrameRender(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo) override;
    void onGuiRender(Gui* pGui) override;
    bool onKeyEvent(const KeyboardEvent& keyEvent) override;
    bool onMouseEvent(const MouseEvent& mouseEvent) override;

private:
    void buildRenderGraph();
    void buildPathTracerGraph();
    void buildDeferredGraph();
    void rebuildScene(RenderContext* pRenderContext);

    // Camera pose bookmark (F3 saves, F4 loads). Single slot: F3 overwrites the
    // previous save. Backed by a text file next to the executable so the pose
    // survives a restart. Both are silent; failures are ignored on purpose.
    void saveCameraPose();
    void loadCameraPose();

    // F11 screenshot. Writes the on-screen image as PNG. SampleApp's own F12 path
    // is unusable here: the target FBO is RGBA16Float (see runMain) and Falcor's
    // PNG writer feeds the raw half-float bytes to FreeImage as integers, so F12
    // produces garbage for this app. This one blits through an RGBA8UnormSrgb
    // intermediate first, which both converts to 8-bit and applies the
    // linear->sRGB encoding the FP16 swapchain otherwise leaves to the display.
    void saveScreenshot(RenderContext* pRenderContext, const ref<Texture>& pSrc);

    ref<Scene> mpScene;
    ref<RenderGraph> mpGraph;
    ref<BushMaterial> mpBushMat;
    ref<DirectionalLight> mpSunLight;

    // Ambient-occlusion method selector (UI radio). Default: SSAO.
    AOMethod mAOMethod = AOMethod::GTAO;
    uint32_t mGTAOOutputStage = 2u; // --ao gtao:<stage>, see GTAOPass::OutputStage

    // TAA toggle (UI). When on, Halton(8) camera jitter is applied and the
    // TAAPass resolves temporally before tonemapping.
    bool mTAAEnabled = true;

    // Scene parameters.
    float mWheatDensity = 1.f;     ///< Wheat instances per square meter.
    uint32_t mSeed = 42;
    float mGroundSize = 50.f;      ///< Side length of the ground plane.

    // Per-instance wheat distribution parameters. Not exposed in the UI; tune
    // by editing these defaults. All ranges are half-open [lo, hi).
    //   yaw  : spin around the up axis (Z), degrees.
    //   tilt : lodging lean magnitude, degrees (0 = standing upright).
    //   scale: uniform instance scale.
    //   sink : how far the instance origin is pushed below the ground (Z-).
    // Tilt direction is world-fixed: 0° = +X, counter-clockwise positive,
    // default +90° (leans toward +Y). Each instance picks a direction around
    // the base by ±tiltDirJitter, so a field leans one way with small variation.
    struct Range { float lo, hi; };
    Range mYawRange          = { 90.f,   135.f };
    Range mTiltRange         = { 0.f,    15.f };
    Range mScaleRange        = { 0.75f,   1.25f };
    Range mSinkRange         = { -0.2f,  0.2f };
    float mTiltDirBaseDeg    = 0.f;
    float mTiltDirJitterDeg  = 10.f; ///< ±jitter on lodging direction (deg).

    // Set by F11, serviced at the top of the next onFrameRender. The screenshot
    // has to include the ImGui overlay, which is only drawn after onFrameRender
    // returns, so the request is deliberately one frame late: at that point the
    // target FBO still holds the previous frame's fully composited image.
    bool mScreenshotRequested = false;

    // Auto-capture flag for headless validation.
    bool mAutoCapture = false;
    bool mIndirectProfile = false;
    uint32_t mIndirectDebugView = 1; ///< 1 = E down, 2 = E up (see the pass' UI).
    uint32_t mFrameCount = 0;
    // Parsed --capture argument: which buffer/channel to capture. Defaults to
    // final:RGB so plain `--capture` matches the old behavior.
    uint32_t mCaptureBufferIdx = 0;
    uint32_t mCaptureChannelIdx = 0;
    std::string mCaptureBufferName = "final";
    std::string mCaptureChannelName = "RGB";
    // --captureframe N: frame the snapshot is taken on. 256 gives TAA a converged
    // history; other values exist so consecutive frames can be diffed, which is
    // the only way to observe a temporal effect (dither animation, jitter, TAA).
    uint32_t mCaptureFrame = 256;

    // Asset paths (relative to the assets/ search path).
    static constexpr const char* kWheatAssetPath = "wheat/Wheat_006_LOD0.fbx";
    static constexpr const char* kAlbedoTexturePath = "wheat/textures/Wheat_006_BCA.tga";
    static constexpr const char* kNormalTexturePath = "wheat/textures/Wheat_006_NSR.tga";
    static constexpr const char* kSubsurfaceTexturePath = "wheat/textures/Wheat_006_SCD.tga";
    static constexpr const char* kEnvMapPath = "env/environment_Reflection.exr";
    static constexpr bool kIsOctahedralEnvMap = true;

    // Camera pose bookmark file, resolved against getRuntimeDirectory().
    static constexpr const char* kCameraPoseFileName = "Wheatfield_camera_pose.txt";
};
