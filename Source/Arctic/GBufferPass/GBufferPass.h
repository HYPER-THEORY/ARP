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

/** Rasterized G-buffer pass emitting Unreal Engine's GBuffer MRT layout.

    Outputs color render targets (GBufferA-E) plus a hardware
    depth attachment (D32Float). This pass is intentionally standalone: it does
    not inherit GBufferBase and is not consumed by Falcor's built-in PathTracer.
    It bakes Falcor's StandardMaterial (MetalRough) as DefaultLit and
    BushMaterial as TwoSidedFoliage into UE's GBuffer encoding (see
    GBufferHelpers.slang).
*/
class GBufferPass : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(GBufferPass, "GBufferPass", "Rasterized UE-layout G-buffer generation pass.");

    static ref<GBufferPass> create(ref<Device> pDevice, const Properties& props) { return make_ref<GBufferPass>(pDevice, props); }

    GBufferPass(ref<Device> pDevice, const Properties& props);

    RenderPassReflection reflect(const CompileData& compileData) override;
    void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    void renderUI(Gui::Widgets& widget) override;
    void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;
    Properties getProperties() const override;

private:
    void parseProperties(const Properties& props);
    void recreatePrograms();
    void updateFrameDim(const uint2 frameDim);

    // Internal state
    ref<Scene> mpScene;
    ref<Fbo> mpFbo;
    sigs::Connection mUpdateFlagsConnection;
    IScene::UpdateFlags mUpdateFlags = IScene::UpdateFlags::None;

    struct
    {
        ref<GraphicsState> pState;
        ref<Program> pProgram;
        ref<ProgramVars> pVars;
    } mGBufferPass;

    // Frame state
    uint2 mFrameDim = {};
    uint32_t mFrameCount = 0;
    float2 mPrevJitter = {};    ///< Last frame's camera jitter. Feeds the previous-frame half of the projection offset.

    // UI / config state
    bool mUseAlphaTest = true;
    bool mAdjustShadingNormals = true;
    bool mOptionsChanged = false;
};
