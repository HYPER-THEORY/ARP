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
#include "ShadowViewData.h"

using namespace Falcor;

/** Raster pass that renders the scene from the first DirectionalLight's
    point of view into an R32Float shadow depth texture (UE-style linearized
    depth). The light's orthographic frustum is recomputed every frame and
    fitted to the light-space bounding box of the scene AABB's 8 corners.

    The vertex shader is position-only but still goes through Scene::rasterize
    so it reuses Falcor's mesh VAO and gScene parameter block. Alpha testing
    reuses gScene.materials.alphaTest so masked geometry (AlphaMode::Mask)
    does not write spurious shadow depth.

    The computed light view parameters are published to the RenderData
    dictionary as a ShadowViewData (see ShadowViewData.h) so that
    ShadowProjectionPass re-projects with the identical transform.
*/
class ShadowPass : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(ShadowPass, "ShadowPass", "Directional light shadow depth pass (UE-style linearized depth).");

    static ref<ShadowPass> create(ref<Device> pDevice, const Properties& props) { return make_ref<ShadowPass>(pDevice, props); }

    ShadowPass(ref<Device> pDevice, const Properties& props);

    RenderPassReflection reflect(const CompileData& compileData) override;
    void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    void renderUI(Gui::Widgets& widget) override;
    void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;
    Properties getProperties() const override;

private:
    void parseProperties(const Properties& props);
    void recreatePrograms();
    void updateLightView();
    ShadowViewData getShadowViewData() const;

    ref<Scene> mpScene;
    ref<Fbo> mpFbo;
    // Internal depth attachment for the shadow FBO. The shadowmap COLOR target
    // (R32Float) stores linear light depth in meters written by the pixel
    // shader; without a hardware depth buffer, DepthFunc::Always lets every
    // draw call's PS run and overwrite the color, so draw order decides which
    // geometry's depth wins. With a real depth buffer + LessEqual test, closer
    // geometry's PS wins regardless of draw order — required when a small
    // opaque caster overlaps a large receiver in light clip space.
    ref<Texture> mpDepthTexture;
    uint32_t mDepthTexRes = 0;
    sigs::Connection mUpdateFlagsConnection;
    IScene::UpdateFlags mUpdateFlags = IScene::UpdateFlags::None;

    struct
    {
        ref<GraphicsState> pState;
        ref<Program> pProgram;
        ref<ProgramVars> pVars;
    } mShadowPass;

    // Shadow view parameters (recomputed every frame, published via
    // ShadowViewData). See ShadowPass::updateLightView for how the orthographic
    // frustum is fitted.
    float4x4 mLightViewProj = float4x4::identity();
    float3 mLightDir = float3(0.f, -1.f, 0.f);
    float3 mLightPos = float3(0.f);
    // Upper bound on any depth the pixel shader can write, INCLUDING the
    // largest possible bias. ShadowProjectionPass treats sampled values above
    // this as "no geometry was written" (the shadowmap is cleared far past it).
    // The bias headroom matters: without it, the surfaces farthest from the
    // light write depth+bias just past the threshold and stop occluding.
    float mMaxSubjectDepth = 1.f;
    // Half-extent of the fitted orthographic frustum along the light's
    // right/up axes, in world units. Published so the projection pass can turn
    // a filter radius in meters into shadowmap UV.
    float2 mOrthoHalfExtent = float2(1.f);
    // False until a directional light and valid scene bounds have been found.
    bool mLightViewValid = false;
    uint32_t mShadowResolution = 2048;

    // Depth bias in METERS. NOTE: UE's equivalents (ShadowParams.x/y/z) are
    // dimensionless — its directional shadowmap stores 1 - z_clip normalized
    // over the cascade's depth range, so its defaults are NOT transferable
    // here. Only the FORM of the bias is ported from UE
    // (ShadowDepthVertexShader.usf SetShadowDepthOutputs): a constant term plus
    // SlopeDepthBias * tan(angle between the vertex normal and the light),
    // clamped by MaxSlopeDepthBias. Working in meters instead is deliberate —
    // the values stay meaningful when the frustum size changes.
    float mConstantDepthBias = 0.01f;
    float mSlopeDepthBias = 0.5f;
    float mMaxSlopeDepthBias = 2.0f;

    bool mUseAlphaTest = true;
    bool mOptionsChanged = false;
};
