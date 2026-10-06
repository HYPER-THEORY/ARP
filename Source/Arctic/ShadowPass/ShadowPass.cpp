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
#include "Falcor.h"
#include "RenderGraph/RenderPassStandardFlags.h"
#include "ShadowPass.h"
#include "ShadowViewData.h"
#include "Scene/Lights/Light.h"

namespace
{
const std::string kShadowPassProgramFile = "Arctic/ShadowPass/ShadowPass.3d.slang";
const RasterizerState::CullMode kDefaultCullMode = RasterizerState::CullMode::None;

const ChannelList kOutputs = {
    { "shadowDepth", "gShadowDepth", "Linearized shadow depth (R32Float)", false, ResourceFormat::R32Float },
};

const std::string kDepthName = "shadowDepth";

const char kShadowResolution[] = "shadowResolution";
const char kUseAlphaTest[] = "useAlphaTest";
const char kConstantDepthBias[] = "constantDepthBias";
const char kSlopeDepthBias[] = "slopeDepthBias";
const char kMaxSlopeDepthBias[] = "maxSlopeDepthBias";
} // namespace

ShadowPass::ShadowPass(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    if (!mpDevice->isShaderModelSupported(ShaderModel::SM6_2))
        FALCOR_THROW("ShadowPass requires Shader Model 6.2 support.");

    parseProperties(props);

    mShadowPass.pState = GraphicsState::create(mpDevice);

    // Shadow depth pass: hardware depth test (LessEqual) + write. The shadowmap
    // COLOR target stores linear light depth in meters written by the PS, but
    // the hardware depth buffer is still needed for occlusion: without it,
    // DepthFunc::Always lets every draw call's PS run and overwrite the color,
    // so draw order decides which geometry wins the shadowmap.
    DepthStencilState::Desc dsDesc;
    dsDesc.setDepthFunc(ComparisonFunc::LessEqual).setDepthWriteMask(true);
    ref<DepthStencilState> pDsState = DepthStencilState::create(dsDesc);
    mShadowPass.pState->setDepthStencilState(pDsState);

    mpFbo = Fbo::create(mpDevice);
}

void ShadowPass::parseProperties(const Properties& props)
{
    for (const auto& [key, value] : props)
    {
        if (key == kShadowResolution) mShadowResolution = value;
        else if (key == kUseAlphaTest) mUseAlphaTest = value;
        else if (key == kConstantDepthBias) mConstantDepthBias = value;
        else if (key == kSlopeDepthBias) mSlopeDepthBias = value;
        else if (key == kMaxSlopeDepthBias) mMaxSlopeDepthBias = value;
    }
}

Properties ShadowPass::getProperties() const
{
    Properties props;
    props[kShadowResolution] = mShadowResolution;
    props[kUseAlphaTest] = mUseAlphaTest;
    props[kConstantDepthBias] = mConstantDepthBias;
    props[kSlopeDepthBias] = mSlopeDepthBias;
    props[kMaxSlopeDepthBias] = mMaxSlopeDepthBias;
    return props;
}

RenderPassReflection ShadowPass::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    const uint2 sz = uint2(mShadowResolution, mShadowResolution);
    addRenderPassOutputs(reflector, kOutputs, ResourceBindFlags::RenderTarget, sz);
    return reflector;
}

void ShadowPass::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mUpdateFlagsConnection = {};
    mUpdateFlags = IScene::UpdateFlags::None;

    mpScene = pScene;
    recreatePrograms();

    if (pScene)
    {
        mUpdateFlagsConnection = mpScene->getUpdateFlagsSignal().connect([&](IScene::UpdateFlags flags) { mUpdateFlags |= flags; });
    }
}

void ShadowPass::recreatePrograms()
{
    mShadowPass.pProgram = nullptr;
    mShadowPass.pVars = nullptr;
}

void ShadowPass::updateLightView()
{
    mLightViewValid = false;
    if (!mpScene) return;

    // Find first DirectionalLight in the scene.
    const auto& lights = mpScene->getLights();
    DirectionalLight* pDirLight = nullptr;
    for (const auto& l : lights)
    {
        if (l && l->getType() == LightType::Directional)
        {
            pDirLight = dynamic_cast<DirectionalLight*>(l.get());
            if (pDirLight) break;
        }
    }
    if (!pDirLight) return;

    const AABB bounds = mpScene->getSceneBounds();
    if (!bounds.valid()) return;

    mLightDir = normalize(pDirLight->getWorldDirection());

    // Build the light basis. `f` is the light travel direction; `up` is any
    // vector not parallel to it.
    float3 up = float3(0.f, 0.f, 1.f);
    if (std::abs(dot(mLightDir, up)) > 0.99f) up = float3(0.f, 1.f, 0.f);
    const float3 f = mLightDir;                // forward (light travel dir)
    const float3 r = normalize(cross(f, up));  // right
    const float3 u = cross(r, f);              // true up

    // Fit the orthographic frustum to the LIGHT-SPACE bounding box of the
    // scene AABB's 8 corners, rather than to its circumsphere.
    //
    // The previous code used `radius = length(extent) * 0.5` (the AABB's
    // circumradius) and then padded by another 1.5x. The circumsphere is a very
    // poor fit for a wide, flat subject, and the 1.5x pad then wasted another
    // 2.25x of area on top.
    const float3 center = bounds.center();
    AABB lsBounds; // light-space box, relative to `center`
    for (int i = 0; i < 8; ++i)
    {
        const float3 corner = float3(
            (i & 1) ? bounds.maxPoint.x : bounds.minPoint.x,
            (i & 2) ? bounds.maxPoint.y : bounds.minPoint.y,
            (i & 4) ? bounds.maxPoint.z : bounds.minPoint.z);
        const float3 d = corner - center;
        lsBounds.include(float3(dot(r, d), dot(u, d), dot(f, d)));
    }

    // Lateral half-extents, padded by kBorderTexels so the PCF/disk kernel can
    // reach past the subject without clamping against real geometry at the
    // shadowmap border. Padding is computed from the *unpadded* texel size,
    // which is accurate enough at these ratios.
    const float kBorderTexels = 8.f;
    const float res = float(std::max(mShadowResolution, 1u));
    float2 halfExtent = float2(
        std::max(0.5f * (lsBounds.maxPoint.x - lsBounds.minPoint.x), 1e-3f),
        std::max(0.5f * (lsBounds.maxPoint.y - lsBounds.minPoint.y), 1e-3f));
    halfExtent += halfExtent * (2.f * kBorderTexels / res);
    mOrthoHalfExtent = halfExtent;

    const float2 lsCenter = float2(
        0.5f * (lsBounds.minPoint.x + lsBounds.maxPoint.x),
        0.5f * (lsBounds.minPoint.y + lsBounds.maxPoint.y));

    // Depth range along the light axis, with headroom on both sides.
    //
    // The reference point for the light-space depth metric is placed just in
    // front of the nearest geometry instead of outside the scene: depths then
    // start near 0 rather than at ~2x the scene radius, which buys back the
    // float mantissa the old offset was burning. Bias values are unaffected — they are
    // deltas.
    const float depthRange = std::max(lsBounds.maxPoint.z - lsBounds.minPoint.z, 1e-3f);
    // Generous near padding: this is also what keeps geometry from being
    // clipped against the near plane if it ever moves outside the reported
    // scene bounds (e.g. if a wind/WPO vertex offset is added later, which
    // getSceneBounds() would not account for). UE handles that case with
    // bClampToNearPlane; padding the frustum is the cheaper equivalent here.
    const float zPad = std::max(depthRange * 0.05f, 1.f);
    mLightPos = center + f * (lsBounds.minPoint.z - zPad - dot(f, center));

    // With that reference point, depth(p) = dot(f, p - mLightPos) lies in
    // [zPad, depthRange + zPad] for all scene geometry.
    const float zNear = 0.f;
    const float zFar = depthRange + 2.f * zPad;

    // Upper bound on any value the pixel shader can write: the farthest
    // geometry depth PLUS the largest bias it can add. ShadowProjectionPass
    // treats sampled values above this as "never written". Without the bias
    // headroom, the surfaces farthest from the light write depth+bias just
    // past the threshold and stop occluding entirely.
    mMaxSubjectDepth = (depthRange + zPad) + (mConstantDepthBias + mSlopeDepthBias * mMaxSlopeDepthBias) + 1.f;

    // View matrix (camera looks along +f, so rows: r, u, -f).
    float4x4 view = float4x4::identity();
    view[0] = float4(r, -dot(r, mLightPos));
    view[1] = float4(u, -dot(u, mLightPos));
    view[2] = float4(-f, dot(f, mLightPos));
    view[3] = float4(0.f, 0.f, 0.f, 1.f);

    // Off-center orthographic projection.
    //   clip.x = (x_view - lsCenter.x) / halfExtent.x
    //   clip.z = (depth - zNear) / (zFar - zNear)     [depth = -z_view]
    // RH view space has z<0 in front of the camera and Falcor uses the D3D
    // depth range [0,1] (0=near), so proj[2].z must be NEGATIVE to map
    // -zNear -> 0 and -zFar -> 1. A positive sign flips the mapping and pushes
    // all geometry outside [0,1], where it gets clipped and the shadowmap
    // stays at the cleared value (= unshadowed).
    //
    // NOTE: this matrix only positions geometry in the shadowmap viewport (it
    // produces SV_POSITION for the rasterizer). The depth VALUE stored in the
    // shadowmap is computed in the pixel shader as dot(gLightDir, posW -
    // gLightPos): a light-space LINEAR depth in METERS. See ShadowPass.3d.slang
    // for why that differs from UE's normalized units.
    float4x4 proj = float4x4::identity();
    proj[0] = float4(1.f / halfExtent.x, 0.f, 0.f, -lsCenter.x / halfExtent.x);
    proj[1] = float4(0.f, 1.f / halfExtent.y, 0.f, -lsCenter.y / halfExtent.y);
    proj[2] = float4(0.f, 0.f, -1.f / (zFar - zNear), -zNear / (zFar - zNear));
    proj[3] = float4(0.f, 0.f, 0.f, 1.f);

    // Falcor is column-vector: clip = proj * view * world * pos, so the
    // combined matrix is mul(proj, view). mul(view, proj) would apply view
    // AFTER proj and produce garbage.
    mLightViewProj = mul(proj, view);
    mLightViewValid = true;
}

ShadowViewData ShadowPass::getShadowViewData() const
{
    ShadowViewData d;
    d.lightViewProj = mLightViewProj;
    d.lightPos = mLightPos;
    d.lightDir = mLightDir;
    d.maxSubjectDepth = mMaxSubjectDepth;
    d.orthoHalfExtent = mOrthoHalfExtent;
    d.valid = mLightViewValid;
    return d;
}

void ShadowPass::renderUI(Gui::Widgets& widget)
{
    if (widget.var("Shadow resolution", mShadowResolution, 256u, 8192u))
    {
        mOptionsChanged = true;
        requestRecompile();
    }
    mOptionsChanged |= widget.checkbox("Alpha Test", mUseAlphaTest);
    widget.tooltip("Use alpha testing on masked geometry so transparent pixels do not write shadow depth.");
    // Ranges in METERS (shadowmap stores light-space linear depth in meters).
    mOptionsChanged |= widget.var("Constant bias (m)", mConstantDepthBias, 0.f, 1.f, 0.001f);
    mOptionsChanged |= widget.var("Slope bias (m)", mSlopeDepthBias, 0.f, 5.f, 0.01f);
    mOptionsChanged |= widget.var("Max slope bias (m)", mMaxSlopeDepthBias, 0.f, 10.f, 0.1f);
}

void ShadowPass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (mOptionsChanged)
    {
        auto& dict = renderData.getDictionary();
        auto flags = dict.getValue(kRenderPassRefreshFlags, RenderPassRefreshFlags::None);
        dict[Falcor::kRenderPassRefreshFlags] = flags | Falcor::RenderPassRefreshFlags::RenderOptionsChanged;
        mOptionsChanged = false;
    }

    // Recompute the light view every frame and publish it unconditionally —
    // including when there is no scene or no directional light, in which case
    // updateLightView() leaves ShadowViewData::valid false. Publishing before
    // any early-out keeps ShadowProjectionPass from reading a value left over
    // from a previous frame.
    updateLightView();
    renderData.getDictionary()[kShadowViewDataKey] = getShadowViewData();

    auto pShadowDepth = renderData.getTexture(kDepthName);
    FALCOR_ASSERT(pShadowDepth);
    FALCOR_ASSERT(pShadowDepth->getWidth() == mShadowResolution && pShadowDepth->getHeight() == mShadowResolution);

    // (Re)allocate the internal depth attachment to match the shadow resolution.
    // The graph allocates the R32Float color target via reflect(); we own the
    // depth buffer locally since it's not consumed by any other pass.
    if (!mpDepthTexture || mDepthTexRes != mShadowResolution)
    {
        mpDepthTexture = mpDevice->createTexture2D(
            mShadowResolution, mShadowResolution,
            ResourceFormat::D32Float,
            1, 1, nullptr,
            ResourceBindFlags::DepthStencil);
        mDepthTexRes = mShadowResolution;
    }

    // Clear shadow depth color far beyond any depth the pixel shader can write
    // (mMaxSubjectDepth already includes the maximum bias). ShadowProjectionPass
    // treats sampled values above that threshold as "no geometry written here"
    // and returns unshadowed.
    mpFbo->attachColorTarget(pShadowDepth, 0);
    mpFbo->attachDepthStencilTarget(mpDepthTexture);
    // Clear color (1e9, far beyond MaxSubjectDepth) and depth (1.0 = far).
    pRenderContext->clearFbo(mpFbo.get(), float4(1e9f, 0.f, 0.f, 0.f), 1.f, 0, FboAttachmentType::All);

    if (mpScene == nullptr) return;

    if (is_set(mpScene->getUpdates(), IScene::UpdateFlags::RecompileNeeded))
    {
        recreatePrograms();
    }

    if (!mShadowPass.pProgram)
    {
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kShadowPassProgramFile).vsEntry("vsMain").psEntry("psMain");
        desc.addTypeConformances(mpScene->getTypeConformances());

        mShadowPass.pProgram = Program::create(mpDevice, desc, mpScene->getSceneDefines());
        mShadowPass.pState->setProgram(mShadowPass.pProgram);
    }

    mShadowPass.pProgram->addDefine("USE_ALPHA_TEST", mUseAlphaTest ? "1" : "0");

    if (!mShadowPass.pVars)
        mShadowPass.pVars = ProgramVars::create(mpDevice, mShadowPass.pProgram.get());

    auto var = mShadowPass.pVars->getRootVar();
    var["PerFrameCB"]["gLightViewProj"] = mLightViewProj;
    var["PerFrameCB"]["gLightPos"] = mLightPos;
    var["PerFrameCB"]["gLightDir"] = mLightDir;
    var["PerFrameCB"]["gMaxSubjectDepth"] = mMaxSubjectDepth;
    var["PerFrameCB"]["gConstantDepthBias"] = mConstantDepthBias;
    var["PerFrameCB"]["gSlopeDepthBias"] = mSlopeDepthBias;
    var["PerFrameCB"]["gMaxSlopeDepthBias"] = mMaxSlopeDepthBias;

    mShadowPass.pState->setFbo(mpFbo);

    mpScene->rasterize(pRenderContext, mShadowPass.pState.get(), mShadowPass.pVars.get(), kDefaultCullMode);
}
