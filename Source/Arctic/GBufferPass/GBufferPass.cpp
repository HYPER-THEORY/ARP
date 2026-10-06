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
#include "Falcor.h"
#include "RenderGraph/RenderPassStandardFlags.h"
#include "GBufferPass.h"

namespace
{
const std::string kGBufferPassProgramFile = "Arctic/GBufferPass/GBufferPass.3d.slang";

// UE GBuffer MRT layout (5 UE-layout color RTs + emissive + depth). Channel order
// must match SV_TARGET index order in GBufferPass.3d.slang psMain output struct.
const ChannelList kGBufferChannels = {
    // clang-format off
    { "gBufferA", "gGBufferA", "World normal (xyz, encoded N*0.5+0.5) + per-object data (a)", false, ResourceFormat::RGB10A2Unorm },
    { "gBufferB", "gGBufferB", "Metallic (r), Specular (g), Roughness (b), ShadingModelID|SelectiveOutputMask (a)", false, ResourceFormat::RGBA8Unorm },
    { "gBufferC", "gGBufferC", "Base color (rgb, sRGB) + GBufferAO (a)", false, ResourceFormat::RGBA8UnormSrgb },
    { "gBufferD", "gGBufferD", "Custom data (sqrt(subsurfaceColor) rgb + opacity a; 0 for DefaultLit)", false, ResourceFormat::RGBA8Unorm },
    { "gBufferE", "gGBufferE", "Precomputed shadow factors (always 0: no baked lighting)", false, ResourceFormat::RGBA8Unorm },
    // Emissive is not part of UE's GBuffer -- UE's base pass writes it straight
    // into SceneColor. Arctic has no base-pass color target, so it gets its own
    // HDR MRT that CompositePass adds back. It cannot share GBufferE, which is
    // 8-bit UNORM and would clamp anything above 1.
    { "emissive", "gEmissive", "Emissive radiance (RGBA16Float, HDR)", false, ResourceFormat::RGBA16Float },
    // Velocity is produced by the standalone MotionVectorPass and consumed
    // directly by TAAPass, so GBufferPass no longer writes a vvec MRT.
    // clang-format on
};

const std::string kDepthName = "depth";

// Scripting option keys.
const char kUseAlphaTest[] = "useAlphaTest";
const char kAdjustShadingNormals[] = "adjustShadingNormals";
} // namespace

GBufferPass::GBufferPass(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    if (!mpDevice->isShaderModelSupported(ShaderModel::SM6_2))
        FALCOR_THROW("GBufferPass requires Shader Model 6.2 support.");
    if (!mpDevice->isFeatureSupported(Device::SupportedFeatures::Barycentrics))
        FALCOR_THROW("GBufferPass requires pixel shader barycentrics support.");

    parseProperties(props);

    mGBufferPass.pState = GraphicsState::create(mpDevice);

    // This pass has no depth prepass, so use standard LessEqual depth test with writes enabled.
    DepthStencilState::Desc dsDesc;
    dsDesc.setDepthFunc(ComparisonFunc::LessEqual).setDepthWriteMask(true);
    ref<DepthStencilState> pDsState = DepthStencilState::create(dsDesc);
    mGBufferPass.pState->setDepthStencilState(pDsState);

    mpFbo = Fbo::create(mpDevice);
}

void GBufferPass::parseProperties(const Properties& props)
{
    for (const auto& [key, value] : props)
    {
        if (key == kUseAlphaTest)
            mUseAlphaTest = value;
        else if (key == kAdjustShadingNormals)
            mAdjustShadingNormals = value;
    }
}

Properties GBufferPass::getProperties() const
{
    Properties props;
    props[kUseAlphaTest] = mUseAlphaTest;
    props[kAdjustShadingNormals] = mAdjustShadingNormals;
    return props;
}

RenderPassReflection GBufferPass::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    const uint2 sz = compileData.defaultTexDims;

    // Depth output (hardware DSV).
    reflector.addOutput(kDepthName, "Depth buffer")
        .format(ResourceFormat::D32Float)
        .bindFlags(ResourceBindFlags::DepthStencil)
        .texture2D(sz.x, sz.y);

    // Color MRT outputs (GBufferA-E + emissive).
    addRenderPassOutputs(reflector, kGBufferChannels, ResourceBindFlags::RenderTarget, sz);

    return reflector;
}

void GBufferPass::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mUpdateFlagsConnection = {};
    mUpdateFlags = IScene::UpdateFlags::None;

    mpScene = pScene;
    mFrameCount = 0;
    mPrevJitter = {};

    recreatePrograms();

    if (pScene)
    {
        mUpdateFlagsConnection = mpScene->getUpdateFlagsSignal().connect([&](IScene::UpdateFlags flags) { mUpdateFlags |= flags; });

        if (pScene->getMeshVao() && pScene->getMeshVao()->getPrimitiveTopology() != Vao::Topology::TriangleList)
        {
            FALCOR_THROW("GBufferPass: Requires triangle list geometry due to usage of SV_Barycentrics.");
        }
    }
}

void GBufferPass::recreatePrograms()
{
    mGBufferPass.pProgram = nullptr;
    mGBufferPass.pVars = nullptr;
}

void GBufferPass::updateFrameDim(const uint2 frameDim)
{
    FALCOR_ASSERT(frameDim.x > 0 && frameDim.y > 0);
    mFrameDim = frameDim;
}

void GBufferPass::renderUI(Gui::Widgets& widget)
{
    mOptionsChanged |= widget.checkbox("Alpha Test", mUseAlphaTest);
    widget.tooltip("Use alpha testing on non-opaque triangles.");

    mOptionsChanged |= widget.checkbox("Adjust shading normals", mAdjustShadingNormals);
    widget.tooltip("Enables adjustment of the shading normals to reduce the risk of black pixels due to back-facing vectors.", true);
}

void GBufferPass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    // Update refresh flag if options that affect the output have changed.
    if (mOptionsChanged)
    {
        auto& dict = renderData.getDictionary();
        auto flags = dict.getValue(kRenderPassRefreshFlags, RenderPassRefreshFlags::None);
        dict[Falcor::kRenderPassRefreshFlags] = flags | Falcor::RenderPassRefreshFlags::RenderOptionsChanged;
        mOptionsChanged = false;
    }

    auto pDepth = renderData.getTexture(kDepthName);
    FALCOR_ASSERT(pDepth);
    updateFrameDim(uint2(pDepth->getWidth(), pDepth->getHeight()));

    // Clear depth buffer.
    pRenderContext->clearDsv(pDepth->getDSV().get(), 1.f, 0);

    // Bind color targets and clear them.
    for (size_t i = 0; i < kGBufferChannels.size(); ++i)
    {
        ref<Texture> pTex = renderData.getTexture(kGBufferChannels[i].name);
        FALCOR_ASSERT(pTex);
        mpFbo->attachColorTarget(pTex, uint32_t(i));
    }
    pRenderContext->clearFbo(mpFbo.get(), float4(0), 1.f, 0, FboAttachmentType::Color);

    mpFbo->attachDepthStencilTarget(pDepth);

    if (mpScene == nullptr)
    {
        return;
    }

    if (is_set(mpScene->getUpdates(), IScene::UpdateFlags::RecompileNeeded))
    {
        recreatePrograms();
    }

    // GBuffer pass.
    {
        if (!mGBufferPass.pProgram)
        {
            ProgramDesc desc;
            desc.addShaderModules(mpScene->getShaderModules());
            desc.addShaderLibrary(kGBufferPassProgramFile).vsEntry("vsMain").psEntry("psMain");
            desc.addTypeConformances(mpScene->getTypeConformances());

            mGBufferPass.pProgram = Program::create(mpDevice, desc, mpScene->getSceneDefines());
            mGBufferPass.pState->setProgram(mGBufferPass.pProgram);
        }

        mGBufferPass.pProgram->addDefine("ADJUST_SHADING_NORMALS", mAdjustShadingNormals ? "1" : "0");
        mGBufferPass.pProgram->addDefine("USE_ALPHA_TEST", mUseAlphaTest ? "1" : "0");

        if (!mGBufferPass.pVars)
            mGBufferPass.pVars = ProgramVars::create(mpDevice, mGBufferPass.pProgram.get());

        auto var = mGBufferPass.pVars->getRootVar();

        // PixelParameters' per-frame half. A material that dithers its coverage
        // needs a temporal history to integrate the stipple into apparent
        // semi-transparency; the camera's jitter pattern generator is the TAA
        // switch, so it is also the signal for whether such a history exists.
        const ref<Camera>& pCamera = mpScene->getCamera();
        const bool temporalJitterActive = pCamera && pCamera->getPatternGenerator() != nullptr;
        const float2 jitter = temporalJitterActive ? float2(pCamera->getJitterX(), pCamera->getJitterY()) : float2(0.f);

        var["PerFrameCB"]["gFrameDim"] = mFrameDim;
        var["PerFrameCB"]["gIsTemporalEnable"] = temporalJitterActive ? 1u : 0u;
        var["PerFrameCB"]["gJitterFrameIndex"] = (int32_t)mFrameCount;
        // Arctic does not implement the distance-based transparency dither these
        // two drive, so they stay neutral.
        var["PerFrameCB"]["gDitherClipOffset"] = 0.f;
        var["PerFrameCB"]["gTransparencyDistanceThreshold"] = 0.f;
        // Falcor applies jitter as an NDC translation of 2 * (jitterX, jitterY)
        // (Camera::calculateCameraParameters), which is the offset carried here.
        var["PerFrameCB"]["gJitterProjectionOffset"] =
            float4(2.f * jitter.x, 2.f * jitter.y, 2.f * mPrevJitter.x, 2.f * mPrevJitter.y);
        mPrevJitter = jitter;

        // Bush material parameters travel in the material data blob (see
        // BushMaterialData.slang), so this pass has no per-material uniforms to
        // upload and the scene may contain any number of bush materials.

        mGBufferPass.pState->setFbo(mpFbo); // Sets the viewport

        mpScene->rasterize(pRenderContext, mGBufferPass.pState.get(), mGBufferPass.pVars.get());
    }

    mFrameCount++;
}
