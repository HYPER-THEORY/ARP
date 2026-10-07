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
#include "Falcor.h"
#include "AmbientPass.h"

namespace
{
const std::string kProgramFile = "Arctic/AmbientPass/AmbientPass.3d.slang";
const std::string kIrradianceProgramFile = "Arctic/AmbientPass/IrradianceConvolve.cs.slang";
const std::string kPreIntegratedGFProgramFile = "Arctic/AmbientPass/PreIntegratedGF.cs.slang";

const ChannelList kInputs = {
    { "gBufferA", "gGBufferA", "World normal + per-object", true, ResourceFormat::Unknown },
    { "gBufferB", "gGBufferB", "Metallic/Specular/Roughness/ShadingModelID", true, ResourceFormat::Unknown },
    { "gBufferC", "gGBufferC", "Base color + AO", true, ResourceFormat::Unknown },
    { "gBufferD", "gGBufferD", "Custom data (subsurface for foliage)", true, ResourceFormat::Unknown },
    { "sceneDepth", "gSceneDepth", "GBuffer scene depth", true, ResourceFormat::Unknown },
    { "aoMap", "gAOMap", "Screen-space AO in .r", true, ResourceFormat::Unknown },
};
const ChannelList kOutputs = {
    { "ambient", "gAmbient", "Sky direct (single-bounce ambient IBL, RGBA32Float HDR)", false, ResourceFormat::RGBA32Float },
};

// Irradiance map resolution. A 64x32 lat-long is 5.6 degrees per texel, finer than
// the 8x8 cube face UE's diffuse mip resolves to (11.25 degrees), and true irradiance
// has no angular detail beyond the first few spherical-harmonic bands anyway. Finer
// than that would only cost build time.
constexpr uint32_t kIrradianceWidth = 64;
constexpr uint32_t kIrradianceHeight = 32;

// Cosine-weighted samples per irradiance texel. The env map already has mips, so the
// source LOD does most of the variance reduction; this only has to resolve the
// hemisphere-integral shape.
constexpr uint32_t kIrradianceSamples = 512;

// UE SystemTextures.cpp: the PreintegratedGF table is 128x32 (the 128x128 variant is
// behind `bReference`, off by default) filled with 128 samples per texel.
constexpr uint32_t kPreIntegratedGFWidth = 128;
constexpr uint32_t kPreIntegratedGFHeight = 32;
constexpr uint32_t kPreIntegratedGFSamples = 128;

const char kTint[] = "tint";
const char kIntensity[] = "intensity";
const char kUseFixedColor[] = "useFixedColor";
const char kAmbientColor[] = "ambientColor";
} // namespace

AmbientPass::AmbientPass(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    for (const auto& [key, value] : props)
    {
        if (key == kTint) mTint = value;
        else if (key == kIntensity) mIntensity = value;
        else if (key == kUseFixedColor) mUseFixedColor = value;
        else if (key == kAmbientColor) mAmbientColor = value;
    }

    // Ambient IBL binds both environment-derived resources unconditionally, so both
    // need a valid placeholder before the first update. A 1x1 black irradiance keeps
    // the descriptor valid when the scene has no environment map (or in fixed-color
    // mode), where the shader would otherwise sample an unbound resource.
    mpIrradiance = mpDevice->createTexture2D(1, 1, ResourceFormat::RGBA32Float, 1, 1, nullptr, ResourceBindFlags::ShaderResource);
    mpIrradiance->setName("AmbientPass.irradiancePlaceholder");

    Sampler::Desc irradianceSamplerDesc;
    irradianceSamplerDesc.setFilterMode(TextureFilteringMode::Linear, TextureFilteringMode::Linear, TextureFilteringMode::Linear);
    // The lat-long map wraps horizontally but not vertically.
    irradianceSamplerDesc.setAddressingMode(TextureAddressingMode::Wrap, TextureAddressingMode::Clamp, TextureAddressingMode::Clamp);
    mpIrradianceSampler = mpDevice->createSampler(irradianceSamplerDesc);

    Sampler::Desc gfSamplerDesc;
    gfSamplerDesc.setFilterMode(TextureFilteringMode::Linear, TextureFilteringMode::Linear, TextureFilteringMode::Linear);
    gfSamplerDesc.setAddressingMode(TextureAddressingMode::Clamp, TextureAddressingMode::Clamp, TextureAddressingMode::Clamp);
    mpPreIntegratedGFSampler = mpDevice->createSampler(gfSamplerDesc);
}

Properties AmbientPass::getProperties() const
{
    Properties props;
    props[kTint] = mTint;
    props[kIntensity] = mIntensity;
    props[kUseFixedColor] = mUseFixedColor;
    props[kAmbientColor] = mAmbientColor;
    return props;
}

RenderPassReflection AmbientPass::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    const uint2 sz = compileData.defaultTexDims;
    addRenderPassInputs(reflector, kInputs, ResourceBindFlags::ShaderResource, uint2(0));
    addRenderPassOutputs(reflector, kOutputs, ResourceBindFlags::UnorderedAccess, sz);
    return reflector;
}

void AmbientPass::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
    mpPass.reset();
    // The irradiance map was convolved from the previous scene's env map.
    mpIrradianceSrc = nullptr;
    mIrradianceSrcDim = uint2(0);
    mDirty = true;
}

void AmbientPass::rebuildPass()
{
    if (!mpScene) return;
    if (!mpPass)
    {
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kProgramFile).csEntry("csMain");
        desc.addTypeConformances(mpScene->getTypeConformances());
        mpPass = ComputePass::create(mpDevice, desc, mpScene->getSceneDefines());
    }
    mDirty = false;
}

void AmbientPass::updatePreIntegratedGF(RenderContext* pRenderContext)
{
    if (mpPreIntegratedGF) return;

    // RG16Unorm matches UE's PF_G16R16, including its [0,1] clamp on both channels.
    mpPreIntegratedGF = mpDevice->createTexture2D(
        kPreIntegratedGFWidth,
        kPreIntegratedGFHeight,
        ResourceFormat::RG16Unorm,
        1,
        1,
        nullptr,
        ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
    );
    mpPreIntegratedGF->setName("AmbientPass.preIntegratedGF");

    if (!mpPreIntegratedGFPass)
    {
        mpPreIntegratedGFPass = ComputePass::create(mpDevice, kPreIntegratedGFProgramFile, "csMain");
    }

    auto var = mpPreIntegratedGFPass->getVars()->getRootVar();
    var["gTable"] = mpPreIntegratedGF;
    var["PerFrameCB"]["gTableDim"] = uint2(kPreIntegratedGFWidth, kPreIntegratedGFHeight);
    var["PerFrameCB"]["gNumSamples"] = kPreIntegratedGFSamples;

    mpPreIntegratedGFPass->execute(pRenderContext, uint3(kPreIntegratedGFWidth, kPreIntegratedGFHeight, 1));
    pRenderContext->uavBarrier(mpPreIntegratedGF.get());
}

void AmbientPass::updateIrradiance(RenderContext* pRenderContext)
{
    if (!mpScene) return;

    auto pEnvMap = mpScene->getEnvMap();
    auto pEnvTexture = pEnvMap ? pEnvMap->getEnvMap() : nullptr;
    if (!pEnvTexture) return;

    const uint2 srcDim = uint2(pEnvTexture->getWidth(), pEnvTexture->getHeight());

    // Rebuild only when the source texture or its resolution changed. Rotation and
    // intensity/tint are applied at lookup time (EnvMap::toLocal / getIntensity), so
    // they must not invalidate the convolution.
    if (mpIrradiance && mpIrradianceSrc == pEnvTexture && all(mIrradianceSrcDim == srcDim)) return;

    if (!mpIrradiancePass)
    {
        // No scene dependency: the convolution only reads the env map and MathHelpers.
        mpIrradiancePass = ComputePass::create(mpDevice, kIrradianceProgramFile, "csMain");
    }

    if (!mpIrradiance || mpIrradiance->getWidth() != kIrradianceWidth || mpIrradiance->getHeight() != kIrradianceHeight)
    {
        mpIrradiance = mpDevice->createTexture2D(
            kIrradianceWidth,
            kIrradianceHeight,
            ResourceFormat::RGBA32Float,
            1,
            1,
            nullptr,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
        );
        mpIrradiance->setName("AmbientPass.irradiance");
    }

    // Match one sample's footprint to a source texel so a small bright sun cannot
    // land as single-texel spikes in the mean. A cosine sample covers 2*PI/N
    // steradians; a source texel at the equator covers (2*PI/W)*(PI/(W/2)) =
    // 4*PI^2/W^2. Equating the two gives W_m = sqrt(2*PI*N), i.e.
    // lod = log2(W / sqrt(2*PI*N)).
    const float samples = float(kIrradianceSamples);
    const float matchedWidth = std::sqrt(2.f * float(M_PI) * samples);
    float srcLod = std::log2(std::max(float(srcDim.x) / matchedWidth, 1.f));
    srcLod = std::clamp(srcLod, 0.f, float(std::max(pEnvTexture->getMipCount(), 1u) - 1u));

    auto var = mpIrradiancePass->getVars()->getRootVar();
    var["gEnvMap"] = pEnvTexture;
    var["gEnvSampler"] = pEnvMap->getEnvSampler();
    var["gIrradiance"] = mpIrradiance;
    var["PerFrameCB"]["gIrradianceDim"] = uint2(kIrradianceWidth, kIrradianceHeight);
    var["PerFrameCB"]["gNumSamples"] = kIrradianceSamples;
    var["PerFrameCB"]["gSrcLod"] = srcLod;

    mpIrradiancePass->execute(pRenderContext, uint3(kIrradianceWidth, kIrradianceHeight, 1));
    pRenderContext->uavBarrier(mpIrradiance.get());

    mpIrradianceSrc = pEnvTexture;
    mIrradianceSrcDim = srcDim;
}

void AmbientPass::renderUI(Gui::Widgets& widget)
{
    mDirty |= widget.var("Tint", mTint, 0.f, 4.f, 0.01f);
    mDirty |= widget.var("Intensity", mIntensity, 0.f, 16.f, 0.01f);
    mDirty |= widget.checkbox("Use Fixed Color", mUseFixedColor);
    widget.tooltip("When on, ambient IBL uses Ambient Color instead of the envMap.");
    if (mUseFixedColor)
    {
        mDirty |= widget.var("Ambient Color", mAmbientColor, 0.f, 4.f, 0.01f);
    }
}

void AmbientPass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    auto pAmbient = renderData.getTexture("ambient");
    if (!pAmbient || !mpScene) return;

    if (mDirty) rebuildPass();
    if (!mpPass) return;

    updatePreIntegratedGF(pRenderContext);
    updateIrradiance(pRenderContext);

    // Compute envmap mip count from the env map texture dimensions.
    // AmbientCubemapMipAdjust.w = MipCount.
    float mipCount = 1.f;
    if (mpScene->getEnvMap() && mpScene->getEnvMap()->getEnvMap())
    {
        uint32_t w = mpScene->getEnvMap()->getEnvMap()->getWidth();
        // ComputeCubemapMipFromRoughness's MipCount is a *cube face* mip count
        // (CubemapCommon.ush -- "e.g. 10 for x 512x512"), but our env map is a
        // lat-long. A lat-long of width W resolves the same angle per texel as a
        // cube face of W/4 (360/W == 90/(W/4)), so the equivalent count is
        // log2(W/4) + 1 == log2(W) - 1. Using log2(W) + 1 here would land two mips
        // coarser at every roughness, i.e. 4x the angular blur UE would pick.
        //
        // This is also what makes the shader's AbsoluteDiffuseMip = MipCount - 4
        // land on the right mip: for the 2048-wide lat-long it is 10 - 4 = 6, i.e.
        // the same 11.25 degrees per texel as UE's mip 6 of a 512 cube.
        mipCount = std::max(1.f, std::log2(float(w)) - 1.f);
    }

    auto var = mpPass->getVars()->getRootVar();
    mpScene->bindShaderData(var["gScene"]);
    var["gGBufferA"] = renderData.getTexture("gBufferA");
    var["gGBufferB"] = renderData.getTexture("gBufferB");
    var["gGBufferC"] = renderData.getTexture("gBufferC");
    var["gGBufferD"] = renderData.getTexture("gBufferD");
    var["gSceneDepth"] = renderData.getTexture("sceneDepth");
    var["gAOMap"] = renderData.getTexture("aoMap");
    var["gIrradiance"] = mpIrradiance;
    var["gIrradianceSampler"] = mpIrradianceSampler;
    // updatePreIntegratedGF() above guarantees this exists by the time we bind it.
    var["gPreIntegratedGF"] = mpPreIntegratedGF;
    var["gPreIntegratedGFSampler"] = mpPreIntegratedGFSampler;
    var["gAmbient"] = pAmbient;

    auto cb = var["PerFrameCB"];
    cb["gFrameDim"] = uint2(pAmbient->getWidth(), pAmbient->getHeight());
    cb["gTint"] = mTint;
    cb["gIntensity"] = mIntensity;
    cb["gAmbientColor"] = mAmbientColor;
    cb["gUseFixedColor"] = mUseFixedColor ? 1u : 0u;
    cb["gMipCount"] = mipCount;

    mpPass->execute(pRenderContext, uint3(pAmbient->getWidth(), pAmbient->getHeight(), 1));
}
