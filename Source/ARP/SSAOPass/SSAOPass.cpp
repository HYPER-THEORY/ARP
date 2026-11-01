/// Copyright (C) 2026-2027 ARP (Auto Render Pipeline). All Rights Reserved.

#include "SSAOPass.h"
#include "RenderGraph/RenderPassHelpers.h"

namespace
{

const std::string kSSAOShader       = "ARP/SSAOPass/SSAO.cs.slang";
const std::string kSSAOSmoothShader = "ARP/SSAOPass/SSAOSmooth.cs.slang";

const ChannelList kInputChannels = {
    {   "gBufferA",   "gGBufferA",   "GBuffer A", false, ResourceFormat::Unknown },
    { "sceneDepth", "gSceneDepth", "Scene Depth", false, ResourceFormat::Unknown },
    {  "zFurthest",  "gZFurthest",  "Z Furthest", false, ResourceFormat::Unknown },
};

const ChannelList kOutputChannels = {
    {      "aoMap",      "gAOMap",      "AO Map", false, ResourceFormat::R8Unorm },
};

const std::string kIntensity          = "intensity";
const std::string kPower              = "power";
const std::string kRadius             = "radius";
const std::string kBias               = "bias";
const std::string kFadeDistance       = "fadeDistance";
const std::string kFadeRadius         = "fadeRadius";
const std::string kMipLevelFactor     = "mipLevelFactor";
const std::string kRadiusInWorldSpace = "radiusInWorldSpace";
const std::string kTemporal           = "temporal";
const std::string kEnableSmooth       = "enableSmooth";
const std::string kDifferentiable     = "differentiable";

}

ref<SSAOPass> SSAOPass::create(ref<Device> pDevice, const Properties& props)
{
    return make_ref<SSAOPass>(pDevice, props);
}

SSAOPass::SSAOPass(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    setProperties(props);

    {
        Sampler::Desc desc;
        desc.setFilterMode(TextureFilteringMode::Point, TextureFilteringMode::Point, TextureFilteringMode::Point);
        desc.setAddressingMode(TextureAddressingMode::Clamp, TextureAddressingMode::Clamp, TextureAddressingMode::Clamp);
        mpZFurthestSampler = mpDevice->createSampler(desc);
    }

    {
        Sampler::Desc desc;
        desc.setFilterMode(TextureFilteringMode::Point, TextureFilteringMode::Point, TextureFilteringMode::Point);
        desc.setAddressingMode(TextureAddressingMode::Wrap, TextureAddressingMode::Wrap, TextureAddressingMode::Wrap);
        mpRandomNormalSampler = mpDevice->createSampler(desc);
    }

    initRandomNormalTexture();
}

RenderPassReflection SSAOPass::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    addRenderPassInputs(reflector, kInputChannels);
    addRenderPassOutputs(reflector, kOutputChannels);
    reflector.getField("zFurthest")->texture2D(0, 0, 1, 0);
    return reflector;
}

void SSAOPass::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
    mpSSAOPass.reset();
    mpSSAOSmoothPass.reset();
    mFrameIndex = 0;
}

void SSAOPass::compile(RenderContext* pRenderContext, const CompileData& compileData)
{
    if (any(mFrameDim != compileData.defaultTexDims))
    {
        mFrameDim = compileData.defaultTexDims;

        mpAOMap = mpDevice->createTexture2D(
            mFrameDim.x, mFrameDim.y, ResourceFormat::R8Unorm, 1, 1, nullptr, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource);
    }
}

void SSAOPass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    auto pGBufferA   = renderData.getTexture("gBufferA");
    auto pSceneDepth = renderData.getTexture("sceneDepth");
    auto pZFurthest  = renderData.getTexture("zFurthest");
    auto pAOMap      = renderData.getTexture("aoMap");

    if (!pGBufferA || !pSceneDepth || !pZFurthest || !pAOMap || !mpScene) return;

    compilePasses();

    if (!mTemporal) mFrameIndex = 0;

    float2 hzbSize = float2(pZFurthest->getWidth(0), pZFurthest->getHeight(0));
    float2 hzbUVFactor = float2(mFrameDim.x, mFrameDim.y) / hzbSize * 0.5f;

    {
        auto var = mpSSAOPass->getVars()->getRootVar();
        mpScene->bindShaderData(var["gScene"]);

        var["gGBufferA"]                = pGBufferA;
        var["gSceneDepth"]              = pSceneDepth;
        var["gZFurthest"]               = pZFurthest;
        var["gZFurthestSampler"]        = mpZFurthestSampler;
        var["gRandomNormal"]            = mpRandomNormalMap;
        var["gRandomNormalSampler"]     = mpRandomNormalSampler;
        var["gAOMap"]                   = mEnableSmooth ? mpAOMap : pAOMap;

        var["CB"]["intensity"]          = mIntensity;
        var["CB"]["power"]              = mPower;
        var["CB"]["radius"]             = mRadius / (mRadiusInWorldSpace ? 4.0f : 1600.0f);
        var["CB"]["bias"]               = mBias / 1000.0f;
        var["CB"]["fadeMul"]            = 1.0f / mFadeRadius;
        var["CB"]["fadeAdd"]            = -mFadeDistance / mFadeRadius;
        var["CB"]["mipLevelFactor"]     = mMipLevelFactor;
        var["CB"]["radiusInWorldSpace"] = mRadiusInWorldSpace ? 1.0f : 0.0f;
        var["CB"]["frameDim"]           = mFrameDim;
        var["CB"]["temporalOffset"]     = float2(0.03875f, 0.1175f) * float(mFrameIndex % 8);
        var["CB"]["hzbUVFactor"]        = hzbUVFactor;

        mpSSAOPass->execute(pRenderContext, uint3(mFrameDim, 1));
    }

    if (mEnableSmooth)
    {
        pRenderContext->uavBarrier(mpAOMap.get());

        auto var = mpSSAOSmoothPass->getVars()->getRootVar();

        var["gAOMap"]         = mpAOMap;
        var["gSmoothedAOMap"] = pAOMap;
        var["CB"]["frameDim"] = mFrameDim;

        mpSSAOSmoothPass->execute(pRenderContext, uint3(mFrameDim, 1));
    }

    ++mFrameIndex;
}

void SSAOPass::setProperties(const Properties& props)
{
    if (props.has(kIntensity))          mIntensity          = props[kIntensity];
    if (props.has(kPower))              mPower              = props[kPower];
    if (props.has(kRadius))             mRadius             = props[kRadius];
    if (props.has(kBias))               mBias               = props[kBias];
    if (props.has(kFadeDistance))       mFadeDistance       = props[kFadeDistance];
    if (props.has(kFadeRadius))         mFadeRadius         = props[kFadeRadius];
    if (props.has(kMipLevelFactor))     mMipLevelFactor     = props[kMipLevelFactor];
    if (props.has(kRadiusInWorldSpace)) mRadiusInWorldSpace = props[kRadiusInWorldSpace];
    if (props.has(kEnableSmooth))       mEnableSmooth       = props[kEnableSmooth];
    if (props.has(kTemporal))           mTemporal           = props[kTemporal];
    if (props.has(kDifferentiable))   { mDifferentiable     = props[kDifferentiable]; mpSSAOPass.reset(); }
}

Properties SSAOPass::getProperties() const
{
    Properties props;
    props[kIntensity]          = mIntensity;
    props[kPower]              = mPower;
    props[kRadius]             = mRadius;
    props[kBias]               = mBias;
    props[kFadeDistance]       = mFadeDistance;
    props[kFadeRadius]         = mFadeRadius;
    props[kMipLevelFactor]     = mMipLevelFactor;
    props[kRadiusInWorldSpace] = mRadiusInWorldSpace;
    props[kEnableSmooth]       = mEnableSmooth;
    props[kTemporal]           = mTemporal;
    props[kDifferentiable]     = mDifferentiable;
    return props;
}

void SSAOPass::renderUI(RenderContext* pRenderContext, Gui::Widgets& widget)
{
    widget.var("Intensity", mIntensity, 0.f, 5.f, 0.01f);
    widget.var("Power", mPower, 0.1f, 5.f, 0.01f);
    widget.checkbox("Radius in World Space", mRadiusInWorldSpace);
    widget.var("Radius", mRadius, 1.f, 100000.f, 1.f);
    widget.var("Bias", mBias, 0.f, 10.f, 0.01f);
    widget.var("Fade Distance", mFadeDistance, 1.f, 100000.f, 1.f);
    widget.var("Fade Radius", mFadeRadius, 1.f, 100000.f, 1.f);
    widget.var("Mip Level Factor", mMipLevelFactor, 0.f, 100.f, 0.01f);
    widget.checkbox("Enable Smooth", mEnableSmooth);
    widget.checkbox("Temporal", mTemporal);
}

void SSAOPass::compilePasses()
{
    if (!mpSSAOPass)
    {
        DefineList defines = mpScene->getSceneDefines();
        if (mDifferentiable) defines.add("__A");

        ProgramDesc desc;
        desc.addShaderLibrary(kSSAOShader).csEntry("main");
        mpSSAOPass = ComputePass::create(mpDevice, desc, defines);
    }

    if (mEnableSmooth && !mpSSAOSmoothPass)
    {
        DefineList defines = mpScene->getSceneDefines();

        ProgramDesc desc;
        desc.addShaderLibrary(kSSAOSmoothShader).csEntry("main");
        mpSSAOSmoothPass = ComputePass::create(mpDevice, desc, defines);
    }
}

void SSAOPass::initRandomNormalTexture()
{
    constexpr int kReorder[16] = {0, 11, 7, 3, 10, 4, 15, 12, 6, 8, 1, 14, 13, 2, 9, 5};

    using RG = math::vector<uint8_t, 2>;

    RG bases[16];

    for (int pos = 0; pos < 16; ++pos)
    {
        float w = kReorder[pos];
        float ww = w / 16.0f * M_PI;
        float lenm = 1.0f - (std::sin(w * 1.98f) * 0.5f + 0.5f) * 0.23f;
        float c = std::cos(ww) * lenm;
        float s = std::sin(ww) * lenm;
        bases[pos].r = uint8_t((c * 0.5f + 0.5f) * 255.f + 0.5f);
        bases[pos].g = uint8_t((s * 0.5f + 0.5f) * 255.f + 0.5f);
    }

    const uint32_t kExtent = 64;

    std::vector<RG> dest(kExtent * kExtent);

    for (int y = 0; y < kExtent; ++y)
    {
        for (int x = 0; x < kExtent; ++x)
        {
            dest[y * kExtent + x] = bases[(x % 4) + (y % 4) * 4];
        }
    }

    mpRandomNormalMap = mpDevice->createTexture2D(
        kExtent, kExtent, ResourceFormat::RG8Unorm, 1, 1, dest.data(), ResourceBindFlags::ShaderResource);
}
