/// Copyright (C) 2026-2027 ARP (Auto Render Pipeline). All Rights Reserved.

#include "ASSAOPass.h"
#include "SSAOHelpers.h"
#include "RenderGraph/RenderPassHelpers.h"

namespace
{

const std::string kASSAOShader = "ARP/SSAOPass/ASSAO.cs.slang";

const ChannelList kInputChannels = {
    { "gBufferA", "gGBufferA", "GBuffer A", false, ResourceFormat::Unknown },
    { "sceneDepth", "gSceneDepth", "Scene Depth", false, ResourceFormat::Unknown },
    { "zFurthest", "gZFurthest", "Z Furthest", false, ResourceFormat::Unknown },
};

const ChannelList kOutputChannels = {
    { "aoMap", "gAOMap", "AO Map", false, ResourceFormat::R32Float },
};

const std::string kIntensity = "intensity";
const std::string kPower = "power";
const std::string kRadius = "radius";
const std::string kBias = "bias";
const std::string kFadeDistance = "fadeDistance";
const std::string kFadeRadius = "fadeRadius";
const std::string kMipLevelFactor = "mipLevelFactor";
const std::string kRadiusInWorldSpace = "radiusInWorldSpace";
const std::string kLockedFrame = "lockedFrame";

/// SSAOPass folds radius and bias into shader units with these factors.
constexpr float kRadiusScale = 1600.0f;
constexpr float kRadiusWorldScale = 4.0f;
constexpr float kBiasScale = 1000.0f;

} // namespace

ref<ASSAOPass> ASSAOPass::create(ref<Device> pDevice, const Properties& props)
{
    return make_ref<ASSAOPass>(pDevice, props);
}

ASSAOPass::ASSAOPass(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    setProperties(props);

    mpRandomNormalSampler = SSAOHelpers::createRandomNormalSampler(mpDevice);
    mpRandomNormalMap = SSAOHelpers::createRandomNormalTexture(mpDevice);

    createBuffers();
}

RenderPassReflection ASSAOPass::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    addRenderPassInputs(reflector, kInputChannels);
    addRenderPassOutputs(reflector, kOutputChannels);

    // The HZB is read through integer loads rather than a sampler, so it needs
    // the real mip chain: the fractional part of the level is what carries the
    // trilinear blend. Mip count 0 keeps whatever the producer declares.
    if (auto* pField = reflector.getField("zFurthest"))
        pField->texture2D(0, 0, 1, 0);

    return reflector;
}

void ASSAOPass::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
    mpForwardPass.reset();
    mpBackwardPass.reset();
    mpReducePass.reset();
    mFrameIndex = 0;
}

void ASSAOPass::compile(RenderContext* pRenderContext, const CompileData& compileData)
{
    mFrameDim = compileData.defaultTexDims;
    createBuffers();
}

void ASSAOPass::compilePasses()
{
    if (!mpScene || mpForwardPass)
        return;

    const DefineList defines = mpScene->getSceneDefines();

    auto createPass = [&](const char* entryPoint)
    {
        ProgramDesc desc;
        desc.addShaderLibrary(kASSAOShader).csEntry(entryPoint);
        return ComputePass::create(mpDevice, desc, defines);
    };

    mpForwardPass = createPass("main");
    mpBackwardPass = createPass("backward");
    mpReducePass = createPass("reduce");
}

void ASSAOPass::createBuffers()
{
    if (mFrameDim.x == 0 || mFrameDim.y == 0)
        return;

    // The reverse pass reduces one group's threads in shared memory and writes one
    // partial vector per group, so the partial buffer is indexed by group count,
    // not by pixel count.
    mGroupCount = uint2(
        div_round_up(mFrameDim.x, kGroupDimX),
        div_round_up(mFrameDim.y, kGroupDimY));
    const size_t groupCount = size_t(mGroupCount.x) * size_t(mGroupCount.y);

    // The GPU reads and writes these as structured buffers, and CUDA can map
    // them, which is what makes Buffer.to_torch() a zero-copy view.
    const auto flags =
        ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Shared;

    // Reallocate only when the element count actually changes: the CUDA mappings
    // and any tensors derived from them do not survive a new buffer.
    auto recreate = [&](ref<Buffer>& pBuffer, size_t elements, const char* name)
    {
        if (!pBuffer || pBuffer->getElementCount() != elements)
        {
            pBuffer = mpDevice->createStructuredBuffer(
                sizeof(float), uint32_t(elements), flags, MemoryType::DeviceLocal, nullptr);
            pBuffer->setName(name);
        }
    };

    recreate(mpGradPartials, groupCount * kGradientCount, "ASSAOPass::mpGradPartials");

    // The reduction's output. Created once; it is never resized.
    if (!mpParamGrads)
    {
        mpParamGrads = mpDevice->createStructuredBuffer(
            sizeof(float), kGradientCount, flags, MemoryType::DeviceLocal, nullptr);
        mpParamGrads->setName("ASSAOPass::mpParamGrads");
    }
}

void ASSAOPass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    auto pGBufferA = renderData.getTexture("gBufferA");
    auto pSceneDepth = renderData.getTexture("sceneDepth");
    auto pZFurthest = renderData.getTexture("zFurthest");
    auto pAOMap = renderData.getTexture("aoMap");

    if (!pGBufferA || !pSceneDepth || !pZFurthest || !pAOMap || !mpScene)
        return;

    mpGBufferA = pGBufferA;
    mpSceneDepth = pSceneDepth;
    mpZFurthest = pZFurthest;
    mpAOMapOutput = pAOMap;

    if (mFrameDim.x == 0 || mFrameDim.y == 0)
        mFrameDim = uint2(pAOMap->getWidth(0), pAOMap->getHeight(0));

    forward(pRenderContext);
}

void ASSAOPass::forward(RenderContext* pRenderContext)
{
    if (!mpGBufferA || !mpSceneDepth || !mpZFurthest || !mpAOMapOutput || !mpScene)
        return;

    compilePasses();
    if (!mpForwardPass)
        return;

    auto var = mpForwardPass->getVars()->getRootVar();
    bindShaderData(var);
    var["gAOMap"] = mpAOMapOutput;

    mpForwardPass->execute(pRenderContext, uint3(mFrameDim, 1));
    ++mFrameIndex;
}

void ASSAOPass::backward(RenderContext* pRenderContext, const ref<Texture>& pdLdI)
{
    if (!mpGBufferA || !mpSceneDepth || !mpZFurthest || !mpScene)
        return;

    FALCOR_CHECK(pdLdI, "ASSAOPass::backward() needs a dL/dAO texture.");
    FALCOR_CHECK(
        pdLdI->getWidth(0) == mFrameDim.x && pdLdI->getHeight(0) == mFrameDim.y,
        "ASSAOPass::backward(): dL/dAO is {}x{}, the AO image is {}x{}.",
        pdLdI->getWidth(0), pdLdI->getHeight(0), mFrameDim.x, mFrameDim.y);

    compilePasses();
    createBuffers();
    if (!mpBackwardPass || !mpGradPartials || !mpParamGrads)
        return;

    dispatchBackward(pRenderContext, pdLdI);
    dispatchReduction(pRenderContext);
}

void ASSAOPass::dispatchBackward(RenderContext* pRenderContext, const ref<Texture>& pdLdI)
{
    auto var = mpBackwardPass->getVars()->getRootVar();
    bindShaderData(var);
    var["gdLdI"] = pdLdI;
    var["gGradPartials"] = mpGradPartials;

    mpBackwardPass->execute(pRenderContext, uint3(mFrameDim, 1));
}

void ASSAOPass::dispatchReduction(RenderContext* pRenderContext)
{
    // The reduction reads only the partials and writes the output, so it binds
    // only those. Its dispatch is a single group of kGroupSize threads, so its
    // shape does not depend on the frame size.
    auto var = mpReducePass->getVars()->getRootVar();
    var["gGradPartials"] = mpGradPartials;
    var["gParamGrads"] = mpParamGrads;
    var["CB"]["groupCount"] = mGroupCount;

    mpReducePass->execute(pRenderContext, uint3(kGroupSize, 1, 1));
}

void ASSAOPass::bindShaderData(const ShaderVar& var)
{
    FALCOR_ASSERT(mpScene);

    mpScene->bindShaderData(var["gScene"]);

    var["gGBufferA"] = mpGBufferA;
    var["gSceneDepth"] = mpSceneDepth;
    var["gZFurthest"] = mpZFurthest;
    var["gRandomNormal"] = mpRandomNormalMap;
    var["gRandomNormalSampler"] = mpRandomNormalSampler;

    const float2 hzbSize = float2(mpZFurthest->getWidth(0), mpZFurthest->getHeight(0));
    const float2 hzbUVFactor = float2(mFrameDim.x, mFrameDim.y) / hzbSize * 0.5f;

    var["CB"]["intensity"] = mIntensity;
    var["CB"]["power"] = mPower;
    var["CB"]["radius"] = mRadius / (mRadiusInWorldSpace ? kRadiusWorldScale : kRadiusScale);
    var["CB"]["bias"] = mBias / kBiasScale;
    var["CB"]["fadeMul"] = 1.0f / mFadeRadius;
    var["CB"]["fadeAdd"] = -mFadeDistance / mFadeRadius;
    var["CB"]["mipLevelFactor"] = mMipLevelFactor;
    var["CB"]["radiusInWorldSpace"] = mRadiusInWorldSpace ? 1.0f : 0.0f;
    var["CB"]["frameDim"] = mFrameDim;
    var["CB"]["temporalOffset"] =
        mLockedFrame ? float2(0.f, 0.f) : float2(2.48f, 7.52f) * float(mFrameIndex % 8) / 64.0f;
    var["CB"]["hzbUVFactor"] = hzbUVFactor;
    // dispatchReduction() also sets this, since it is the one dispatch that does
    // not call bindShaderData().
    var["CB"]["groupCount"] = mGroupCount;
}

void ASSAOPass::setProperties(const Properties& props)
{
    if (props.has(kIntensity))          mIntensity          = props[kIntensity];
    if (props.has(kPower))              mPower              = props[kPower];
    if (props.has(kRadius))             mRadius             = props[kRadius];
    if (props.has(kBias))               mBias               = props[kBias];
    if (props.has(kFadeDistance))       mFadeDistance       = props[kFadeDistance];
    if (props.has(kFadeRadius))         mFadeRadius         = props[kFadeRadius];
    if (props.has(kMipLevelFactor))     mMipLevelFactor     = props[kMipLevelFactor];
    if (props.has(kRadiusInWorldSpace)) mRadiusInWorldSpace = props[kRadiusInWorldSpace];
    if (props.has(kLockedFrame))        mLockedFrame        = props[kLockedFrame];
}

Properties ASSAOPass::getProperties() const
{
    Properties props;
    props[kIntensity] = mIntensity;
    props[kPower] = mPower;
    props[kRadius] = mRadius;
    props[kBias] = mBias;
    props[kFadeDistance] = mFadeDistance;
    props[kFadeRadius] = mFadeRadius;
    props[kMipLevelFactor] = mMipLevelFactor;
    props[kRadiusInWorldSpace] = mRadiusInWorldSpace;
    props[kLockedFrame] = mLockedFrame;
    return props;
}
