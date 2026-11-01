/// Copyright (C) 2026-2027 ARP (Auto Render Pipeline). All Rights Reserved.

#include "SSAOOptimizer.h"

#include "Core/API/Device.h"
#include "Core/Plugin.h"
#include "Core/AssetResolver.h"
#include "Core/Pass/FullScreenPass.h"
#include "Scene/SceneBuilder.h"
#include "Scene/Lights/Light.h"
#include "GBufferPass/GBufferPass.h"
#include "HierarchicalZPass/HierarchicalZPass.h"
#include "Utils/Scripting/Scripting.h"
#include "Utils/Scripting/ScriptBindings.h"
#include "Utils/Timing/Clock.h"

#include <imgui.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <sstream>

FALCOR_EXPORT_D3D12_AGILITY_SDK

namespace
{

struct PassRegistrar
{
    template<typename T>
    static void registerOne()
    {
        if (!PluginManager::instance().hasClass<RenderPass>(T::kPluginType))
        {
            PluginRegistry registry(PluginManager::instance(), nullptr);
            registry.registerClass<RenderPass, T>();
        }
    }
};

/// A loss of zero is reachable in exact arithmetic and would put log10 at -inf
/// on the curve's y-axis, so the log view floors it well below any real loss.
constexpr float kLossFloor = 1e-30f;

/// Plot height in pixels, matching the resolution the curves are read at.
constexpr float kLossPlotHeight = 90.f;

} // namespace

const Gui::DropdownList SSAOOptimizer::kTargetModeList = {
    { uint32_t(TargetMode::SSAO),      "SSAO (differentiable HZB)" },
    { uint32_t(TargetMode::SSAOPlain), "SSAO (as-shipped, point HZB)" },
    { uint32_t(TargetMode::ASSAO),     "ASSAO (its own forward)" },
};

const Gui::DropdownList SSAOOptimizer::kBackgroundViewList = {
    { uint32_t(BackgroundView::AO),        "AO" },
    { uint32_t(BackgroundView::Reference), "Reference AO" },
    { uint32_t(BackgroundView::Error),     "Error map" },
};

const std::array<const char*, SSAOOptimizer::kParamCount> SSAOOptimizer::kParamNames = {
    "intensity", "power", "radius", "bias"
};

SSAOOptimizer::SSAOOptimizer(const SampleAppConfig& config) : SampleApp(config)
{
    registerOptimizerBindings();
}

SSAOOptimizer::~SSAOOptimizer() {}

void SSAOOptimizer::setCommandLine(int argc, char** argv)
{
    mArgc = argc;
    mArgv = argv;
}

void SSAOOptimizer::registerPasses()
{
    PassRegistrar::registerOne<GBufferPass>();
    PassRegistrar::registerOne<HierarchicalZPass>();
    PassRegistrar::registerOne<SSAOPass>();
    PassRegistrar::registerOne<ASSAOPass>();
}

void SSAOOptimizer::onLoad(RenderContext* pRenderContext)
{
    parseCommandLine();
    registerPasses();

    AssetResolver::getDefaultResolver().addSearchPath(getProjectDirectory() / "assets");

    loadScene(pRenderContext);
    buildRenderGraph();

    applySetupParamsToShaderUnits();
    if (mpASSAOPass)
        setASSAOParams(mInitParams);

    mpScene->update(pRenderContext, getGlobalClock().getTime());
    runGraph(pRenderContext);

    // The shared buffers take the AO image's size, which the graph run above
    // is what first establishes.
    createInteropResources();

    // Headless runs the script itself in onFrameRender(). In the GUI the user
    // picks the script, so a path given on the command line is only preloaded to
    // show its OPTIONS.
    if (!mHeadless && !mScriptPath.empty())
        loadScript(mScriptPath);
}

void SSAOOptimizer::parseCommandLine()
{
    for (int i = 1; i < mArgc; ++i)
    {
        const std::string arg = mArgv[i];
        if (arg == "--script" && i + 1 < mArgc)
        {
            mScriptPath = resolveScriptPath(mArgv[++i]);
        }
        else if (arg == "--headless")
        {
            mHeadless = true;
        }
        else if (arg == "--target" && i + 1 < mArgc)
        {
            const std::string next = mArgv[++i];
            if (next == "ssao")            mTargetMode = TargetMode::SSAO;
            else if (next == "ssao_plain") mTargetMode = TargetMode::SSAOPlain;
            else if (next == "assao")      mTargetMode = TargetMode::ASSAO;
            else logWarning("--target expects 'ssao', 'ssao_plain' or 'assao', got '{}'", next);
        }
        // --target-params and --init-params take the four learnable parameters in
        // UI units, comma-separated: intensity,power,radius,bias. They exist
        // because the setup sliders are GUI-only, and a headless run has to be
        // able to state the experiment it is repeating — without them a headless
        // run always starts from the compiled-in defaults, where the guess and the
        // target coincide and there is nothing to recover.
        else if (arg == "--target-params" && i + 1 < mArgc)
        {
            if (parseParamsArg(mArgv[++i], mTargetUI))
                mTargetUIFromCommandLine = true;
            else
                logWarning("--target-params expects four comma-separated numbers "
                           "(intensity,power,radius,bias), got '{}'", mArgv[i]);
        }
        else if (arg == "--init-params" && i + 1 < mArgc)
        {
            if (parseParamsArg(mArgv[++i], mInitUI))
                mInitUIFromCommandLine = true;
            else
                logWarning("--init-params expects four comma-separated numbers "
                           "(intensity,power,radius,bias), got '{}'", mArgv[i]);
        }
        else
        {
            mScriptArgs.push_back(arg);
        }
    }
}

bool SSAOOptimizer::parseParamsArg(const std::string& text, UIParams& out)
{
    std::vector<float> values;
    std::stringstream ss(text);
    std::string field;
    while (std::getline(ss, field, ','))
    {
        try
        {
            values.push_back(std::stof(field));
        }
        catch (const std::exception&)
        {
            return false;
        }
    }

    if (values.size() != 4)
        return false;

    out.intensity = values[0];
    out.power     = values[1];
    out.radius    = values[2];
    out.bias      = values[3];
    return true;
}

std::filesystem::path SSAOOptimizer::resolveScriptPath(const std::filesystem::path& given)
{
    std::vector<std::filesystem::path> roots;
    roots.push_back(std::filesystem::current_path());
    roots.push_back(getProjectDirectory());

    auto dir = std::filesystem::absolute(getExecutablePath()).parent_path();
    for (int levels = 0; levels < 6 && !dir.empty(); ++levels)
    {
        roots.push_back(dir);
        const auto parent = dir.parent_path();
        if (parent == dir) break;
        dir = parent;
    }

    for (const auto& root : roots)
    {
        std::error_code ec;
        const auto candidate = (root / given).lexically_normal();
        if (std::filesystem::exists(candidate, ec))
        {
            if (candidate != given)
                logInfo("Resolved script '{}' to '{}'", given.string(), candidate.string());
            return candidate;
        }
    }

    logWarning("Script '{}' was not found relative to any of {} candidate roots.", given.string(), roots.size());
    return given;
}

void SSAOOptimizer::loadScene(RenderContext* pRenderContext)
{
    Settings sceneSettings;
    sceneSettings.addOption("PBRTImporter:usePBRTMaterials", false);

    const SceneBuilder::Flags flags = SceneBuilder::Flags::DontMergeMaterials | SceneBuilder::Flags::UseCache;
    SceneBuilder builder(getDevice(), kKrokenScenePath, sceneSettings, flags);

    auto pSun = DirectionalLight::create("Sun");
    pSun->setWorldDirection(normalize(float3(0.866f, -0.5f, 0.f)));
    pSun->setIntensity(float3(7.5f));
    builder.addLight(pSun);

    mpScene = builder.getScene();
    if (!mpScene)
        FALCOR_THROW("Failed to load Kroken scene: {}", kKrokenScenePath);

    const float radius = mpScene->getSceneBounds().radius();
    auto pCamera = mpScene->getCamera();
    if (pCamera)
    {
        pCamera->setDepthRange(std::max(0.1f, radius / 750.0f), radius * 10.0f);
        pCamera->setAspectRatio(float(getTargetFbo()->getWidth()) / float(getTargetFbo()->getHeight()));
    }

    mpScene->setEnvMap(EnvMap::createFromFile(
        getDevice(), AssetResolver::getDefaultResolver().resolvePath(kEnvMapPath)));

    // Target and init guess in the units SSAOPass' own panel edits, which is
    // what the setup sliders show. Folded into shader units on Start.
    //
    // The command line may already have set these — --target-params and
    // --init-params run before the scene loads — so this only fills in the ones
    // that were left at their default. Resetting unconditionally would silently
    // discard a headless run's experiment.
    if (!mTargetUIFromCommandLine)
    {
        UIParams ui;
        mTargetUI = ui;
    }
    if (!mInitUIFromCommandLine)
    {
        UIParams ui;
        mInitUI = ui;
    }
}

void SSAOOptimizer::buildRenderGraph()
{
    mpGraph = RenderGraph::create(getDevice(), "SSAOOptimizerGraph");

    mpGraph->createPass("GBufferPass", "GBufferPass", Properties{});
    mpGraph->createPass("HZPass",      "HierarchicalZPass", Properties{});

    {
        Properties props;
        props["aoMapFormat"]         = ResourceFormat::R32Float;
        props["temporal"]            = false;
        props["differentiable"]      = (mTargetMode != TargetMode::SSAOPlain);
        props["radiusInWorldSpace"]  = false;
        mpGraph->createPass("RefAOPass", "SSAOPass", props);
    }

    {
        Properties props;
        props["aoMapFormat"]        = ResourceFormat::R32Float;
        props["lockedFrame"]        = true;
        props["radiusInWorldSpace"] = false;
        mpGraph->createPass("OptAOPass", "ASSAOPass", props);
    }

    mpGraph->addEdge("GBufferPass.depth",    "HZPass.sceneDepth");
    mpGraph->addEdge("GBufferPass.gBufferA", "RefAOPass.gBufferA");
    mpGraph->addEdge("GBufferPass.depth",    "RefAOPass.sceneDepth");
    mpGraph->addEdge("HZPass.zFurthest",     "RefAOPass.zFurthest");
    mpGraph->addEdge("GBufferPass.gBufferA", "OptAOPass.gBufferA");
    mpGraph->addEdge("GBufferPass.depth",    "OptAOPass.sceneDepth");
    mpGraph->addEdge("HZPass.zFurthest",     "OptAOPass.zFurthest");

    mpGraph->markOutput("RefAOPass.aoMap");
    mpGraph->markOutput("OptAOPass.aoMap");
    mpGraph->setScene(mpScene);

    mpRefAOPass = dynamic_ref_cast<SSAOPass>(mpGraph->getPass("RefAOPass"));
    mpASSAOPass = dynamic_ref_cast<ASSAOPass>(mpGraph->getPass("OptAOPass"));
    FALCOR_CHECK(mpRefAOPass && mpASSAOPass, "Failed to resolve the AO passes from the graph.");

    mpGraph->onResize(getTargetFbo().get());
}

void SSAOOptimizer::compileViewPass()
{
    if (mpViewPass)
        return;

    mpViewPass = FullScreenPass::create(getDevice(), kViewShader);
}

void SSAOOptimizer::createInteropResources()
{
    const uint2 dim = mpASSAOPass ? getFrameDim() : uint2(0, 0);
    if (dim.x == 0 || dim.y == 0)
        return;

    const size_t pixelCount = size_t(dim.x) * size_t(dim.y);

    // The GPU reads and writes these as structured buffers, and CUDA can map
    // them, which is what makes Buffer.to_torch() a zero-copy view.
    const auto flags =
        ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Shared;

    auto recreate = [&](ref<Buffer>& pBuffer, const char* name)
    {
        if (!pBuffer || pBuffer->getElementCount() != pixelCount)
        {
            pBuffer = getDevice()->createStructuredBuffer(
                sizeof(float), uint32_t(pixelCount), flags, MemoryType::DeviceLocal, nullptr);
            pBuffer->setName(name);
        }
    };

    recreate(mpAOBuffer,        "SSAOOptimizer::mpAOBuffer");
    recreate(mpReferenceBuffer, "SSAOOptimizer::mpReferenceBuffer");
    recreate(mpDLdIBuffer,      "SSAOOptimizer::mpDLdIBuffer");

    if (!mpDLdITexture || mpDLdITexture->getWidth(0) != dim.x || mpDLdITexture->getHeight(0) != dim.y)
    {
        mpDLdITexture = getDevice()->createTexture2D(
            dim.x, dim.y, ResourceFormat::R32Float, 1, 1, nullptr,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
        mpDLdITexture->setName("SSAOOptimizer::mpDLdITexture");
    }
}

void SSAOOptimizer::compileCopyPasses()
{
    if (mpTextureToBufferPass)
        return;

    mpTextureToBufferPass = ComputePass::create(getDevice(), kCopyShader, "textureToBuffer");
    mpBufferToTexturePass = ComputePass::create(getDevice(), kCopyShader, "bufferToTexture");
}

void SSAOOptimizer::copyTextureToBuffer(RenderContext* pRenderContext, const ref<Texture>& pSrc, const ref<Buffer>& pDst)
{
    if (!pSrc || !pDst)
        return;

    const uint2 dim(pSrc->getWidth(0), pSrc->getHeight(0));
    if (pDst->getElementCount() != size_t(dim.x) * size_t(dim.y))
    {
        logError("copyTextureToBuffer: a {}x{} texture does not fit a buffer of {} floats.",
            dim.x, dim.y, pDst->getElementCount());
        return;
    }

    compileCopyPasses();
    auto var = mpTextureToBufferPass->getRootVar();
    var["gSrcTexture"] = pSrc;
    var["gDstBuffer"] = pDst;
    var["CB"]["frameDim"] = dim;
    mpTextureToBufferPass->execute(pRenderContext, uint3(dim, 1));
}

void SSAOOptimizer::copyBufferToTexture(RenderContext* pRenderContext, const ref<Buffer>& pSrc, const ref<Texture>& pDst)
{
    if (!pSrc || !pDst)
        return;

    const uint2 dim(pDst->getWidth(0), pDst->getHeight(0));
    if (pSrc->getElementCount() != size_t(dim.x) * size_t(dim.y))
    {
        logError("copyBufferToTexture: a buffer of {} floats does not fill a {}x{} texture.",
            pSrc->getElementCount(), dim.x, dim.y);
        return;
    }

    compileCopyPasses();
    auto var = mpBufferToTexturePass->getRootVar();
    var["gSrcBuffer"] = pSrc;
    var["gDstTexture"] = pDst;
    var["CB"]["frameDim"] = dim;
    mpBufferToTexturePass->execute(pRenderContext, uint3(dim, 1));
}

void SSAOOptimizer::runGraph(RenderContext* pRenderContext)
{
    if (mpGraph)
        mpGraph->execute(pRenderContext);
}

ref<Texture> SSAOOptimizer::getAOMap() const
{
    if (!mpGraph)
        return nullptr;

    auto pOutput = mpGraph->getOutput("OptAOPass.aoMap");
    return pOutput ? pOutput->asTexture() : nullptr;
}

uint2 SSAOOptimizer::getFrameDim() const
{
    const auto pAOMap = getAOMap();
    return pAOMap ? uint2(pAOMap->getWidth(0), pAOMap->getHeight(0)) : uint2(0, 0);
}

std::array<float, SSAOOptimizer::kRenderParamCount> SSAOOptimizer::ShaderParams::toArray() const
{
    return { intensity, power, radius, bias, fadeMul, fadeAdd, mipLevelFactor };
}

SSAOOptimizer::ShaderParams SSAOOptimizer::ShaderParams::fromArray(const std::array<float, kRenderParamCount>& values)
{
    ShaderParams p;
    p.intensity = values[uint32_t(Param::Intensity)];
    p.power = values[uint32_t(Param::Power)];
    p.radius = values[uint32_t(Param::Radius)];
    p.bias = values[uint32_t(Param::Bias)];
    p.fadeMul = values[uint32_t(Param::FadeMul)];
    p.fadeAdd = values[uint32_t(Param::FadeAdd)];
    p.mipLevelFactor = values[uint32_t(Param::MipLevelFactor)];
    return p;
}

std::array<float, SSAOOptimizer::kParamCount> SSAOOptimizer::ShaderParams::toLearnableArray() const
{
    const auto all = toArray();
    std::array<float, kParamCount> out{};
    for (uint32_t i = 0; i < kParamCount; ++i)
        out[i] = all[kLearnableToRender[i]];
    return out;
}

void SSAOOptimizer::ShaderParams::setLearnable(uint32_t i, float v)
{
    switch (LearnableParam(i))
    {
        case LearnableParam::Intensity: intensity = v; break;
        case LearnableParam::Power:     power = v; break;
        case LearnableParam::Radius:    radius = v; break;
        case LearnableParam::Bias:      bias = v; break;
        default:                        break;
    }
}

SSAOOptimizer::ShaderParams SSAOOptimizer::uiToShaderParams(const UIParams& ui, bool radiusInWorldSpace)
{
    ShaderParams s;
    s.intensity = ui.intensity;
    s.power = ui.power;
    s.radius = ui.radius / (radiusInWorldSpace ? 4.0f : 1600.0f);
    s.bias = ui.bias / 1000.0f;
    s.fadeMul = 1.0f / ui.fadeRadius;
    s.fadeAdd = -ui.fadeDistance / ui.fadeRadius;
    s.mipLevelFactor = ui.mipLevelFactor;
    return s;
}

SSAOOptimizer::UIParams SSAOOptimizer::shaderToUIParams(const ShaderParams& shader, bool radiusInWorldSpace)
{
    UIParams ui;
    ui.intensity = shader.intensity;
    ui.power = shader.power;
    ui.radius = shader.radius * (radiusInWorldSpace ? 4.0f : 1600.0f);
    ui.bias = shader.bias * 1000.0f;
    // fadeDistance and fadeRadius are the pre-image of fadeMul/fadeAdd, so both
    // diverge as fadeMul -> 0 (the fade pushed past the far plane).
    if (std::abs(shader.fadeMul) > 1e-12f)
    {
        ui.fadeRadius = 1.0f / shader.fadeMul;
        ui.fadeDistance = -shader.fadeAdd / shader.fadeMul;
    }
    else
    {
        ui.fadeRadius = std::numeric_limits<float>::infinity();
        ui.fadeDistance = std::numeric_limits<float>::infinity();
    }
    ui.mipLevelFactor = shader.mipLevelFactor;
    return ui;
}

Properties SSAOOptimizer::makeUIProperties(const UIParams& ui)
{
    Properties props;
    props["intensity"] = ui.intensity;
    props["power"] = ui.power;
    props["radius"] = ui.radius;
    props["bias"] = ui.bias;
    props["fadeDistance"] = ui.fadeDistance;
    props["fadeRadius"] = ui.fadeRadius;
    props["mipLevelFactor"] = ui.mipLevelFactor;
    return props;
}

SSAOOptimizer::UIParams SSAOOptimizer::readUIProperties(const Properties& props)
{
    UIParams ui;
    ui.intensity = props.get<float>("intensity", ui.intensity);
    ui.power = props.get<float>("power", ui.power);
    ui.radius = props.get<float>("radius", ui.radius);
    ui.bias = props.get<float>("bias", ui.bias);
    ui.fadeDistance = props.get<float>("fadeDistance", ui.fadeDistance);
    ui.fadeRadius = props.get<float>("fadeRadius", ui.fadeRadius);
    ui.mipLevelFactor = props.get<float>("mipLevelFactor", ui.mipLevelFactor);
    return ui;
}

SSAOOptimizer::ShaderParams SSAOOptimizer::getASSAOParams() const
{
    if (!mpASSAOPass)
        return mInitParams;

    const Properties props = mpASSAOPass->getProperties();
    return uiToShaderParams(readUIProperties(props), props.get<bool>("radiusInWorldSpace", false));
}

void SSAOOptimizer::setASSAOParams(const ShaderParams& params)
{
    if (!mpASSAOPass)
        return;

    const Properties props = mpASSAOPass->getProperties();
    const bool radiusInWorldSpace = props.get<bool>("radiusInWorldSpace", false);
    mpASSAOPass->setProperties(makeUIProperties(shaderToUIParams(params, radiusInWorldSpace)));
}

void SSAOOptimizer::setSSAOParams(SSAOPass& pass, const ShaderParams& params)
{
    const Properties props = pass.getProperties();
    const bool radiusInWorldSpace = props.get<bool>("radiusInWorldSpace", false);
    pass.setProperties(makeUIProperties(shaderToUIParams(params, radiusInWorldSpace)));
}

void SSAOOptimizer::captureTarget(RenderContext* pRenderContext)
{
    if (!mpASSAOPass || !mpRefAOPass)
        return;

    createInteropResources();

    if (mTargetMode == TargetMode::ASSAO)
    {
        // Freeze ASSAOPass' own primal output: recovering it recovers the
        // parameters exactly, which is what a self-consistency test wants.
        setASSAOParams(mTargetParams);
        mpASSAOPass->forward(pRenderContext);
        copyTextureToBuffer(pRenderContext, getAOMap(), mpReferenceBuffer);
        return;
    }

    setSSAOParams(*mpRefAOPass, mTargetParams);

    runGraph(pRenderContext);

    auto pRefOutput = mpGraph->getOutput("RefAOPass.aoMap");
    ref<Texture> pRefTexture = pRefOutput ? pRefOutput->asTexture() : nullptr;
    if (!pRefTexture)
    {
        logError("RefAOPass has no usable aoMap output; the target could not be captured.");
        return;
    }

    // An image rendered by SSAOPass, so the loss also carries that pass'
    // quantization and point-sampling differences.
    copyTextureToBuffer(pRenderContext, pRefTexture, mpReferenceBuffer);
}

void SSAOOptimizer::forward(RenderContext* pRenderContext)
{
    if (!mpASSAOPass)
        return;

    createInteropResources();
    mpASSAOPass->forward(pRenderContext);
    copyTextureToBuffer(pRenderContext, getAOMap(), mpAOBuffer);

#if FALCOR_HAS_CUDA
    // Submits the copy and makes the CUDA stream wait for it on the device. The
    // host does not block; the script's reads of mpAOBuffer queue behind it.
    pRenderContext->waitForFalcor();
#endif
}

std::vector<float> SSAOOptimizer::backward(RenderContext* pRenderContext)
{
    if (!mpASSAOPass)
        return {};

    createInteropResources();

#if FALCOR_HAS_CUDA
    // The script wrote mpDLdIBuffer on the CUDA stream; the copy below must not
    // read it before that write lands. A device-side wait, like forward()'s.
    pRenderContext->waitForCuda();
#endif

    copyBufferToTexture(pRenderContext, mpDLdIBuffer, mpDLdITexture);
    mpASSAOPass->backward(pRenderContext, mpDLdITexture);

    // The one host wait of an iteration: copy, flush, read four floats.
    std::vector<float> grad = readBackBuffer(pRenderContext, mpASSAOPass->getParamGrads(), kParamCount);
    FALCOR_CHECK(
        grad.size() == kParamCount,
        "backward() read back {} gradients, expected {}.", grad.size(), kParamCount);
    return grad;
}

std::vector<float> SSAOOptimizer::getParams() const
{
    const auto arr = mpASSAOPass ? getASSAOParams().toLearnableArray()
                                 : mInitParams.toLearnableArray();
    return std::vector<float>(arr.begin(), arr.end());
}

void SSAOOptimizer::setParams(const std::vector<float>& values)
{
    if (values.size() != kParamCount)
    {
        logWarning("setParams expects {} values, got {}", kParamCount, values.size());
        return;
    }
    if (mpASSAOPass)
    {
        auto p = getASSAOParams();
        for (uint32_t i = 0; i < kParamCount; ++i)
            p.setLearnable(i, values[i]);
        setASSAOParams(p);
    }
}

std::vector<float> SSAOOptimizer::getTargetParams() const
{
    const auto arr = mTargetParams.toLearnableArray();
    return std::vector<float>(arr.begin(), arr.end());
}

void SSAOOptimizer::setTargetParams(const std::vector<float>& values)
{
    if (values.size() != kParamCount) return;
    for (uint32_t i = 0; i < kParamCount; ++i)
        mTargetParams.setLearnable(i, values[i]);
}

std::vector<float> SSAOOptimizer::getInitParams() const
{
    const auto arr = mInitParams.toLearnableArray();
    return std::vector<float>(arr.begin(), arr.end());
}

void SSAOOptimizer::setInitParams(const std::vector<float>& values)
{
    if (values.size() != kParamCount) return;
    for (uint32_t i = 0; i < kParamCount; ++i)
        mInitParams.setLearnable(i, values[i]);
}

std::vector<float> SSAOOptimizer::getParamsUI() const
{
    const UIParams ui = mpASSAOPass ? readUIProperties(mpASSAOPass->getProperties())
                                    : shaderToUIParams(mInitParams, false);
    return { ui.intensity, ui.power, ui.radius, ui.bias, ui.fadeDistance, ui.fadeRadius, ui.mipLevelFactor };
}

void SSAOOptimizer::setParamsUI(const std::vector<float>& values)
{
    if (values.size() != kRenderParamCount) return;
    UIParams ui;
    ui.intensity          = values[0];
    ui.power              = values[1];
    ui.radius             = values[2];
    ui.bias               = values[3];
    ui.fadeDistance       = values[4];
    ui.fadeRadius         = values[5];
    ui.mipLevelFactor     = values[6];
    if (mpASSAOPass)
        mpASSAOPass->setProperties(makeUIProperties(ui));
}

void SSAOOptimizer::setTargetMode(uint32_t mode)
{
    if (mode > uint32_t(TargetMode::ASSAO)) return;
    mTargetMode = TargetMode(mode);
    if (mpRefAOPass)
    {
        auto properties = mpRefAOPass->getProperties();
        properties["differentiable"] = mTargetMode != TargetMode::SSAOPlain;
        mpRefAOPass->setProperties(properties);
    }
    logInfo("target renderer set to '{}'", kTargetModeList[size_t(mTargetMode)].label);
}

SSAOOptimizer::GradCheckResult SSAOOptimizer::gradientCheckAt(const IterRecord& rec, float relativeStep)
{
    GradCheckResult out;
    if (!mpASSAOPass || rec.params.size() != kParamCount)
        return out;

    // The loss is defined in the script, so the central differences are taken
    // there too. The script leaves the pass at rec.params when it returns.
    try
    {
        namespace py = pybind11;
        Scripting::Context& ctx = Scripting::getDefaultContext();
        if (!ctx.containsObject("gradient_check"))
        {
            logError("The loaded script does not export gradient_check().");
            return out;
        }

        py::object check = ctx.getObject<py::object>("gradient_check");
        py::dict report = check(rec.params, relativeStep).cast<py::dict>();
        for (py::handle h : report["rows"].cast<py::list>())
        {
            py::dict r = h.cast<py::dict>();
            GradCheckRow row;
            row.name     = r["name"].cast<std::string>();
            row.analytic = r["analytic"].cast<double>();
            row.numeric  = r["numeric"].cast<double>();
            row.relErr   = r["rel_err"].cast<double>();
            row.frozen   = r["frozen"].cast<bool>();
            out.rows.push_back(std::move(row));
        }
        out.worst = report["worst"].cast<double>();
    }
    catch (const std::exception& e)
    {
        logError("Gradient check failed: {}", e.what());
        return GradCheckResult{};
    }

    out.it = rec.it;
    out.valid = true;
    return out;
}

std::vector<std::string> SSAOOptimizer::getParamNames()
{
    const auto& names = kParamNames;
    return std::vector<std::string>(names.begin(), names.end());
}

std::vector<float> SSAOOptimizer::readBackBuffer(RenderContext* pRenderContext, const ref<Buffer>& pBuffer, size_t floatCount)
{
    // A single-buffer readback is one copy and one flush. Prefer the batched form
    // below whenever more than one buffer is wanted: a flush is a full queue drain
    // and costs far more than the sixteen bytes it moves.
    std::vector<ref<Buffer>> buffers{ pBuffer };
    std::vector<float> out;
    readBackBuffers(pRenderContext, buffers, { floatCount }, out);
    return out;
}

void SSAOOptimizer::readBackBuffers(
    RenderContext* pRenderContext,
    const std::vector<ref<Buffer>>& buffers,
    const std::vector<size_t>& floatCounts,
    std::vector<float>& out)
{
    out.clear();
    if (buffers.size() != floatCounts.size() || buffers.empty())
        return;

    size_t totalBytes = 0;
    for (size_t i = 0; i < buffers.size(); ++i)
    {
        if (!buffers[i])
            return;
        totalBytes += floatCounts[i] * sizeof(float);
    }

    if (!mpStagingBuffer || mpStagingBuffer->getSize() < totalBytes)
    {
        mpStagingBuffer = getDevice()->createBuffer(totalBytes, ResourceBindFlags::None, MemoryType::ReadBack, nullptr);
        mpStagingBuffer->setName("SSAOOptimizer::mpStagingBuffer");
    }

    // All copies are enqueued back to back and then a single submit drains them.
    // The previous version called submit(true) once per buffer, so reading the
    // loss and the gradient — twenty bytes apart in two device buffers — cost two
    // full round trips per evaluation, and the optimizer evaluates several times
    // per iteration.
    size_t offset = 0;
    for (size_t i = 0; i < buffers.size(); ++i)
    {
        const size_t bytes = floatCounts[i] * sizeof(float);
        pRenderContext->copyBufferRegion(mpStagingBuffer.get(), offset, buffers[i].get(), 0, bytes);
        offset += bytes;
    }
    pRenderContext->submit(true);

    out.resize(offset / sizeof(float));
    mpStagingBuffer->getBlob(out.data(), 0, offset);
}

void SSAOOptimizer::applySetupParamsToShaderUnits()
{
    // The init guess inherits the target's fade and mipLevelFactor. Neither is
    // learnable, so a difference between the two is a constant offset in the loss
    // that the optimizer can neither see nor remove — it would just sit under the
    // reported residual looking like a failure to converge.
    mInitUI.fadeDistance = mTargetUI.fadeDistance;
    mInitUI.fadeRadius   = mTargetUI.fadeRadius;
    mInitUI.mipLevelFactor = mTargetUI.mipLevelFactor;

    const bool worldSpace = mpASSAOPass &&
        mpASSAOPass->getProperties().get<bool>("radiusInWorldSpace", false);
    mTargetParams = uiToShaderParams(mTargetUI, worldSpace);
    mInitParams   = uiToShaderParams(mInitUI, worldSpace);
}

int SSAOOptimizer::displayedIndex() const
{
    if (mHistory.empty())
        return -1;

    // While running, the newest record is the one whose error map is on screen.
    if (mPhase == Phase::Running)
        return int(mHistory.size()) - 1;

    // Records are contiguous from iteration 0, so the index usually equals the
    // iteration number; searching keeps that true when a run stopped early.
    for (int i = int(mHistory.size()) - 1; i >= 0; --i)
        if (mHistory[i].it <= mViewIt)
            return i;
    return 0;
}

void SSAOOptimizer::applyDisplayParams()
{
    if (!mpASSAOPass)
        return;

    // Setup previews the target: pressing Start is the only thing that should
    // change what the window shows before a run exists.
    if (mPhase == Phase::Setup)
    {
        setASSAOParams(mTargetParams);
        return;
    }

    // The graph forwards at whatever parameters the pass holds, so this single
    // write is what makes the next runGraph() render the displayed iteration
    // into the AO texture the error map reads. The generator re-derives its own
    // parameters from its internal state at the top of every iteration, so
    // writing here cannot steer the optimization.
    const int idx = displayedIndex();
    if (idx >= 0)
        setParams(mHistory[idx].params);
}

bool SSAOOptimizer::loadScript(const std::filesystem::path& path)
{
    if (!std::filesystem::exists(path))
    {
        logWarning("Script not found: {}", path.string());
        return false;
    }

    try
    {
        namespace py = pybind11;
        Scripting::Context& ctx = Scripting::getDefaultContext();
        ctx.setObject("falcor_argv", mScriptArgs);

        // The script decides whether it is the entry point by testing __name__,
        // and pybind11::exec leaves that global unbound — which a guard that
        // treats a missing __name__ as "__main__" reads as "yes". Naming it
        // something else keeps loading the file to defining optimize() and
        // OPTIONS, instead of also starting a headless run inside the GUI.
        // The headless path does not come through here.
        ctx.setObject("__name__", std::string("ssao_optimizer_gui"));

        Scripting::runScriptFromFile(path, ctx);

        // Read OPTIONS dict from the script's globals.
        if (ctx.containsObject("OPTIONS"))
        {
            py::object opts = ctx.getObject<py::object>("OPTIONS");
            py::dict d = opts.cast<py::dict>();
            if (d.contains("iterations"))          mConfig.iterations        = d["iterations"].cast<int>();
            if (d.contains("lr"))                  mConfig.lr                = d["lr"].cast<float>();
            if (d.contains("skip_gradient_check")) mConfig.skipGradientCheck = d["skip_gradient_check"].cast<bool>();
        }

        if (!ctx.containsObject("optimize"))
        {
            logError("Script '{}' does not export an optimize(cfg) generator.", path.string());
            return false;
        }

        mScriptPath = path;
        const auto str = path.string();
        const size_t n  = std::min(str.size(), sizeof(mScriptPathBuf) - 1);
        str.copy(mScriptPathBuf, n);
        mScriptPathBuf[n] = '\0';

        logInfo("Loaded optimizer script '{}': {} iterations, lr {}",
            path.filename().string(), mConfig.iterations, mConfig.lr);

        return true;
    }
    catch (const std::exception& e)
    {
        logError("Failed to load script '{}': {}", path.string(), e.what());
        return false;
    }
}

pybind11::dict SSAOOptimizer::buildConfigDict() const
{
    namespace py = pybind11;
    py::dict d;
    d["iterations"]          = mConfig.iterations;
    d["lr"]                  = mConfig.lr;
    d["skip_gradient_check"] = mConfig.skipGradientCheck;
    // The live widget, not a separate copy: a script-driven gradient check and
    // the results-phase one then agree on the step.
    d["grad_check_rel_step"] = mGradCheckRelStep;
    return d;
}

void SSAOOptimizer::startOptimization()
{
    mHistory.clear();
    mGradCheck     = GradCheckResult{};
    mCurrentIt     = 0;
    mTotalIts      = mConfig.iterations;
    mViewIt        = 0;
    mGeneratorDone = false;
    mGenerator     = pybind11::object();

    auto* pCtx = getRenderContext();
    if (!pCtx)
        return;

    if (mpASSAOPass)
        setASSAOParams(mInitParams);

    // The pass learns its frame size from the aoMap texture the graph hands it,
    // so a target captured before any graph run would go into a zero-sized
    // buffer and quietly produce a loss of nothing.
    if (mpASSAOPass && getFrameDim().x == 0)
    {
        mpScene->update(pCtx, getGlobalClock().getTime());
        runGraph(pCtx);
    }

    captureTarget(pCtx);
    pCtx->submit(true);

    try
    {
        namespace py = pybind11;
        Scripting::Context& ctx = Scripting::getDefaultContext();
        py::object optFn = ctx.getObject<py::object>("optimize");
        mGenerator = optFn(buildConfigDict());
        mPhase     = Phase::Running;
        logInfo("Optimization started: {} iterations, lr {}, target '{}'",
            mConfig.iterations, mConfig.lr, kTargetModeList[size_t(mTargetMode)].label);
    }
    catch (const std::exception& e)
    {
        logError("Failed to start optimizer: {}", e.what());
        mPhase = Phase::Setup;
    }
}

bool SSAOOptimizer::advanceGenerator()
{
    if (mGeneratorDone || !mGenerator)
        return false;

    try
    {
        namespace py = pybind11;
        py::object next = mGenerator.attr("__next__")();
        py::dict   rec  = next.cast<py::dict>();

        IterRecord ir;
        ir.it        = rec["it"].cast<int>();
        ir.loss      = rec["loss"].cast<float>();
        ir.params    = rec["params"].cast<std::vector<float>>();
        ir.grad      = rec["grad"].cast<std::vector<float>>();
        ir.converged = rec.contains("converged") ? rec["converged"].cast<bool>() : false;
        ir.done      = rec.contains("done")      ? rec["done"].cast<bool>()      : false;

        mCurrentIt = ir.it;
        mLastLoss  = ir.loss;
        if (!ir.grad.empty())
            mLastGradient.assign(ir.grad.begin(), ir.grad.end());

        const bool done = ir.done || ir.converged;
        mHistory.push_back(std::move(ir));

        if (done)
        {
            mGeneratorDone = true;
            mPhase         = Phase::Results;
            mViewIt        = mHistory.back().it;
        }
        return true;
    }
    catch (const pybind11::stop_iteration&)
    {
        mGeneratorDone = true;
        mPhase         = Phase::Results;
        if (!mHistory.empty())
            mViewIt = mHistory.back().it;
        return false;
    }
    catch (const std::exception& e)
    {
        logError("Optimizer generator error: {}", e.what());
        mGeneratorDone = true;
        mPhase         = Phase::Results;
        if (!mHistory.empty())
            mViewIt = mHistory.back().it;
        return false;
    }
}

void SSAOOptimizer::renderBackground(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo)
{
    compileViewPass();
    if (!mpViewPass || !mpASSAOPass)
        return;

    const uint2 dim = getFrameDim();
    if (dim.x == 0 || dim.y == 0)
        return;
    if (!getAOMap() || !mpReferenceBuffer)
        return;

    // Which background belongs to the phase:
    //   Running — the error map, which is the one thing the phase is about, and
    //             there is no control that could change it anyway.
    //   Setup   — the target AO. The pass holds the target parameters here, so
    //             AO mode is a preview of the image about to be fitted. Reference
    //             and error would show the previous run's capture, which is stale
    //             before the first run and misleading after one.
    //   Results — whatever the dropdown says.
    BackgroundView view = mBackgroundView;
    if (mPhase == Phase::Running)
        view = BackgroundView::Error;
    else if (mPhase == Phase::Setup)
        view = BackgroundView::AO;

    // The AO is ASSAOPass' own texture, which every graph run refreshes; the
    // reference has no texture, so it is read from the buffer captureTarget()
    // filled. The draw transitions both to the states the shader binds them in.
    auto var = mpViewPass->getRootVar();
    var["gAOMap"]       = getAOMap();
    var["gAOReference"] = mpReferenceBuffer;
    var["CB"]["frameDim"]    = dim;
    var["CB"]["viewMode"]    = uint32_t(view);
    var["CB"]["invMaxError"] = (mErrorMapMaxVal > 0.f) ? 1.f / mErrorMapMaxVal : 1.f;

    mpViewPass->execute(pRenderContext, pTargetFbo);
}

void SSAOOptimizer::onFrameRender(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo)
{
    ++mFrameCount;

    // Headless legacy path: run the script directly on frame 2+.
    if (mHeadless && !mScriptRun && !mScriptPath.empty() && mFrameCount > 1)
    {
        mScriptRun = true;
        int exitCode = 0;
        try
        {
            Scripting::Context& context = Scripting::getDefaultContext();
            context.setObject("falcor_argv", mScriptArgs);
            Scripting::runScriptFromFile(mScriptPath, context);
        }
        catch (const std::exception& e)
        {
            logError("Script '{}' failed: {}", mScriptPath.string(), e.what());
            exitCode = 1;
        }
        shutdown(exitCode);
        return;
    }

    // One generator step per frame while running. The yield hands back the
    // record its loss and gradient belong to; applyDisplayParams() then puts
    // those parameters into the pass, so the error map drawn below is that
    // iteration's and not the last line-search trial's.
    if (mPhase == Phase::Running)
        advanceGenerator();

    applyDisplayParams();

    mpScene->update(pRenderContext, getGlobalClock().getTime());
    runGraph(pRenderContext);

    renderBackground(pRenderContext, pTargetFbo);
}

void SSAOOptimizer::onGuiRender(Gui* pGui)
{
    Gui::Window w(pGui, "SSAO Optimizer", { 420, 540 }, { 40, 40 });

    switch (mPhase)
    {
        case Phase::Setup:   renderSetupUI(w);   break;
        case Phase::Running: renderRunningUI(w); break;
        case Phase::Results: renderResultsUI(w); break;
    }
}

void SSAOOptimizer::renderSetupUI(Gui::Window& w)
{
    // ---- Script picker ----
    if (auto g = w.group("Script", true))
    {
        g.textbox("##scriptpath", mScriptPathBuf, sizeof(mScriptPathBuf));
        ImGui::SameLine();
        if (g.button("Browse"))
        {
            std::filesystem::path chosen;
            if (openFileDialog(Scripting::kFileExtensionFilters, chosen))
                loadScript(resolveScriptPath(chosen));
        }
    }

    // ---- Target parameters ----
    if (auto g = w.group("Target Parameters", true))
    {
        uint32_t mode = uint32_t(mTargetMode);
        if (g.dropdown("Renderer", kTargetModeList, mode))
            setTargetMode(mode);

        g.var("Intensity",          mTargetUI.intensity,          0.f,   5.f,      0.01f);
        g.var("Power",              mTargetUI.power,              0.1f,  5.f,      0.01f);
        g.var("Radius",             mTargetUI.radius,             1.f,   500.f,    0.5f);
        g.var("Bias",               mTargetUI.bias,               0.f,   10.f,     0.01f);
        g.var("FadeDistance",       mTargetUI.fadeDistance,       1.f,   100000.f, 10.f);
        g.var("FadeRadius",         mTargetUI.fadeRadius,         1.f,   100000.f, 10.f);
        g.var("MipLevelFactor",     mTargetUI.mipLevelFactor,     0.f,   100.f,    0.01f);
    }

    // ---- Initial guess ----
    // Only the learnable parameters. mipLevelFactor is not here and not in
    // applySetupParamsToShaderUnits()' fold: it takes the target's value, for the
    // same reason the fade does — a mismatch the optimizer cannot see is a
    // constant offset in the loss that no step will remove.
    if (auto g = w.group("Initial Guess", true))
    {
        g.var("Intensity",          mInitUI.intensity,          0.f,  5.f,   0.01f);
        g.var("Power",              mInitUI.power,              0.1f, 5.f,   0.01f);
        g.var("Radius",             mInitUI.radius,             1.f,  500.f, 0.5f);
        g.var("Bias",               mInitUI.bias,               0.f,  10.f,  0.01f);
    }

    // ---- Optimizer hyper-parameters ----
    if (auto g = w.group("Optimizer", true))
    {
        g.var("Iterations",     mConfig.iterations, 1,     2000,  1);
        g.var("Learning rate",  mConfig.lr,         1e-4f, 1.f,   1e-4f);
    }

    w.separator();
    const bool canStart = !mScriptPath.empty();
    if (!canStart)
        ImGui::BeginDisabled();
    if (w.button("Start Optimization"))
        startOptimization();
    if (!canStart)
        ImGui::EndDisabled();

    applySetupParamsToShaderUnits();
}

void SSAOOptimizer::renderRunningUI(Gui::Window& w)
{
    w.text("Optimizing...");

    const float frac = (mTotalIts > 0) ? float(mCurrentIt) / float(mTotalIts) : 0.f;
    ImGui::ProgressBar(frac, ImVec2(-1.f, 0.f),
        fmt::format("{}/{}", mCurrentIt, mTotalIts).c_str());

    w.text(fmt::format("Loss: {:.6e}", mLastLoss));
    w.text(fmt::format("Target: {}", kTargetModeList[size_t(mTargetMode)].label));
}

void SSAOOptimizer::renderResultsUI(Gui::Window& w)
{
    if (w.button("Restart"))
        mPhase = Phase::Setup;

    if (mHistory.empty())
    {
        return;
    }

    const int firstIt = mHistory.front().it;
    const int lastIt  = mHistory.back().it;

    // ---- Iteration scrub slider, defaulting to the final iteration ----
    if (w.slider("Iteration", mViewIt, firstIt, lastIt))
    {
        // The check belongs to the iteration it measured, and that is no longer
        // the one on screen.
        mGradCheck = GradCheckResult{};
    }
    if (mHistory.back().converged)
        w.text("(the line search gave up at the last iteration)");

    // ---- Loss curve ----
    if (auto g = w.group("Loss Curve", true))
        renderLossCurve(g);

    // ---- Parameter table ----
    if (auto g = w.group("Parameters", true))
        renderParamTable(g);

    // ---- Background view selector ----
    if (auto g = w.group("Background", true))
    {
        g.dropdown("View", kBackgroundViewList, reinterpret_cast<uint32_t&>(mBackgroundView));
        if (mBackgroundView == BackgroundView::Error)
        {
            g.var("Max error", mErrorMapMaxVal, 0.001f, 1.f, 0.001f);
        }
    }

    // ---- Gradient check ----
    if (auto g = w.group("Gradient Check", true))
        renderGradCheckUI(g);
}

void SSAOOptimizer::renderLossCurve(Gui::Widgets& w)
{
    if (mHistory.empty())
    {
        w.text("(no data)");
        return;
    }

    const int n = int(mHistory.size());

    // The curve, transformed for display. The loss falls by orders of magnitude
    // over a run, which in linear space is a cliff followed by a flat line.
    std::vector<float> values(n);
    float lo = std::numeric_limits<float>::max();
    float hi = std::numeric_limits<float>::lowest();
    for (int i = 0; i < n; ++i)
    {
        const float loss = std::max(mHistory[i].loss, kLossFloor);
        values[i] = mLogScale ? std::log10(loss) : loss;
        lo = std::min(lo, values[i]);
        hi = std::max(hi, values[i]);
    }
    if (hi - lo < 1e-6f)
        hi = lo + 1e-6f;   // A flat curve still needs a span to map into.

    ImGui::PlotLines("##loss", values.data(), n, 0, nullptr, lo, hi,
        ImVec2(0, kLossPlotHeight));

    // Mark the displayed iteration. ImGui draws the curve inside the frame
    // minus the frame padding, so the point is placed in that same inner rect;
    // using the item rect alone would drift by the padding.
    const ImVec2 frameMin = ImGui::GetItemRectMin();
    const ImVec2 frameMax = ImGui::GetItemRectMax();
    const ImVec2 pad      = ImGui::GetStyle().FramePadding;
    const ImVec2 innerMin(frameMin.x + pad.x, frameMin.y + pad.y);
    const ImVec2 innerMax(frameMax.x - pad.x, frameMax.y - pad.y);

    const int idx = displayedIndex();
    if (idx >= 0 && innerMax.x > innerMin.x && innerMax.y > innerMin.y)
    {
        const float tx = (n > 1) ? float(idx) / float(n - 1) : 0.f;
        const float ty = (values[idx] - lo) / (hi - lo);
        const ImVec2 p(
            innerMin.x + tx * (innerMax.x - innerMin.x),
            innerMin.y + (1.f - ty) * (innerMax.y - innerMin.y));

        ImDrawList* pDraw = ImGui::GetWindowDrawList();
        pDraw->AddCircleFilled(p, 4.f, IM_COL32(235, 70, 70, 255));
        pDraw->AddCircle(p, 4.f, IM_COL32(20, 20, 20, 200));
    }

    ImGui::Checkbox("Log scale", &mLogScale);
    if (idx >= 0)
    {
        ImGui::SameLine();
        ImGui::Text("iter %d   loss %.6e", mHistory[idx].it, mHistory[idx].loss);
    }
}

void SSAOOptimizer::renderParamTable(Gui::Widgets& w)
{
    const int idx = displayedIndex();
    if (idx < 0)
    {
        w.text("(no data)");
        return;
    }

    const IterRecord& rec = mHistory[idx];
    const auto truth      = mTargetParams.toLearnableArray();
    const auto& names     = kParamNames;

    // Shader units throughout: the parameters, the truth and the gradient all
    // live in the same space, so the residual is a subtraction and the gradient
    // needs no chain rule to line up with it.
    constexpr ImGuiTableFlags flags =
        ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingFixedFit;

    if (ImGui::BeginTable("##params", 6, flags))
    {
        ImGui::TableSetupColumn("param");
        ImGui::TableSetupColumn("value");
        ImGui::TableSetupColumn("truth");
        ImGui::TableSetupColumn("residual");
        ImGui::TableSetupColumn("rel err");
        ImGui::TableSetupColumn("gradient");
        ImGui::TableHeadersRow();

        for (uint32_t i = 0; i < kParamCount; ++i)
        {
            const float val = (i < rec.params.size()) ? rec.params[i] : 0.f;
            const float tru = truth[i];
            const float res = val - tru;
            const float rel = (std::abs(tru) > 1e-12f) ? std::abs(res) / std::abs(tru) : 0.f;
            const float grd = (i < rec.grad.size()) ? rec.grad[i] : 0.f;

            ImGui::TableNextRow();
            ImGui::TableNextColumn(); ImGui::TextUnformatted(names[i]);
            ImGui::TableNextColumn(); ImGui::Text("%.6e", val);
            ImGui::TableNextColumn(); ImGui::Text("%.6e", tru);
            ImGui::TableNextColumn(); ImGui::Text("%+.3e", res);
            ImGui::TableNextColumn(); ImGui::Text("%.3e", rel);
            ImGui::TableNextColumn(); ImGui::Text("%+.3e", grd);
        }
        ImGui::EndTable();
    }

    w.text(fmt::format("iteration {} of {}", rec.it, mHistory.back().it));
}

void SSAOOptimizer::renderGradCheckUI(Gui::Widgets& w)
{
    w.var("Relative step", mGradCheckRelStep, 1e-5f, 0.1f, 1e-5f);
    w.tooltip("h = rel step * |param|, so one setting covers parameters whose "
              "magnitudes differ by orders of magnitude.");

    const int idx = displayedIndex();
    const bool canRun = (idx >= 0) && mpASSAOPass;
    if (!canRun)
        ImGui::BeginDisabled();
    if (w.button("Check analytic gradient at this iteration"))
    {
        mGradCheck = gradientCheckAt(mHistory[idx], mGradCheckRelStep);
        if (mGradCheck.valid)
            logInfo("Gradient check at iteration {}: worst relative error {:.3e}",
                mGradCheck.it, mGradCheck.worst);
    }
    if (!canRun)
        ImGui::EndDisabled();

    if (!mGradCheck.valid)
    {
        return;
    }

    w.text(fmt::format("iteration {}   worst relative error {:.3e}",
        mGradCheck.it, mGradCheck.worst));

    constexpr ImGuiTableFlags flags =
        ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingFixedFit;
    if (ImGui::BeginTable("##gradcheck", 4, flags))
    {
        ImGui::TableSetupColumn("param");
        ImGui::TableSetupColumn("analytic");
        ImGui::TableSetupColumn("numeric");
        ImGui::TableSetupColumn("rel err");
        ImGui::TableHeadersRow();

        for (const GradCheckRow& row : mGradCheck.rows)
        {
            ImGui::TableNextRow();
            ImGui::TableNextColumn(); ImGui::TextUnformatted(row.name.c_str());
            ImGui::TableNextColumn(); ImGui::Text("%+.4e", row.analytic);
            ImGui::TableNextColumn(); ImGui::Text("%+.4e", row.numeric);
            ImGui::TableNextColumn();
            if (row.frozen)
                ImGui::TextUnformatted("frozen");
            else
                ImGui::Text("%.3e", row.relErr);
        }
        ImGui::EndTable();
    }

    for (const GradCheckRow& row : mGradCheck.rows)
    {
        if (row.frozen)
        {
            w.textWrapped(fmt::format(
                "'{}' is frozen: its analytic gradient was zero at the first "
                "iteration, so the run never stepped it. Left out of the worst error.",
                row.name));
            break;
        }
    }
}

bool SSAOOptimizer::onKeyEvent(const KeyboardEvent& keyEvent)
{
    // Deliberately empty: every state change goes through an explicit control,
    // and while a run is in flight the phase is meant to be non-interactive.
    return false;
}

bool SSAOOptimizer::onMouseEvent(const MouseEvent& mouseEvent)
{
    return false;
}

void SSAOOptimizer::registerOptimizerBindings()
{
    using namespace pybind11::literals;

    auto bindFunc = [this](pybind11::module& m)
    {
        m.def("num_params",        []() { return kParamCount; });
        m.def("param_names",       []() { return SSAOOptimizer::getParamNames(); });

        m.def("set_params",        [this](const std::vector<float>& v) { setParams(v); }, "values"_a);
        m.def("get_params",        [this]()                            { return getParams(); });
        m.def("set_params_ui",     [this](const std::vector<float>& v) { setParamsUI(v); }, "values"_a);
        m.def("get_params_ui",     [this]()                            { return getParamsUI(); });
        m.def("get_target_params", [this]()                            { return getTargetParams(); });
        m.def("set_target_params", [this](const std::vector<float>& v) { setTargetParams(v); }, "values"_a);
        m.def("get_init_params",   [this]()                            { return getInitParams(); });
        m.def("set_init_params",   [this](const std::vector<float>& v) { setInitParams(v); }, "values"_a);

        m.def("get_target_mode",   [this]()            { return getTargetMode(); });
        m.def("set_target_mode",   [this](uint32_t m2) { setTargetMode(m2); }, "mode"_a);
        m.def("target_modes", []()
        {
            std::vector<std::string> names;
            for (const auto& e : kTargetModeList) names.push_back(e.label);
            return names;
        });

        m.def("capture_target", [this]() { captureTarget(getRenderContext()); });

        /// ASSAOPass as a differentiable op. forward() renders AO at the current
        /// parameters into ao_map_buffer(); backward() reads dL/dAO from
        /// dL_dI_buffer() and returns dL/dparam. Both order themselves against
        /// the CUDA stream, so the script needs no sync around them.
        m.def("forward",  [this]() { forward(getRenderContext()); });
        m.def("backward", [this]() { return backward(getRenderContext()); });

        m.def("ao_map_buffer",      [this]() { return mpAOBuffer; });
        m.def("reference_buffer",   [this]() { return mpReferenceBuffer; });
        m.def("dL_dI_buffer",       [this]() { return mpDLdIBuffer; });
        m.def("param_grads_buffer", [this]() { return mpASSAOPass ? mpASSAOPass->getParamGrads() : nullptr; });

        m.def("frame_dim", [this]()
        {
            if (!mpGraph) return std::vector<uint32_t>{ 0u, 0u };
            auto pOut = mpGraph->getOutput("OptAOPass.aoMap");
            ref<Texture> pTex = pOut ? pOut->asTexture() : nullptr;
            if (!pTex) return std::vector<uint32_t>{ 0u, 0u };
            return std::vector<uint32_t>{ pTex->getWidth(0), pTex->getHeight(0) };
        });

        m.def("sync", [this]() { getRenderContext()->submit(true); });
    };

    ScriptBindings::registerBinding(bindFunc);
}

int runMain(int argc, char** argv)
{
    SampleAppConfig config;
    config.windowDesc.title = "SSAO Optimizer";
    config.windowDesc.resizableWindow = true;
    config.pauseTime = true;

    for (int i = 1; i < argc; ++i)
    {
        if (std::string(argv[i]) == "--headless")
        {
            config.headless = true;
            break;
        }
    }

    SSAOOptimizer sample(config);
    sample.setCommandLine(argc, argv);
    return sample.run();
}

int main(int argc, char** argv)
{
    return catchAndReportAllExceptions([&] { return runMain(argc, argv); });
}
