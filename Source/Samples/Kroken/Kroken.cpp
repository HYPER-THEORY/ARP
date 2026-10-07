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
#include "Kroken.h"

#include "Utils/Image/Bitmap.h"
#include "Core/Plugin.h"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>

FALCOR_EXPORT_D3D12_AGILITY_SDK

namespace
{
    // Registers Arctic's render passes with the in-process plugin registry.
    // Arctic is built as a static library (not a plugin DLL), so its passes are
    // not in plugins.json and must be registered manually before the render
    // graph tries to create them by name.
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

    // Buffer name -> DebugViewPass selection index. These indices must match the
    // switch in DebugViewPass.3d.slang.
    //
    // Deliberately absent: index 12 ("indirect") and 14 ("fieldDebug"). Both are
    // produced by IndirectApproxPass, which Kroken does not include, so those two
    // inputs stay unconnected and `renderData.getTexture()` returns null. See
    // parseCaptureArg() and the Debug View UI group for why they are never selected.
    const std::vector<std::pair<std::string, uint32_t>> kCaptureBuffers = {
        {"final", 0},
        {"gBufferA", 1}, {"gBufferB", 2}, {"gBufferC", 3}, {"gBufferD", 4}, {"gBufferE", 5},
        {"depth", 6}, {"emissive", 7}, {"shadowMask", 8},
        {"aoMap", 9}, {"ao", 9},
        {"di", 10}, {"diColor", 10},
        {"ambient", 11}, {"ambientOnly", 11},
        {"composite", 13}, {"ambientColor", 13},
        {"shadingModel", 15}, {"shadeMode", 15},
    };

    const std::vector<std::pair<std::string, uint32_t>> kCaptureChannels = {
        {"RGB", 0}, {"R", 1}, {"G", 2}, {"B", 3}, {"A", 4},
    };

    bool matchName(const std::string& s, const std::vector<std::pair<std::string, uint32_t>>& table, uint32_t& outIdx, std::string& outName)
    {
        for (const auto& [name, idx] : table)
        {
            if (s.length() == name.length() && std::equal(s.begin(), s.end(), name.begin(), [](char a, char b) { return std::tolower(a) == std::tolower(b); }))
            {
                outIdx = idx;
                outName = name;
                return true;
            }
        }
        return false;
    }

    /** Parses a --capture argument of the form "buffer[:channel]" into the indices
        DebugViewPass expects, plus canonical names for the output filename.
        Returns false on an unrecognized name; the caller keeps the default.
    */
    bool parseCaptureArg(const std::string& arg, uint32_t& bufferIdx, std::string& bufferName, uint32_t& channelIdx, std::string& channelName)
    {
        std::string bufPart = arg, chanPart;
        auto colon = arg.find(':');
        if (colon != std::string::npos)
        {
            bufPart = arg.substr(0, colon);
            chanPart = arg.substr(colon + 1);
        }

        if (!matchName(bufPart, kCaptureBuffers, bufferIdx, bufferName)) return false;
        if (chanPart.empty())
        {
            channelIdx = 0;
            channelName = "RGB";
            return true;
        }
        if (!matchName(chanPart, kCaptureChannels, channelIdx, channelName)) return false;
        return true;
    }
} // namespace

Kroken::Kroken(const SampleAppConfig& config) : SampleApp(config) {}
Kroken::~Kroken() = default;

void Kroken::setCommandLine(int argc, char** argv)
{
    mArgc = argc;
    mArgv = argv;
}

void Kroken::parseCommandLine()
{
    // Supported flags:
    //   --capture [buffer[:channel]]  capture one buffer headlessly and exit.
    //                                 Defaults to final:RGB. Note there is no
    //                                 --iaprofile (no IndirectApproxPass) and no
    //                                 --ao (AO is fixed to GTAO).
    //   --captureframe N              frame to capture on (default 256, by which
    //                                 point TAA has converged).
    for (int i = 1; i < mArgc; ++i)
    {
        const std::string arg = mArgv[i];

        if (arg == "--captureframe")
        {
            if (i + 1 < mArgc)
            {
                const uint32_t frame = (uint32_t)std::atoi(mArgv[i + 1]);
                if (frame > 0) { mCaptureFrame = frame; ++i; }
                else logWarning("--captureframe expects a positive frame number.");
            }
        }
        else if (arg == "--capture")
        {
            mAutoCapture = true;
            if (i + 1 < mArgc)
            {
                const std::string next = mArgv[i + 1];
                if (!next.empty() && next.substr(0, 2) != "--")
                {
                    uint32_t bufIdx = 0, chanIdx = 0;
                    std::string bufName, chanName;
                    if (parseCaptureArg(next, bufIdx, bufName, chanIdx, chanName))
                    {
                        mCaptureBufferIdx = bufIdx;
                        mCaptureChannelIdx = chanIdx;
                        mCaptureBufferName = bufName;
                        mCaptureChannelName = chanName;
                    }
                    else
                    {
                        logWarning("Unrecognized --capture argument '{}', defaulting to final:RGB", next);
                    }
                    ++i; // consume the buffer argument
                }
            }
        }
    }
}

void Kroken::onLoad(RenderContext* pRenderContext)
{
    parseCommandLine();

    // Extend asset search paths so the Kroken assets under <project>/assets are
    // visible. The default search path is <project>/media. This must happen
    // before the SceneBuilder is constructed: it snapshots the default resolver.
    AssetResolver::getDefaultResolver().addSearchPath(getProjectDirectory() / "assets");

    // Register Arctic's render passes with the in-process plugin registry.
    // SSAOPass and IndirectApproxPass are deliberately not registered: neither is
    // in the Kroken graph.
    PassRegistrar::registerOne<GBufferPass>();
    PassRegistrar::registerOne<ShadowPass>();
    PassRegistrar::registerOne<ShadowProjectionPass>();
    PassRegistrar::registerOne<SkyboxPass>();
    PassRegistrar::registerOne<DIPass>();
    PassRegistrar::registerOne<GTAOPass>();
    PassRegistrar::registerOne<AmbientPass>();
    PassRegistrar::registerOne<CompositePass>();
    PassRegistrar::registerOne<MotionVectorPass>();
    PassRegistrar::registerOne<TAAPass>();
    PassRegistrar::registerOne<DebugViewPass>();

    // The scene supplies the scale-dependent defaults for the graph, so load it
    // first and build the graph afterwards.
    loadScene(pRenderContext);
    buildRenderGraph();
}

void Kroken::loadScene(RenderContext* pRenderContext)
{
    // Force StandardMaterial (Metal-Rough) for every imported pbrt material.
    // This is already the PBRTImporter default, but stating it makes the intent
    // explicit: the Arctic GBufferPass only understands Metal-Rough, so a pbrt
    // material (PBRTDiffuseMaterial etc.) would fall through to the unsupported
    // material path.
    Settings sceneSettings;
    sceneSettings.addOption("PBRTImporter:usePBRTMaterials", false);

    // DontMergeMaterials keeps the 77 distinct pbrt material names recognizable
    // instead of folding equal ones together, which makes per-material debugging
    // possible. UseCache caches the parsed scene under the runtime directory;
    // the Kroken scene is 221 PLY meshes totalling several GB, so the first
    // import is expensive and every subsequent launch reuses the cache.
    //
    // Caveat: the cache key is derived from the scene path and the build flags
    // only -- it does NOT include sceneSettings. Changing
    // `PBRTImporter:usePBRTMaterials` or the importer itself will keep serving a
    // stale cache until it is deleted by hand.
    const SceneBuilder::Flags flags = SceneBuilder::Flags::DontMergeMaterials | SceneBuilder::Flags::UseCache;
    SceneBuilder builder(getDevice(), kKrokenScenePath, sceneSettings, flags);

    // lights.pbrt has no DirectionalLight, and DIPass / ShadowPass /
    // ShadowProjectionPass all key off the first one in the scene. Add a
    // synthetic sun before the scene is built.
    //
    // Ordering matters: on a cache miss the builder imports the scene on
    // construction and the light added here lands in the scene data and gets
    // written into the cache. On a cache hit the builder returns early with the
    // scene already built from the cache, and this addLight() is discarded -- but
    // the cached scene already contains the sun from the miss that wrote it, so
    // either way the light is present afterwards. syncSunLight() below re-reads
    // it from the scene so the UI never holds a dangling reference.
    mpSunLight = DirectionalLight::create("Sun");
    mpSunLight->setWorldDirection(normalize(mSunDirection));
    mpSunLight->setIntensity(float3(mSunIntensity));
    builder.addLight(mpSunLight);

    mpScene = builder.getScene();
    if (!mpScene)
    {
        FALCOR_THROW("Failed to load Kroken scene: {}", kKrokenScenePath);
    }

    syncSunLight();

    // Derive the scale-dependent tuning from the scene bounds. Arctic's defaults
    // assume a scene tens of units across; Kroken is an interior spanning several
    // hundred units, so world-space quantities (shadow depth bias, PCF filter
    // radius, soft-transition width, AO radius) are rescaled here.
    const float radius = mpScene->getSceneBounds().radius();
    if (radius > 0.f) mSceneRadius = radius;

    // The shadow frustum is fitted to the scene's light-space bounding box, so a
    // conservative world size for one shadowmap texel is (2 * radius) / res. Both
    // ShadowPass and ShadowProjectionPass work in METERS (light-space linear
    // depth), so their biases must scale with the world size, not stay at the
    // values tuned for a 50-unit scene.
    mShadowTexelWorld = (2.f * mSceneRadius) / float(mShadowResolution);
    mShadowConstantDepthBias = 1.0f * mShadowTexelWorld;
    mShadowSlopeDepthBias = 2.0f * mShadowTexelWorld;
    mShadowMaxSlopeDepthBias = 12.0f; // Dimensionless: clamps the NoL slope term.
    mShadowProjectionFilterRadiusWorld = 2.0f * mShadowTexelWorld;

    logInfo(
        "Kroken scene radius {:.1f}, shadow texel {:.3f} world units",
        mSceneRadius,
        mShadowTexelWorld
    );

    mpScene->setCameraSpeed(mSceneRadius * 0.25f);
    auto pCamera = mpScene->getCamera();
    if (pCamera)
    {
        // The camera itself comes from camera-1.pbrt (fov 17, positioned by the
        // importer). Only the depth range and aspect ratio are adjusted here.
        pCamera->setDepthRange(std::max(0.1f, mSceneRadius / 750.0f), mSceneRadius * 10.0f);
        pCamera->setAspectRatio((float)getTargetFbo()->getWidth() / (float)getTargetFbo()->getHeight());
    }

    mpScene->setEnvMap(EnvMap::createFromFile(getDevice(), AssetResolver::getDefaultResolver().resolvePath(kEnvMapPath)));

    if (mpGraph) mpGraph->setScene(mpScene);
}

void Kroken::syncSunLight()
{
    if (!mpScene) return;

    // Re-find the directional light in the scene rather than trusting the member
    // set before getScene(). On a cache hit the scene is rebuilt from cached data,
    // so the object added to the builder is not the one that ends up in the scene.
    for (const auto& pLight : mpScene->getLights())
    {
        if (!pLight || pLight->getType() != LightType::Directional) continue;

        auto pDirLight = dynamic_ref_cast<DirectionalLight>(pLight);
        if (!pDirLight) continue;

        mpSunLight = pDirLight;
        mpSunLight->setWorldDirection(normalize(mSunDirection));
        mpSunLight->setIntensity(float3(mSunIntensity));
        return;
    }

    // Only reachable if a scene cache written before the synthetic sun was added
    // is still on disk.
    logWarning(
        "Kroken: no DirectionalLight found in the scene. DIPass, ShadowPass and "
        "ShadowProjectionPass will do nothing. Delete the stale scene cache and rerun."
    );
}

void Kroken::buildRenderGraph()
{
    mpGraph = RenderGraph::create(getDevice(), "KrokenDeferredGraph");

    mpGraph->createPass("GBufferPass", "GBufferPass", Properties{});

    {
        Properties props;
        props["shadowResolution"] = mShadowResolution;
        props["constantDepthBias"] = mShadowConstantDepthBias;
        props["slopeDepthBias"] = mShadowSlopeDepthBias;
        props["maxSlopeDepthBias"] = mShadowMaxSlopeDepthBias;
        mpGraph->createPass("ShadowPass", "ShadowPass", props);
    }

    mpGraph->createPass("SkyboxPass", "SkyboxPass", Properties{});

    {
        Properties props;
        props["constantDepthBias"] = mShadowConstantDepthBias;
        props["filterRadiusWorld"] = mShadowProjectionFilterRadiusWorld;
        props["softTransitionTexels"] = mShadowProjectionSoftTransitionTexels;
        mpGraph->createPass("ShadowProjectionPass", "ShadowProjectionPass", props);
    }

    {
        Properties props;
        props["effectRadius"] = 100.0f;
        props["thickness"] = 0.1f;
        props["power"] = 2.0f;
        mpGraph->createPass("AOPass", "GTAOPass", props);
    }

    mpGraph->createPass("DIPass", "DIPass", Properties{});

    {
        Properties props;
        props["intensity"] = 0.5f;
        mpGraph->createPass("AmbientPass", "AmbientPass", props);
    }

    mpGraph->createPass("CompositePass", "CompositePass", Properties{});

    mpGraph->createPass("MotionVectorPass", "MotionVectorPass", Properties{});

    mpGraph->createPass("TAAPass", "TAAPass", Properties{});

    mpGraph->createPass("ToneMapper", "ToneMapper", Properties{});

    mpGraph->createPass("DebugViewPass", "DebugViewPass", Properties{});

    // --- Edges ---

    // Background / shadows.
    mpGraph->addEdge("GBufferPass.depth", "SkyboxPass.sceneDepth");
    mpGraph->addEdge("GBufferPass.depth", "ShadowProjectionPass.sceneDepth");
    mpGraph->addEdge("GBufferPass.gBufferA", "ShadowProjectionPass.gBufferA");
    mpGraph->addEdge("GBufferPass.gBufferB", "ShadowProjectionPass.gBufferB");
    mpGraph->addEdge("GBufferPass.gBufferD", "ShadowProjectionPass.gBufferD");
    mpGraph->addEdge("ShadowPass.shadowDepth", "ShadowProjectionPass.shadowDepth");

    // Ambient occlusion. GTAO's temporal filter reprojects its half-res history,
    // so it needs the motion vectors; without this edge it logs a warning and
    // falls back to spatial-only filtering.
    mpGraph->addEdge("GBufferPass.gBufferA", "AOPass.gBufferA");
    mpGraph->addEdge("GBufferPass.depth", "AOPass.sceneDepth");
    mpGraph->addEdge("MotionVectorPass.motionVecs", "AOPass.motionVecs");

    // Direct (sun) lighting.
    mpGraph->addEdge("GBufferPass.gBufferA", "DIPass.gBufferA");
    mpGraph->addEdge("GBufferPass.gBufferB", "DIPass.gBufferB");
    mpGraph->addEdge("GBufferPass.gBufferC", "DIPass.gBufferC");
    mpGraph->addEdge("GBufferPass.gBufferD", "DIPass.gBufferD");
    mpGraph->addEdge("GBufferPass.gBufferE", "DIPass.gBufferE");
    mpGraph->addEdge("GBufferPass.depth", "DIPass.sceneDepth");
    mpGraph->addEdge("ShadowProjectionPass.shadowMask", "DIPass.shadowMask");

    // Ambient IBL.
    mpGraph->addEdge("GBufferPass.gBufferA", "AmbientPass.gBufferA");
    mpGraph->addEdge("GBufferPass.gBufferB", "AmbientPass.gBufferB");
    mpGraph->addEdge("GBufferPass.gBufferC", "AmbientPass.gBufferC");
    mpGraph->addEdge("GBufferPass.gBufferD", "AmbientPass.gBufferD");
    mpGraph->addEdge("GBufferPass.depth", "AmbientPass.sceneDepth");
    mpGraph->addEdge("AOPass.aoMap", "AmbientPass.aoMap");

    // Composite. No `indirect` edge: IndirectApproxPass is not in this graph.
    mpGraph->addEdge("GBufferPass.depth", "CompositePass.sceneDepth");
    mpGraph->addEdge("SkyboxPass.skyboxColor", "CompositePass.skyboxColor");
    mpGraph->addEdge("DIPass.color", "CompositePass.diColor");
    mpGraph->addEdge("AmbientPass.ambient", "CompositePass.ambient");
    mpGraph->addEdge("GBufferPass.emissive", "CompositePass.emissive");

    // Temporal AA: velocity + depth in, resolved HDR out, then tonemap.
    mpGraph->addEdge("GBufferPass.depth", "MotionVectorPass.sceneDepth");
    mpGraph->addEdge("MotionVectorPass.motionVecs", "TAAPass.motionVecs");
    mpGraph->addEdge("GBufferPass.depth", "TAAPass.sceneDepth");
    mpGraph->addEdge("CompositePass.color", "TAAPass.colorIn");
    mpGraph->addEdge("TAAPass.colorOut", "ToneMapper.src");

    // Debug view. The `indirect` (12) and `fieldDebug` (14) inputs are left
    // unconnected; Kroken's own Debug View UI never selects them.
    mpGraph->addEdge("ToneMapper.dst", "DebugViewPass.final");
    mpGraph->addEdge("DIPass.color", "DebugViewPass.diColor");
    mpGraph->addEdge("GBufferPass.gBufferA", "DebugViewPass.gBufferA");
    mpGraph->addEdge("GBufferPass.gBufferB", "DebugViewPass.gBufferB");
    mpGraph->addEdge("GBufferPass.gBufferC", "DebugViewPass.gBufferC");
    mpGraph->addEdge("GBufferPass.gBufferD", "DebugViewPass.gBufferD");
    mpGraph->addEdge("GBufferPass.gBufferE", "DebugViewPass.gBufferE");
    mpGraph->addEdge("GBufferPass.depth", "DebugViewPass.depth");
    mpGraph->addEdge("ShadowProjectionPass.shadowMask", "DebugViewPass.shadowMask");
    mpGraph->addEdge("AOPass.aoMap", "DebugViewPass.aoMap");
    mpGraph->addEdge("AmbientPass.ambient", "DebugViewPass.ambient");
    mpGraph->addEdge("CompositePass.color", "DebugViewPass.composite");
    mpGraph->addEdge("GBufferPass.emissive", "DebugViewPass.emissive");
    mpGraph->markOutput("DebugViewPass.output");

    if (mpScene) mpGraph->setScene(mpScene);
    if (getTargetFbo()) mpGraph->onResize(getTargetFbo().get());
}

void Kroken::onResize(uint32_t width, uint32_t height)
{
    if (mpGraph) mpGraph->onResize(getTargetFbo().get());
    if (mpScene) mpScene->setCameraAspectRatio((float)width / (float)height);
}

void Kroken::onFrameRender(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo)
{
    // Service a pending F11 request before this frame touches the FBO: what is
    // there now is the previous frame after UI compositing.
    if (mScreenshotRequested)
    {
        mScreenshotRequested = false;
        saveScreenshot(pRenderContext, pTargetFbo->getColorTexture(0));
    }

    if (!mpScene || !mpGraph)
    {
        const float4 clearColor(0.f, 0.f, 0.f, 1.f);
        pRenderContext->clearFbo(pTargetFbo.get(), clearColor, 1.0f, 0, FboAttachmentType::All);
        return;
    }

    // Sub-pixel camera jitter for TAA, mirroring GBufferBase::updateFrameDim. The
    // jitter flows into both the jittered projection (so GBufferPass rasterizes
    // sub-pixel-offset geometry) and the MotionVectorPass +jitter term.
    {
        auto pCamera = mpScene->getCamera();
        const auto& fbo = getTargetFbo();
        const float2 scale = 1.f / float2(float(fbo->getWidth()), float(fbo->getHeight()));
        if (mTAAEnabled)
        {
            if (!pCamera->getPatternGenerator())
                pCamera->setPatternGenerator(HaltonSamplePattern::create(8), scale);
            else
                pCamera->setPatternGenerator(pCamera->getPatternGenerator(), scale);
        }
        else
        {
            if (pCamera->getPatternGenerator())
                pCamera->setPatternGenerator(nullptr, {});
        }
        if (auto pTAA = dynamic_ref_cast<TAAPass>(mpGraph->getPass("TAAPass")))
        {
            pTAA->setEnabled(mTAAEnabled);
        }
    }

    // Update the scene.
    IScene::UpdateFlags sceneUpdates = mpScene->update(pRenderContext, getGlobalClock().getTime());
    if (sceneUpdates != IScene::UpdateFlags::None)
    {
        mpGraph->onSceneUpdates(pRenderContext, sceneUpdates);
    }

    // Push the --capture selection into DebugViewPass before execution so the
    // captured frame reflects the requested buffer.
    if (mAutoCapture)
    {
        auto pDebug = dynamic_ref_cast<DebugViewPass>(mpGraph->getPass("DebugViewPass"));
        if (pDebug)
        {
            pDebug->setSelection(mCaptureBufferIdx, mCaptureChannelIdx);
        }
    }

    // Execute the render graph.
    mpGraph->getPassesDictionary()[kRenderPassRefreshFlags] = RenderPassRefreshFlags::None;
    mpGraph->execute(pRenderContext);

    // Blit the graph output to the frame buffer.
    if (mpGraph->getOutputCount() > 0)
    {
        ref<Texture> pOutTex = mpGraph->getOutput(0)->asTexture();
        if (pOutTex)
        {
            pRenderContext->blit(pOutTex->getSRV(), pTargetFbo->getRenderTargetView(0));
        }
    }

    // Auto-capture for headless validation. EXR is mandatory: Falcor's PNG
    // conversion reinterprets float bytes as uint8. The target FBO is
    // RGBA16Float, so float buffers survive the round trip.
    if (mAutoCapture)
    {
        ++mFrameCount;
        if (mFrameCount == mCaptureFrame)
        {
            std::string filename = "Kroken_capture_" + mCaptureBufferName + "_" + mCaptureChannelName + ".exr";
            auto path = getRuntimeDirectory() / filename;

            pTargetFbo->getColorTexture(0)->captureToFile(
                0, 0, path, Bitmap::FileFormat::ExrFile, Bitmap::ExportFlags::Uncompressed, false);
            logInfo("Captured {} to {}", mCaptureBufferName, path.filename().string());
            shutdown(0);
        }
    }
}

void Kroken::onGuiRender(Gui* pGui)
{
    Gui::Window w(pGui, "Kroken", {360, 520}, {10, 80});
    renderGlobalUI(pGui);

    if (auto g = w.group("Debug View", true))
    {
        if (mpGraph)
        {
            if (auto pDebug = dynamic_ref_cast<DebugViewPass>(mpGraph->getPass("DebugViewPass")))
            {
                // A restricted buffer list: DebugViewPass' own UI also offers
                // "Indirect Lighting" (12) and "Field Debug" (14), whose inputs are
                // unconnected in this graph and would sample a null descriptor.
                static const Gui::DropdownList kBuffers = {
                    {0, "Final result"},
                    {1, "GBufferA (normal/per-object)"},
                    {2, "GBufferB (metallic/spec/rough/ID)"},
                    {3, "GBufferC (baseColor/AO)"},
                    {4, "GBufferD (custom data)"},
                    {5, "GBufferE (shadow factors)"},
                    {6, "Depth"},
                    {7, "Emissive"},
                    {8, "ShadowMask (surface/SSS trans)"},
                    {9, "AO Map"},
                    {10, "Direct Lighting"},
                    {11, "Ambient Lighting"},
                    {13, "Composited Lighting"},
                    {15, "Shading Model"},
                };
                static const Gui::DropdownList kChannels = {
                    {0, "RGB"}, {1, "R"}, {2, "G"}, {3, "B"}, {4, "A"},
                };

                uint32_t buffer = mCaptureBufferIdx;
                uint32_t channel = mCaptureChannelIdx;
                bool changed = g.dropdown("Buffer", kBuffers, buffer);
                changed |= g.dropdown("Channel", kChannels, channel);
                if (changed)
                {
                    mCaptureBufferIdx = buffer;
                    mCaptureChannelIdx = channel;
                }
                pDebug->setSelection(mCaptureBufferIdx, mCaptureChannelIdx);
            }
            else
            {
                g.text("DebugViewPass not found in graph.");
            }
        }
    }

    if (auto g = w.group("Sun Light", true))
    {
        if (mpSunLight)
        {
            float3 dir = mpSunLight->getWorldDirection();
            if (g.var("Direction", dir, -1.f, 1.f, 0.01f))
                mpSunLight->setWorldDirection(normalize(dir));

            float3 intensity = mpSunLight->getIntensity();
            if (g.var("Intensity", intensity, 0.f, 100.f, 0.01f))
            {
                mpSunLight->setIntensity(intensity);
                // Keep the shaded direction/intensity in sync with the light.
                mSunDirection = normalize(mpSunLight->getWorldDirection());
                mSunIntensity = intensity.x;
            }
        }
        else
        {
            g.text("No directional light in scene.");
            g.tooltip("Delete the scene cache and rerun to re-import it.", true);
        }
    }

    if (auto g = w.group("Temporal AA", true))
    {
        g.checkbox("Enable TAA", mTAAEnabled);
        g.tooltip("Enables Halton(8) camera jitter + temporal resolve before tonemap.", true);
        if (mTAAEnabled && mpGraph)
        {
            if (auto pTAA = dynamic_ref_cast<TAAPass>(mpGraph->getPass("TAAPass")))
                pTAA->renderUI(g);
            else
                g.text("TAAPass not found in graph.");
        }
    }

    if (auto g = w.group("Scene", true))
    {
        if (mpScene) mpScene->renderUI(g);
    }

    if (auto g = w.group("Render Graph", true))
    {
        if (mpGraph) mpGraph->renderUI(getRenderContext(), g);
    }
}

void Kroken::saveScreenshot(RenderContext* pRenderContext, const ref<Texture>& pSrc)
{
    if (!pSrc) return;

    // Round-trip through an RGBA8UnormSrgb render target: the blit does the
    // linear->sRGB encoding the FP16 swapchain otherwise leaves to the display,
    // and the unorm write clamps out-of-range values.
    ref<Texture> pLdr = getDevice()->createTexture2D(
        pSrc->getWidth(),
        pSrc->getHeight(),
        ResourceFormat::RGBA8UnormSrgb,
        1,
        1,
        nullptr,
        ResourceBindFlags::RenderTarget | ResourceBindFlags::ShaderResource
    );
    pRenderContext->blit(pSrc->getSRV(), pLdr->getRTV());

    // Same naming scheme as SampleApp::captureScreen, so F11 and F12 share one
    // numbering sequence.
    auto path = findAvailableFilename(getExecutableName(), getRuntimeDirectory(), "png");

    // No ExportAlpha: alpha in the target FBO is leftover from the graph and would
    // make viewers show the shot as partly transparent.
    pLdr->captureToFile(0, 0, path, Bitmap::FileFormat::PngFile, Bitmap::ExportFlags::None, false);
}

bool Kroken::onKeyEvent(const KeyboardEvent& keyEvent)
{
    if (keyEvent.type == KeyboardEvent::Type::KeyPressed && keyEvent.mods == Input::ModifierFlags::None)
    {
        if (keyEvent.key == Input::Key::F11)
        {
            mScreenshotRequested = true;
            return true;
        }
    }
    if (mpGraph && mpGraph->onKeyEvent(keyEvent)) return true;
    if (mpScene && mpScene->onKeyEvent(keyEvent)) return true;
    return false;
}

bool Kroken::onMouseEvent(const MouseEvent& mouseEvent)
{
    if (mpGraph && mpGraph->onMouseEvent(mouseEvent)) return true;
    if (mpScene && mpScene->onMouseEvent(mouseEvent)) return true;
    return false;
}

int runMain(int argc, char** argv)
{
    SampleAppConfig config;
    config.windowDesc.title = "Kroken";
    config.windowDesc.resizableWindow = true;
    config.windowDesc.width = 1920;
    config.windowDesc.height = 1080;
    config.colorFormat = ResourceFormat::RGBA16Float;

    Kroken app(config);
    app.setCommandLine(argc, argv);
    return app.run();
}

int main(int argc, char** argv)
{
    return catchAndReportAllExceptions([&]() { return runMain(argc, argv); });
}
