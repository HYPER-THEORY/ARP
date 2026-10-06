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
#include "Wheatfield.h"
#include "Utils/Image/Bitmap.h"
#include "Core/Plugin.h"
#include "RenderGraph/RenderPass.h"
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
#include "IndirectApproxPass/IndirectApproxPass.h"

#include "Utils/SampleGenerators/HaltonSamplePattern.h"

#include <random>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>

// Compile-time switch between the path-tracing graph and the deferred
// GBuffer + Shadow + Skybox + DI graph. Define FALCOR_WHEATFIELD_USE_PATHTRACER
// to 1 to use the path tracer; otherwise the deferred graph is built.
#ifndef FALCOR_WHEATFIELD_USE_PATHTRACER
#define FALCOR_WHEATFIELD_USE_PATHTRACER 0
#endif

FALCOR_EXPORT_D3D12_AGILITY_SDK

namespace
{
    // Helper to register Arctic render passes with the in-process plugin
    // registry. Arctic is built as an executable (not a plugin DLL), so its
    // passes must be registered manually before the render graph creates them
    // by name.
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
} // namespace

namespace
{
    // Octahedral mapping helpers (non-equal-area, matching ref engine's OctahedronToUnitVector).
    // See Falcor's MathHelpers.slang for the canonical implementation.

    float2 octWrap(float2 v)
    {
        return float2(1.f - std::abs(v.y), 1.f - std::abs(v.x)) *
               float2(v.x >= 0.f ? 1.f : -1.f, v.y >= 0.f ? 1.f : -1.f);
    }

    float3 octToDir(float2 p)
    {
        // p in [0,1], convert to [-1,1]
        p = p * 2.f - 1.f;
        float3 n(p.x, p.y, 1.f - std::abs(p.x) - std::abs(p.y));
        if (n.z < 0.f) n.xy() = octWrap(n.xy());
        return normalize(n);
    }

    float2 dirToOct(float3 n)
    {
        n = normalize(n);
        float2 p = n.xy() * (1.f / (std::abs(n.x) + std::abs(n.y) + std::abs(n.z)));
        if (n.z < 0.f) p = octWrap(p);
        return p * 0.5f + 0.5f;
    }

    // Falcor's lat-long mapping: uv.x = atan2(x, -z)/(2pi) + 0.5, uv.y = acos(y)/pi.
    float3 latlongToDir(float2 uv)
    {
        float phi = float(M_PI) * (2.f * uv.x - 1.f);
        float theta = float(M_PI) * uv.y;
        float sinTheta = std::sin(theta);
        float cosTheta = std::cos(theta);
        return float3(sinTheta * std::sin(phi), cosTheta, -sinTheta * std::cos(phi));
    }

    /** Convert an octahedral env map EXR file to a lat-long Texture.
        \param pDevice GPU device.
        \param pRenderContext Render context (for texture upload).
        \param path File path to the octahedral EXR.
        \return A lat-long RGB32Float texture, or nullptr on failure.
    */
    ref<Texture> convertOctahedralFileToLatLongTexture(ref<Device> pDevice, RenderContext* pRenderContext, const std::filesystem::path& path)
    {
        // Load the octahedral EXR as a Bitmap (float RGB data).
        auto pBitmap = Bitmap::createFromFile(path, true /*isTopDown*/, Bitmap::ImportFlags::None);
        if (!pBitmap)
        {
            logWarning("Failed to load octahedral env map bitmap from '{}'", path.string());
            return nullptr;
        }

        const uint32_t srcW = pBitmap->getWidth();
        const uint32_t srcH = pBitmap->getHeight();
        const ResourceFormat srcFormat = pBitmap->getFormat();

        // We expect a float format (RGB32Float or RGBA32Float) from EXR.
        uint32_t srcChannels = getFormatChannelCount(srcFormat);
        if (srcChannels < 3)
        {
            logWarning("Octahedral env map has {} channels, expected >= 3", srcChannels);
            return nullptr;
        }

        const uint8_t* pSrc = pBitmap->getData();
        const uint32_t srcBytesPerPixel = getFormatBytesPerBlock(srcFormat);
        const uint32_t srcRowPitch = pBitmap->getRowPitch();

        // Create the lat-long target: 2:1 aspect, same vertical resolution as source.
        const uint32_t dstW = srcW * 2;
        const uint32_t dstH = srcW; // square source => dst height = src width to keep 2:1

        std::vector<float> dstData(dstW * dstH * 4, 0.f); // RGBA32Float

        auto sampleSrcBilinear = [&](float u, float v) -> float3
        {
            // Sample source octahedral texture with bilinear filtering (manual).
            float fx = u * srcW - 0.5f;
            float fy = v * srcH - 0.5f;
            int32_t x0 = static_cast<int32_t>(std::floor(fx));
            int32_t y0 = static_cast<int32_t>(std::floor(fy));
            float tx = fx - x0;
            float ty = fy - y0;

            auto readPixel = [&](int32_t x, int32_t y) -> float3
            {
                // Wrap horizontally (octahedral is wrapped), clamp vertically.
                x = ((x % static_cast<int32_t>(srcW)) + static_cast<int32_t>(srcW)) % static_cast<int32_t>(srcW);
                y = std::clamp(y, 0, static_cast<int32_t>(srcH) - 1);
                const uint8_t* p = pSrc + y * srcRowPitch + x * srcBytesPerPixel;
                const float* pf = reinterpret_cast<const float*>(p);
                return float3(pf[0], pf[1], pf[2]);
            };

            float3 c00 = readPixel(x0, y0);
            float3 c10 = readPixel(x0 + 1, y0);
            float3 c01 = readPixel(x0, y0 + 1);
            float3 c11 = readPixel(x0 + 1, y0 + 1);
            float3 c0 = lerp(c00, c10, tx);
            float3 c1 = lerp(c01, c11, tx);
            return lerp(c0, c1, ty);
        };

        for (uint32_t y = 0; y < dstH; ++y)
        {
            for (uint32_t x = 0; x < dstW; ++x)
            {
                // Lat-long UV in [0,1]. Note: Bitmap isTopDown=true means row 0 is top.
                // Falcor's lat-long map has v=0 at top (y=+1), v=1 at bottom (y=-1).
                float u = (x + 0.5f) / dstW;
                float v = (y + 0.5f) / dstH;

                // Convert lat-long UV to world direction.
                float3 dir = latlongToDir(float2(u, v));

                // Convert world direction to octahedral UV.
                float2 octUv = dirToOct(dir);

                // Sample the octahedral source texture.
                float3 color = sampleSrcBilinear(octUv.x, octUv.y);

                size_t idx = (size_t(y) * dstW + x) * 4;
                dstData[idx + 0] = color.x;
                dstData[idx + 1] = color.y;
                dstData[idx + 2] = color.z;
                dstData[idx + 3] = 1.f;
            }
        }

        // Create the lat-long texture and upload the converted data.
        // createTexture2D with pInitData=nullptr creates an empty texture; we then upload via updateTextureData.
        ref<Texture> pLatLongTex = pDevice->createTexture2D(
            dstW, dstH, ResourceFormat::RGBA32Float, 1, 1, nullptr,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);

        pRenderContext->updateTextureData(pLatLongTex.get(), dstData.data());

        // Generate mipmaps for better filtering at a distance.
        pLatLongTex->generateMips(pRenderContext);

        return pLatLongTex;
    }
}

Wheatfield::Wheatfield(const SampleAppConfig& config) : SampleApp(config) {}
Wheatfield::~Wheatfield() = default;

namespace
{
// Parses a --capture argument of the form "buffer[:channel]" into the integer
// indices DebugViewPass expects, plus the canonical names for the output
// filename. Returns false on unrecognized name; caller falls back to final:RGB.
bool parseCaptureArg(const std::string& arg, uint32_t& bufferIdx, std::string& bufferName, uint32_t& channelIdx, std::string& channelName)
{
    static const std::vector<std::pair<std::string, uint32_t>> kBuffers = {
        {"final", 0},
        {"gBufferA", 1}, {"gBufferB", 2}, {"gBufferC", 3},
        {"gBufferD", 4}, {"gBufferE", 5},
        {"depth", 6}, {"emissive", 7}, {"shadowMask", 8},
        {"aoMap", 9}, {"ao", 9},
        {"di", 10}, {"diColor", 10},
        {"ambient", 11}, {"ambientOnly", 11},
        {"indirect", 12},
        {"composite", 13}, {"ambientColor", 13},
        {"fieldDebug", 14}, {"field", 14},
        {"shadingModel", 15}, {"shadeMode", 15},
    };
    static const std::vector<std::pair<std::string, uint32_t>> kChannels = {
        {"RGB", 0}, {"R", 1}, {"G", 2}, {"B", 3}, {"A", 4},
    };

    std::string bufPart = arg, chanPart;
    auto colon = arg.find(':');
    if (colon != std::string::npos)
    {
        bufPart = arg.substr(0, colon);
        chanPart = arg.substr(colon + 1);
    }

    auto match = [](const std::string& s, const std::vector<std::pair<std::string, uint32_t>>& table, uint32_t& outIdx, std::string& outName) -> bool
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
    };

    if (!match(bufPart, kBuffers, bufferIdx, bufferName)) return false;
    if (chanPart.empty())
    {
        channelIdx = 0;
        channelName = "RGB";
        return true;
    }
    if (!match(chanPart, kChannels, channelIdx, channelName)) return false;
    return true;
}
} // namespace

void Wheatfield::onLoad(RenderContext* pRenderContext)
{
    // Check for --capture flag to auto-capture a screenshot after a few frames.
    // Optional argument selects which buffer to capture:
    //   --capture                  -> final:RGB (backward compatible)
    //   --capture gBufferA         -> gBufferA:RGB
    //   --capture gBufferA:R       -> gBufferA:R channel
    // Supported buffers: final, di, gBufferA..E, depth, shadowMask.
    // Supported channels: RGB, R, G, B, A.
    for (int i = 1; i < __argc; ++i)
    {
        if (std::string(__argv[i]) == "--iaprofile")
        {
            // Capture one of IndirectApproxPass' debug buffers so the shader's
            // field can be diffed offline against the Python SAIL reference:
            //   --iaprofile edown  -> indirect holds the GPU E_down
            //   --iaprofile eup    -> indirect holds the GPU E_up
            //   --iaprofile field  -> fieldDebug holds (density, tau, LAI)
            mIndirectProfile = true;
            mAutoCapture = true;
            parseCaptureArg("indirect", mCaptureBufferIdx, mCaptureBufferName, mCaptureChannelIdx, mCaptureChannelName);
            if (i + 1 < __argc)
            {
                std::string next(__argv[i + 1]);
                if (next == "eup")
                {
                    mIndirectDebugView = 2;
                }
                else if (next == "field")
                {
                    mIndirectDebugView = 0; // raw radiance; we want fieldDebug instead
                    parseCaptureArg("fieldDebug", mCaptureBufferIdx, mCaptureBufferName, mCaptureChannelIdx, mCaptureChannelName);
                }
            }
        }
        if (std::string(__argv[i]) == "--ao")
        {
            // Selects the AO method for headless runs, since the default is SSAO
            // and there is otherwise no way to exercise GTAO without the UI.
            //   --ao gtao | --ao ssao
            if (i + 1 < __argc)
            {
                std::string next(__argv[i + 1]);
                if (next == "gtao" || next == "GTAO") { mAOMethod = AOMethod::GTAO; ++i; }
                else if (next.rfind("gtao:", 0) == 0)
                {
                    // --ao gtao:<stage> dumps one GTAO stage (0=search, 1=spatial, 2=temporal).
                    mAOMethod = AOMethod::GTAO;
                    const std::string digits = next.substr(5);
                    if (digits.size() == 1 && digits[0] >= '0' && digits[0] <= '2')
                        mGTAOOutputStage = (uint32_t)(digits[0] - '0');
                    else
                        logWarning("--ao '{}': stage must be 0, 1 or 2; using 2 (full chain).", next);
                    ++i;
                }
                else if (next == "ssao" || next == "SSAO") { mAOMethod = AOMethod::SSAO; ++i; }
                else logWarning("Unrecognized --ao argument '{}', expected 'gtao' or 'ssao'.", next);
            }
        }
        if (std::string(__argv[i]) == "--captureframe")
        {
            // Frame number the --capture snapshot is taken on (default 256).
            // Consecutive frames are the only way to see a temporal effect: the
            // bush clip's dither, the camera jitter and the TAA history are all
            // frame-index dependent, so N and N+1 differ where a single frame
            // says nothing.
            if (i + 1 < __argc)
            {
                const uint32_t frame = (uint32_t)std::atoi(__argv[i + 1]);
                if (frame > 0) { mCaptureFrame = frame; ++i; }
                else logWarning("--captureframe expects a positive frame number.");
            }
        }
        if (std::string(__argv[i]) == "--capture")
        {
            mAutoCapture = true;
            if (i + 1 < __argc)
            {
                std::string next(__argv[i + 1]);
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

    // Extend asset search paths so we can find assets under the project's assets/ directory.
    // The default search path is <project>/media, but our wheat asset and env map live in <project>/assets.
    AssetResolver::getDefaultResolver().addSearchPath(getProjectDirectory() / "assets");

    // Register Arctic's render passes with the in-process plugin registry.
    PassRegistrar::registerOne<GBufferPass>();
    PassRegistrar::registerOne<ShadowPass>();
    PassRegistrar::registerOne<ShadowProjectionPass>();
    PassRegistrar::registerOne<SkyboxPass>();
    PassRegistrar::registerOne<DIPass>();
    PassRegistrar::registerOne<GTAOPass>();
    PassRegistrar::registerOne<SSAOPass>();
    PassRegistrar::registerOne<AmbientPass>();
    PassRegistrar::registerOne<CompositePass>();
    PassRegistrar::registerOne<MotionVectorPass>();
    PassRegistrar::registerOne<TAAPass>();
    PassRegistrar::registerOne<IndirectApproxPass>();
    PassRegistrar::registerOne<DebugViewPass>();

    // Build the render graph first; the scene is attached after it is built.
    buildRenderGraph();
    rebuildScene(pRenderContext);
}

void Wheatfield::buildRenderGraph()
{
#if FALCOR_WHEATFIELD_USE_PATHTRACER
    buildPathTracerGraph();
#else
    buildDeferredGraph();
#endif
}

void Wheatfield::buildPathTracerGraph()
{
    // Build the standard path tracing graph:
    //   VBufferRT -> PathTracer -> AccumulatePass -> ToneMapper
    // This mirrors scripts/PathTracer.py.
    // Passes are created by type name through the plugin registry, so we don't
    // need to link against the individual RenderPass DLLs directly.
    mpGraph = RenderGraph::create(getDevice(), "WheatfieldGraph");

    Properties vbufferProps;
    vbufferProps["samplePattern"] = std::string("Stratified");
    vbufferProps["sampleCount"] = uint32_t(16);
    vbufferProps["useAlphaTest"] = true;
    mpGraph->createPass("VBufferRT", "VBufferRT", vbufferProps);

    mpGraph->createPass("PathTracer", "PathTracer", Properties{});

    mpGraph->createPass("AccumulatePass", "AccumulatePass", Properties{});

    mpGraph->createPass("ToneMapper", "ToneMapper", Properties{});

    mpGraph->addEdge("VBufferRT.vbuffer", "PathTracer.vbuffer");
    mpGraph->addEdge("VBufferRT.viewW", "PathTracer.viewW");
    mpGraph->addEdge("VBufferRT.mvec", "PathTracer.mvec");
    mpGraph->addEdge("PathTracer.color", "AccumulatePass.input");
    mpGraph->addEdge("AccumulatePass.output", "ToneMapper.src");
    mpGraph->markOutput("ToneMapper.dst");
}

void Wheatfield::buildDeferredGraph()
{
    mpGraph = RenderGraph::create(getDevice(), "WheatfieldDeferredGraph");

    mpGraph->createPass("GBufferPass", "GBufferPass", Properties{});
    mpGraph->createPass("ShadowPass", "ShadowPass", Properties{});
    mpGraph->createPass("SkyboxPass", "SkyboxPass", Properties{});
    mpGraph->createPass("ShadowProjectionPass", "ShadowProjectionPass", Properties{});
    // AO pass is created under a uniform node name "AOPass" so the downstream
    // edges / markOutput / capture plumbing work identically for both methods.
    // GTAOPass additionally consumes motionVecs for its temporal filter; SSAOPass
    // has no such input, so that edge is added only in GTAO mode.
    {
        const std::string aoType = (mAOMethod == AOMethod::SSAO) ? "SSAOPass" : "GTAOPass";
        Properties aoProps = {{{"effectRadius", 10.0f}, {"thickness", 0.1f}, {"power", 2.0f}}};
        if (mAOMethod == AOMethod::GTAO && mGTAOOutputStage != 2u) aoProps["outputStage"] = mGTAOOutputStage;
        mpGraph->createPass("AOPass", aoType, aoProps);
    }
    mpGraph->createPass("DIPass", "DIPass", Properties{});
    // The indirect pass supplies the >= 2-bounce canopy transport that neither
    // DIPass (1-bounce sun) nor AmbientPass (1-bounce sky) accounts for. It only
    // needs the GBuffer, so it can run alongside DIPass.
    {
        Properties ipProps;
        const char* ipType = "IndirectApproxPass";
        if (mIndirectProfile)
        {
            ipProps["debugView"] = mIndirectDebugView;
        }
        ipProps["lai"] = 6.5f;
        ipProps["densityContrast"] = 2.5f;
        mpGraph->createPass("IndirectPass", ipType, ipProps);
    }
    mpGraph->createPass("AmbientPass", "AmbientPass", Properties{});
    mpGraph->createPass("CompositePass", "CompositePass", Properties{{{"indirectIntensity", 4.0f}, {"skyboxIntensity", 4.0f}}});
    // MotionVectorPass reconstructs camera-only velocity from depth; TAAPass
    // resolves temporally on HDR color before tonemapping (UE convention).
    mpGraph->createPass("MotionVectorPass", "MotionVectorPass", Properties{});
    mpGraph->createPass("TAAPass", "TAAPass", Properties{});
    mpGraph->createPass("ToneMapper", "ToneMapper", Properties{});
    mpGraph->createPass("DebugViewPass", "DebugViewPass", Properties{});

    mpGraph->addEdge("GBufferPass.depth", "SkyboxPass.sceneDepth");
    mpGraph->addEdge("GBufferPass.depth", "ShadowProjectionPass.sceneDepth");
    mpGraph->addEdge("GBufferPass.gBufferA", "ShadowProjectionPass.gBufferA");
    mpGraph->addEdge("GBufferPass.gBufferB", "ShadowProjectionPass.gBufferB");
    mpGraph->addEdge("GBufferPass.gBufferD", "ShadowProjectionPass.gBufferD");
    mpGraph->addEdge("ShadowPass.shadowDepth", "ShadowProjectionPass.shadowDepth");

    mpGraph->addEdge("GBufferPass.gBufferA", "AOPass.gBufferA");
    mpGraph->addEdge("GBufferPass.depth", "AOPass.sceneDepth");
    if (mAOMethod == AOMethod::GTAO)
    {
        // GTAO's temporal filter reprojects its half-res history. Without this the
        // pass falls back to spatial-only filtering (and says so in the log).
        mpGraph->addEdge("MotionVectorPass.motionVecs", "AOPass.motionVecs");
    }

    mpGraph->addEdge("GBufferPass.gBufferA", "DIPass.gBufferA");
    mpGraph->addEdge("GBufferPass.gBufferB", "DIPass.gBufferB");
    mpGraph->addEdge("GBufferPass.gBufferC", "DIPass.gBufferC");
    mpGraph->addEdge("GBufferPass.gBufferD", "DIPass.gBufferD");
    mpGraph->addEdge("GBufferPass.gBufferE", "DIPass.gBufferE");
    mpGraph->addEdge("GBufferPass.depth", "DIPass.sceneDepth");
    mpGraph->addEdge("ShadowProjectionPass.shadowMask", "DIPass.shadowMask");

    mpGraph->addEdge("GBufferPass.depth", "IndirectPass.sceneDepth");
    mpGraph->addEdge("GBufferPass.gBufferA", "IndirectPass.gBufferA");
    mpGraph->addEdge("GBufferPass.gBufferB", "IndirectPass.gBufferB");
    mpGraph->addEdge("GBufferPass.gBufferC", "IndirectPass.gBufferC");
    mpGraph->addEdge("GBufferPass.gBufferD", "IndirectPass.gBufferD");

    mpGraph->addEdge("GBufferPass.gBufferA", "AmbientPass.gBufferA");
    mpGraph->addEdge("GBufferPass.gBufferB", "AmbientPass.gBufferB");
    mpGraph->addEdge("GBufferPass.gBufferC", "AmbientPass.gBufferC");
    mpGraph->addEdge("GBufferPass.gBufferD", "AmbientPass.gBufferD");
    mpGraph->addEdge("GBufferPass.depth", "AmbientPass.sceneDepth");
    mpGraph->addEdge("AOPass.aoMap", "AmbientPass.aoMap");

    // CompositePass sums the four lighting terms. sceneDepth is what tells it
    // which pixels are background (skybox) and which are geometry.
    mpGraph->addEdge("GBufferPass.depth", "CompositePass.sceneDepth");
    mpGraph->addEdge("SkyboxPass.skyboxColor", "CompositePass.skyboxColor");
    mpGraph->addEdge("DIPass.color", "CompositePass.diColor");
    mpGraph->addEdge("AmbientPass.ambient", "CompositePass.ambient");
    mpGraph->addEdge("IndirectPass.indirect", "CompositePass.indirect");
    // Emissive bypasses lighting entirely, exactly as UE's base pass writes it
    // straight into SceneColor. It is also the only radiance an Unlit pixel gets.
    mpGraph->addEdge("GBufferPass.emissive", "CompositePass.emissive");

    // MotionVectorPass consumes GBufferPass depth; TAAPass consumes its
    // velocity plus CompositePass.color (HDR), and feeds ToneMapper. TAAPass also
    // needs depth for UE's nearest-depth velocity dilation (AA_CROSS).
    mpGraph->addEdge("GBufferPass.depth", "MotionVectorPass.sceneDepth");
    mpGraph->addEdge("MotionVectorPass.motionVecs", "TAAPass.motionVecs");
    mpGraph->addEdge("GBufferPass.depth", "TAAPass.sceneDepth");
    mpGraph->addEdge("CompositePass.color", "TAAPass.colorIn");
    mpGraph->addEdge("TAAPass.colorOut", "ToneMapper.src");

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
    mpGraph->addEdge("IndirectPass.indirect", "DebugViewPass.indirect");
    mpGraph->addEdge("IndirectPass.fieldDebug", "DebugViewPass.fieldDebug");
    mpGraph->addEdge("GBufferPass.emissive", "DebugViewPass.emissive");
    mpGraph->markOutput("DebugViewPass.output");
}

void Wheatfield::rebuildScene(RenderContext* pRenderContext)
{
    SceneBuilder builder(getDevice(), Settings(), SceneBuilder::Flags::DontMergeMaterials);

    // White ground plane.
    // createQuad is in the XZ plane with normal +Y. We rotate it -90° around X
    // to lay it in the XY plane with normal +Z (the new "up" direction).
    auto groundMesh = TriangleMesh::createQuad(float2(mGroundSize));
    auto groundMat = StandardMaterial::create(getDevice(), "Ground");
    groundMat->setBaseColor(float4(0.2f, 0.2f, 0.2f, 1.f));
    groundMat->setRoughness(1.f);
    groundMat->setMetallic(0.f);
    // IoR is left at Falcor's default 1.5, which GBufferPass converts to UE's
    // default dielectric Specular of 0.5 (see Materials/Standard/StandardGBuffer.slang).
    auto groundMeshID = builder.addTriangleMesh(groundMesh, groundMat);
    Transform groundTransform;
    groundTransform.setRotationEulerDeg(float3(90.f, 0.f, 0.f));
    SceneBuilder::Node groundNode{ "Ground", groundTransform.getMatrix() };
    auto groundNodeID = builder.addNode(groundNode);
    builder.addMeshInstance(groundNodeID, groundMeshID);

    // Reference sphere at the center of the wheatfield, resting on the ground
    // (z = radius so it sits on the XY ground plane). Plain neutral standard
    // material so it reads as a calibration object alongside the wheat.
    //auto sphereMesh = TriangleMesh::createSphere(4, 64, 32);
    //auto sphereMat = StandardMaterial::create(getDevice(), "ReferenceSphere");
    //sphereMat->setBaseColor(float4(0.2f, 0.2f, 0.2f, 1.f));
    //sphereMat->setRoughness(1.0f);
    //sphereMat->setMetallic(0.f);
    //auto sphereMeshID = builder.addTriangleMesh(sphereMesh, sphereMat);
    //Transform sphereTransform;
    //sphereTransform.setTranslation(float3(0.f, 0.f, 4.f));
    //SceneBuilder::Node sphereNode{ "ReferenceSphere", sphereTransform.getMatrix() };
    //auto sphereNodeID = builder.addNode(sphereNode);
    //builder.addMeshInstance(sphereNodeID, sphereMeshID);

    // Load the wheat asset as a triangle mesh.
    // Note: TriangleMesh::createFromFile pre-transforms vertices and discards
    // material assignments, so we attach our own BushMaterial below.
    auto wheatPath = AssetResolver::getDefaultResolver().resolvePath(kWheatAssetPath);
    if (wheatPath.empty())
    {
        FALCOR_THROW("Failed to resolve wheat asset path: {}", kWheatAssetPath);
    }
    auto wheatMesh = TriangleMesh::createFromFile(wheatPath, true /*smoothNormals*/);
    if (!wheatMesh)
    {
        FALCOR_THROW("Failed to load wheat asset: {}", wheatPath.string());
    }

    // Create the bush material.
    mpBushMat = BushMaterial::create(getDevice(), "WheatFoliage");
    // Single alpha threshold, shared by the GBuffer clip and MaterialSystem::
    // alphaTest (ShadowPass / VBufferRT). This used to be two values -- 0.333 for
    // the GBuffer clip and 0.12 for the shadow test -- which let the shadow
    // silhouette drift from the geometry silhouette. 0.333 is the chaos default
    // and preserves the visible geometry; shadows now match it.
    mpBushMat->setAlphaThreshold(0.333f);

    // ---- Bush parameters ----
    //mpBushMat->setBaseColorMultiplier(float4(0.43f, 0.30f, 0.31f, 1.f));
    //mpBushMat->setBaseColorBrightness(1.22f);
    //mpBushMat->setBaseColorContrast(1.85f);
    //mpBushMat->setBaseColorSaturation(0.95f);

    //mpBushMat->setBaseColorMultiplier(float4(1.12f, 0.68f, 1.12f, 1.f));
    //mpBushMat->setBaseColorBrightness(1.45f);
    //mpBushMat->setBaseColorContrast(1.30f);
    //mpBushMat->setBaseColorSaturation(1.15f);

    //mpBushMat->setSubsurfaceColorMultiplier(float4(0.90f, 0.70f, 0.70f, 1.f));
    //mpBushMat->setSubsurfaceColorBrightness(0.95f);
    //mpBushMat->setSubsurfaceColorContrast(1.0f);
    //mpBushMat->setSubsurfaceColorSaturation(0.85f);

    mpBushMat->setNormalStrength(3.0f);
    mpBushMat->setRoughnessContrast(1.0f);
    mpBushMat->setRoughnessScale(1.0f);
    mpBushMat->setSpecularValue(1.0f);
    mpBushMat->setOpacityValue(1.0f);

    // Bind textures. Channel layout is documented in BushMaterialData.slang:
    // BaseColor = RGB albedo + A opacity, Normal = RG normal + B specular +
    // A roughness, Transmission = RGB subsurface color + A opacity.
    builder.loadMaterialTexture(mpBushMat, Material::TextureSlot::BaseColor, kAlbedoTexturePath);
    builder.loadMaterialTexture(mpBushMat, Material::TextureSlot::Normal, kNormalTexturePath);
    builder.loadMaterialTexture(mpBushMat, Material::TextureSlot::Transmission, kSubsurfaceTexturePath);

    auto wheatMeshID = builder.addTriangleMesh(wheatMesh, mpBushMat);

    // Scatter wheat instances on the ground plane (now in the XY plane, +Z is up).
    // The wheat .fbx model is authored with +Z as up (model space), matching our
    // world up axis, so we only apply a small yaw around Z per instance.
    std::mt19937 rng(mSeed);
    auto distPos = std::uniform_real_distribution<float>(-mGroundSize * 0.5f, mGroundSize * 0.5f);
    // Per-instance random ranges (half-open [lo, hi)). See Wheatfield::m*Range.
    auto distYaw   = std::uniform_real_distribution<float>(mYawRange.lo,   mYawRange.hi);
    auto distTilt  = std::uniform_real_distribution<float>(mTiltRange.lo,  mTiltRange.hi);
    auto distScale = std::uniform_real_distribution<float>(mScaleRange.lo, mScaleRange.hi);
    auto distSink  = std::uniform_real_distribution<float>(mSinkRange.lo,  mSinkRange.hi);
    // Lodging direction: base angle ± symmetric jitter. 0° = +X, CCW positive.
    auto distTiltDir = std::uniform_real_distribution<float>(
        mTiltDirBaseDeg - mTiltDirJitterDeg, mTiltDirBaseDeg + mTiltDirJitterDeg);

    const uint32_t instanceCount = static_cast<uint32_t>(mWheatDensity * mGroundSize * mGroundSize);
    for (uint32_t i = 0; i < instanceCount; ++i)
    {
        // Position on the XY plane (z=0 is the ground).
        float2 pos2d(distPos(rng), distPos(rng));
        float scale = distScale(rng);
        float sink  = distSink(rng);
        // Yaw around Z (the up axis): self-rotation of the wheat in degrees.
        float yawDeg   = distYaw(rng);
        // Lodging lean magnitude and direction (world-fixed, 0° = +X CCW+).
        float tiltDeg  = distTilt(rng);
        float dirDeg   = distTiltDir(rng);

        // Build the per-instance rotation as: tilt * yaw. The wheat asset is
        // authored +Z up (model space) matching world up, so:
        //   1. yaw spins the wheat around the vertical (world Z) axis;
        //   2. tilt leans it by `tiltDeg` around a horizontal axis that is
        //      perpendicular to the lodging direction and lies in the XY plane.
        // Direction 0° = +X, CCW positive; the lean axis is dir rotated +90°
        // so that a +90° direction (default) leans the wheat toward +Y.
        // Tilt direction is world-fixed: it does NOT rotate with yaw, so the
        // whole field leans the same way (like wind), with small per-instance
        // variation from distTiltDir.
        const float deg2rad = float(M_PI) / 180.f;
        float yawRad  = yawDeg  * deg2rad;
        float tiltRad = tiltDeg * deg2rad;
        float dirRad  = dirDeg  * deg2rad;
        // Lean axis: horizontal, perpendicular to lodging direction.
        // dir=(cos d, sin d) -> axis=(-sin d, cos d, 0) (dir rotated +90°).
        float3 tiltAxis(-std::sin(dirRad), std::cos(dirRad), 0.f);

        quatf qYaw  = math::quatFromAngleAxis(yawRad,  float3(0.f, 0.f, 1.f));
        quatf qTilt = math::quatFromAngleAxis(tiltRad, tiltAxis);
        // tilt * yaw: yaw first (self-spin), then tilt (lean in world space).
        quatf q = math::normalize(math::mul(qTilt, qYaw));

        // Translation: place on the ground, then sink below it by `sink`.
        // Scaling is applied first by the Transform (ScaleRotateTranslate), so
        // the sink distance is in world units, not scaled by the instance.
        Transform t;
        t.setTranslation(float3(pos2d.x, pos2d.y, -sink));
        t.setRotation(q);
        t.setScaling(float3(scale));

        SceneBuilder::Node node{ "wheat_" + std::to_string(i), t.getMatrix() };
        auto nodeID = builder.addNode(node);
        builder.addMeshInstance(nodeID, wheatMeshID);
    }

    // Camera.
    // +Z is up, ground is in the XY plane. Place the camera above the ground
    // looking at the wheat field.
    auto cam = Camera::create("MainCamera");
    cam->setName("MainCamera");
    //cam->setPosition(float3(0.f, -10.f, 3.f));
    //cam->setTarget(float3(0.f, 0.f, 1.5f));
    //cam->setUpVector(float3(0, 0, 1));
    cam->setPosition(float3(18.f, -18.f, 4.f));
    cam->setTarget(float3(17.f, -17.f, 3.75f));
    cam->setUpVector(float3(0, 0, 1));
    cam->setFocalLength(50.f);
    builder.addCamera(cam);

    // Directional light (sun). Used as the shadow caster and the direct
    // lighting source in the deferred graph. Only the first DirectionalLight
    // in the scene is shadowed by ShadowPass/ShadowProjectionPass.
    mpSunLight = DirectionalLight::create("Sun");
    mpSunLight->setWorldDirection(normalize(float3(0.94f, -0.03f, -0.33f)));
    mpSunLight->setIntensity(float3(3.75f, 3.6f, 3.9f));
    builder.addLight(mpSunLight);

    // Environment map (used both as distant light and sky background).
    // The source EXR is an octahedral env map (square), but Falcor's EnvMap
    // expects lat-long (2:1) format. Convert at load time.
    auto envPath = AssetResolver::getDefaultResolver().resolvePath(kEnvMapPath);
    ref<EnvMap> envMap;
    if (!envPath.empty())
    {
        if constexpr (kIsOctahedralEnvMap)
        {
            auto pLatLongTex = convertOctahedralFileToLatLongTexture(getDevice(), pRenderContext, envPath);
            if (pLatLongTex)
            {
                envMap = EnvMap::create(getDevice(), pLatLongTex);
            }
        }
        else
        {
            envMap = EnvMap::createFromFile(getDevice(), envPath);
            envMap->setRotation({90, 0, 0});
        }
    }
    if (envMap)
    {
        builder.setEnvMap(envMap);
    }
    else
    {
        logWarning("Failed to load environment map: {}", kEnvMapPath);
    }

    mpScene = builder.getScene();

    // World up is +Z (ground in XY plane). Scene defaults to Y-up, which would
    // make FirstPerson/Orbiter controllers treat Y as up. Override to Z+.
    mpScene->setUpDirection(Scene::UpDirection::ZPos);

    // Adjust camera depth range based on scene bounds.
    float radius = mpScene->getSceneBounds().radius();
    mpScene->setCameraSpeed(radius * 0.25f);
    auto pCamera = mpScene->getCamera();
    if (pCamera)
    {
        pCamera->setDepthRange(std::max(0.1f, radius / 750.0f), radius * 10.0f);
        pCamera->setAspectRatio((float)getTargetFbo()->getWidth() / (float)getTargetFbo()->getHeight());
    }

    // Attach scene to render graph.
    mpGraph->setScene(mpScene);
}

void Wheatfield::onResize(uint32_t width, uint32_t height)
{
    // The base class resizes the target FBO. We need to resize the render graph
    // to match before it executes. The graph compiles on the first execute call
    // after the scene is set and size is known.
    if (mpGraph) mpGraph->onResize(getTargetFbo().get());
    if (mpScene) mpScene->setCameraAspectRatio((float)width / (float)height);
}

void Wheatfield::onFrameRender(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo)
{
    // Service a pending F11 request before this frame touches the FBO. What's in
    // there right now is the previous frame after UI compositing, i.e. exactly
    // what was presented when the key was pressed.
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

#if !FALCOR_WHEATFIELD_USE_PATHTRACER
    // Apply (or clear) sub-pixel camera jitter for TAA. Falcor's Camera takes a
    // CPUSampleGenerator + scale; the jitter flows into both the jittered
    // projection (so GBufferPass rasterizes sub-pixel-offset geometry) and the
    // MotionVectorPass +jitter term. Mirrors GBufferBase::updateFrameDim.
    {
        auto cam = mpScene->getCamera();
        const auto& fbo = getTargetFbo();
        const float2 scale = 1.f / float2(float(fbo->getWidth()), float(fbo->getHeight()));
        if (mTAAEnabled)
        {
            if (!cam->getPatternGenerator())
                cam->setPatternGenerator(HaltonSamplePattern::create(8), scale);
            else
                cam->setPatternGenerator(cam->getPatternGenerator(), scale);
        }
        else
        {
            if (cam->getPatternGenerator())
                cam->setPatternGenerator(nullptr, {});
        }
        // Push the TAA toggle into the TAAPass so it passes through when off.
        if (auto pPass = mpGraph->getPass("TAAPass"))
        {
            if (auto pTAA = dynamic_ref_cast<TAAPass>(pPass))
                pTAA->setEnabled(mTAAEnabled);
        }
    }
#endif

    // Update scene.
    IScene::UpdateFlags sceneUpdates = mpScene->update(pRenderContext, getGlobalClock().getTime());
    if (sceneUpdates != IScene::UpdateFlags::None)
    {
        mpGraph->onSceneUpdates(pRenderContext, sceneUpdates);
    }

    // Push the --capture selection into DebugViewPass before execute so the
    // captured frame reflects the requested buffer. Cheap (two uint32 writes)
    // and idempotent, so we just do it every frame when capturing.
    if (mAutoCapture)
    {
        auto pPass = mpGraph->getPass("DebugViewPass");
        if (auto pDebug = dynamic_ref_cast<DebugViewPass>(pPass))
        {
            pDebug->setSelection(mCaptureBufferIdx, mCaptureChannelIdx);
        }
        else if (mCaptureBufferIdx != 0)
        {
            // Path-tracer graph (or any graph without DebugViewPass) has no
            // intermediate buffers to switch to — only final is capturable.
            logWarning("--capture requested non-final buffer but DebugViewPass is not in the graph; capturing final instead.");
            mCaptureBufferIdx = 0;
            mCaptureBufferName = "final";
        }
    }

    // Execute the render graph.
    mpGraph->getPassesDictionary()[kRenderPassRefreshFlags] = RenderPassRefreshFlags::None;
    mpGraph->execute(pRenderContext);

    // Blit main graph output to the frame buffer.
    if (mpGraph->getOutputCount() > 0)
    {
        ref<Texture> pOutTex = mpGraph->getOutput(0)->asTexture();
        if (pOutTex)
        {
            pRenderContext->blit(pOutTex->getSRV(), pTargetFbo->getRenderTargetView(0));
        }
    }

    // Auto-capture for headless validation. Every buffer uses the same path:
    // DebugViewPass writes the --capture selection to its output, that output is
    // blitted to the target FBO just above, and we capture the target FBO. The
    // target FBO is RGBA16Float (see main()), so float buffers survive the round
    // trip. EXR is mandatory: Falcor's PNG conversion mis-handles float textures
    // (it reinterprets float bytes as uint8).
    if (mAutoCapture)
    {
        ++mFrameCount;
        // Capture a single frame once TAA has converged (frame 256 by default),
        // then shut down. --captureframe moves the trigger, which is how a
        // temporal effect gets validated: dither/jitter/TAA behaviour is invisible
        // in one frame and only shows up when consecutive frames are compared.
        if (mFrameCount == mCaptureFrame)
        {
            std::string filename = "Wheatfield_capture_" + mCaptureBufferName + "_" + mCaptureChannelName + ".exr";
            auto path = getRuntimeDirectory() / filename;

            pTargetFbo->getColorTexture(0)->captureToFile(
                0, 0, path, Bitmap::FileFormat::ExrFile, Bitmap::ExportFlags::Uncompressed, false);
            logInfo("Captured {} to {}", mCaptureBufferName, path.filename().string());
            shutdown(0);
        }
    }
}

void Wheatfield::onGuiRender(Gui* pGui)
{
    Gui::Window w(pGui, "Wheatfield", {350, 500}, {10, 80});
    renderGlobalUI(pGui);

    if (w.button("Rebuild Instances")) rebuildScene(getRenderContext());

    w.slider("Density (per m^2)", mWheatDensity, 0.01f, 5.f);
    w.var("Seed", mSeed);
    w.slider("Ground Size", mGroundSize, 10.f, 400.f);
    w.separator();

#if !FALCOR_WHEATFIELD_USE_PATHTRACER
    if (auto g = w.group("Debug View", true))
    {
        if (mpGraph)
        {
            auto pPass = mpGraph->getPass("DebugViewPass");
            if (auto pDebug = dynamic_ref_cast<DebugViewPass>(pPass))
            {
                pDebug->renderUI(g);
            }
            else
            {
                g.text("DebugViewPass not found in graph.");
            }
        }
    }

    if (auto g = w.group("Ambient Occlusion", true))
    {
        Gui::DropdownList aoList;
        aoList.push_back({(uint32_t)AOMethod::SSAO, "SSAO"});
        aoList.push_back({(uint32_t)AOMethod::GTAO, "GTAO"});
        uint32_t aoSel = (uint32_t)mAOMethod;
        if (g.dropdown("Method", aoList, aoSel))
        {
            AOMethod newMethod = (AOMethod)aoSel;
            if (newMethod != mAOMethod)
            {
                mAOMethod = newMethod;
                // Rebuild the render graph with the new AO pass, then re-attach
                // the scene and resize. We do NOT call rebuildScene() — that
                // would rebuild all geometry. Toggling resets the AO pass's
                // parameters to defaults (acceptable).
                buildRenderGraph();
                if (mpGraph && mpScene)
                {
                    mpGraph->setScene(mpScene);
                    mpGraph->onResize(getTargetFbo().get());
                }
            }
        }
        g.tooltip("Switches the ambient-occlusion algorithm. "
                  "Switching rebuilds the render graph (brief hitch).", true);
    }

    if (auto g = w.group("Temporal AA", true))
    {
        g.checkbox("Enable TAA", mTAAEnabled);
        g.tooltip("Enables Halton(8) camera jitter + temporal resolve before tonemap. ", true);
        if (mTAAEnabled && mpGraph)
        {
            auto pPass = mpGraph->getPass("TAAPass");
            if (auto pTAA = dynamic_ref_cast<TAAPass>(pPass))
                pTAA->renderUI(g);
            else
                g.text("TAAPass not found in graph.");
        }
    }
#endif

    if (auto g = w.group("Foliage Material", true))
    {
        if (mpBushMat) mpBushMat->renderUI(g);
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
                mpSunLight->setIntensity(intensity);
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

void Wheatfield::saveScreenshot(RenderContext* pRenderContext, const ref<Texture>& pSrc)
{
    if (!pSrc) return;

    // Round-trip through an RGBA8UnormSrgb render target. The blit runs as a
    // shader pass (formats differ), and the sRGB render-target write does the
    // linear->sRGB encoding in hardware, matching what the display does with the
    // FP16 swapchain. Out-of-range values are clamped by the unorm write.
    // The intermediate is transient: a screenshot is rare enough that keeping a
    // full-res texture alive between presses isn't worth it.
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
    // numbering sequence and neither overwrites the other's files.
    auto path = findAvailableFilename(getExecutableName(), getRuntimeDirectory(), "png");

    // No ExportAlpha: alpha in the target FBO is leftover from the graph and
    // would make viewers show the shot as partly transparent. Without the flag
    // Bitmap::saveImage forces it opaque and writes a 24-bit PNG.
    pLdr->captureToFile(0, 0, path, Bitmap::FileFormat::PngFile, Bitmap::ExportFlags::None, false);
}

void Wheatfield::saveCameraPose()
{
    if (!mpScene) return;
    auto pCamera = mpScene->getCamera();
    if (!pCamera) return;

    const float3& pos = pCamera->getPosition();
    const float3& target = pCamera->getTarget();
    const float3& up = pCamera->getUpVector();

    // Plain text, one vector per line, so the file can be diffed or hand-edited.
    // Full float precision keeps the reload bit-exact.
    std::ofstream file(getRuntimeDirectory() / kCameraPoseFileName, std::ios::trunc);
    if (!file) return;
    file << std::setprecision(std::numeric_limits<float>::max_digits10);
    file << "position " << pos.x << ' ' << pos.y << ' ' << pos.z << '\n';
    file << "target " << target.x << ' ' << target.y << ' ' << target.z << '\n';
    file << "up " << up.x << ' ' << up.y << ' ' << up.z << '\n';
}

void Wheatfield::loadCameraPose()
{
    if (!mpScene) return;
    auto pCamera = mpScene->getCamera();
    if (!pCamera) return;

    std::ifstream file(getRuntimeDirectory() / kCameraPoseFileName);
    if (!file) return;

    // Parse into locals first: a truncated or malformed file must leave the live
    // camera untouched rather than move it halfway.
    float3 pos, target, up;
    bool hasPos = false, hasTarget = false, hasUp = false;
    std::string key;
    while (file >> key)
    {
        float3 v;
        if (!(file >> v.x >> v.y >> v.z)) break;
        if (key == "position") { pos = v; hasPos = true; }
        else if (key == "target") { target = v; hasTarget = true; }
        else if (key == "up") { up = v; hasUp = true; }
    }
    if (!hasPos || !hasTarget || !hasUp) return;

    // A zero-length view direction would make the view matrix NaN downstream.
    if (all(pos == target)) return;

    pCamera->setPosition(pos);
    pCamera->setTarget(target);
    pCamera->setUpVector(up);
}

bool Wheatfield::onKeyEvent(const KeyboardEvent& keyEvent)
{
    // F3/F4 save and load the camera pose. Handled before the graph and scene so
    // neither can swallow the key first; this deliberately shadows Scene's F3
    // (addViewpoint), whose viewpoint list is session-only anyway. Unmodified
    // presses only, so combos like Alt+F4 are left to the window manager.
    if (keyEvent.type == KeyboardEvent::Type::KeyPressed && keyEvent.mods == Input::ModifierFlags::None)
    {
        if (keyEvent.key == Input::Key::F3)
        {
            saveCameraPose();
            return true;
        }
        if (keyEvent.key == Input::Key::F4)
        {
            loadCameraPose();
            return true;
        }
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

bool Wheatfield::onMouseEvent(const MouseEvent& mouseEvent)
{
    if (mpGraph && mpGraph->onMouseEvent(mouseEvent)) return true;
    if (mpScene && mpScene->onMouseEvent(mouseEvent)) return true;
    return false;
}

int runMain(int argc, char** argv)
{
    SampleAppConfig config;
    config.windowDesc.title = "Wheatfield";
    config.windowDesc.resizableWindow = true;
    config.windowDesc.width = 1920;
    config.windowDesc.height = 1080;
    config.colorFormat = ResourceFormat::RGBA16Float;

    Wheatfield app(config);
    return app.run();
}

int main(int argc, char** argv)
{
    return catchAndReportAllExceptions([&]() { return runMain(argc, argv); });
}
