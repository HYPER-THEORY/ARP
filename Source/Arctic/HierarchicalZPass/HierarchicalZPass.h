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
#pragma once
#include "Falcor.h"
#include "RenderGraph/RenderPass.h"
#include "RenderGraph/RenderPassHelpers.h"

#include <unordered_map>

using namespace Falcor;

/** Hierarchical Z (HZB) pyramid build — port of UnrealEngine's BuildHZB.

    Builds two mip chains from the scene depth buffer:

      - zFurthest: per-tile max of deviceZ. Every surface the tile covers has a
        deviceZ <= this value, so it is the conservative bound used for
        occlusion culling and for depth-buffer ray marches.
      - zClosest: per-tile min of deviceZ. Every surface the tile covers has a
        deviceZ >= this value.

    Depth convention
    ----------------
    UE's device depth is reverse-Z (1 = near, 0 = far), so its "furthest" HZB is
    min(deviceZ) and its "closest" HZB is max(deviceZ) rounded up. Falcor's
    projection is forward-Z (0 = near, 1 = far), so both comparisons are
    mirrored. The chain names keep UE's meaning; only the direction flips.

    Storage format
    --------------
    Both chains are R32Float, not UE's PF_R16F. fp16 is only viable for a
    reverse-Z pyramid: there distant geometry sits near deviceZ 0, where fp16 is
    at its densest (about 0.5 m of depth per code at 800 m). Under forward-Z the
    same geometry sits near deviceZ 1, where one fp16 ulp is ~235 m at 800 m —
    the whole far field collapses onto two or three codes. R32Float stores the
    exact per-tile min/max, so the values are the tightest correct bound and a
    consumer needs no extra tolerance of its own beyond its hit test.

    Sizing and mip count follow UE exactly:

        hzbSize  = max(roundUpToPowerOfTwo(depthSize) >> 1, 1)
        mipCount = max(floor(log2(max(hzbSize.x, hzbSize.y))), 1)

    For a 1920x1080 depth buffer that is a 1024x1024 mip 0 with 10 levels. The
    pyramid is built in batches matching UE's MAX_MIP_BATCH_SIZE: one dispatch
    writes up to four mip levels, reducing a 16x16 parent footprint to a single
    destination texel in shared memory. Later batches therefore start from the
    previous batch's last mip rather than from the depth buffer.

    Each output carries its full mip chain, so consumers can sample a coarse
    level to skip many texels at once. Note that a consumer declaring an input
    for one of these channels has to declare the same mip count: the render
    graph rejects mismatching mip counts when it merges the fields.

    Inputs/outputs:
      - sceneDepth (D32Float): hardware device depth.
      - zFurthest (R32Float, mip chain): per-tile maximum deviceZ.
      - zClosest  (R32Float, mip chain): per-tile minimum deviceZ.

    See HierarchicalZ.slang for sampling helpers and
    HZBBuild.cs.slang for the reduction itself.
*/
class HierarchicalZPass : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(HierarchicalZPass, "HierarchicalZPass", "Hierarchical Z pyramid build (UE BuildHZB port).");

    static ref<HierarchicalZPass> create(ref<Device> pDevice, const Properties& props) { return make_ref<HierarchicalZPass>(pDevice, props); }

    HierarchicalZPass(ref<Device> pDevice, const Properties& props);

    RenderPassReflection reflect(const CompileData& compileData) override;
    void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    void renderUI(Gui::Widgets& widget) override;
    Properties getProperties() const override;

    /** Computes the pyramid's mip 0 size and its mip count for a depth buffer.

        Exposed so consumers (and debug tooling) can derive the same dimensions
        the pass will declare, instead of duplicating the power-of-two rule.
    */
    static void computeHZBDims(const uint2& depthSize, uint2& outHZBSize, uint32_t& outMipCount);

    // Textures written by the most recent execute(). Valid until the graph is
    // recompiled; intended for debug capture rather than as a steady-state API
    // (consumers should connect the graph outputs instead).
    ref<Texture> getFurthestHZB() const { return mpFurthestHZB; }
    ref<Texture> getClosestHZB() const { return mpClosestHZB; }
    uint32_t getHZBMipCount() const { return mMipCount; }

private:
    /** Compile-time configuration of one reduction program.

        One program per (parent kind, chain selection, batch size). The batch
        size is a permutation rather than a runtime loop bound because the
        shader indexes its output descriptor arrays with literal indices.
    */
    struct ProgramKey    {
        bool parentIsDepth = true;
        bool writeFurthest = true;
        bool writeClosest = true;
        uint32_t mipLevelCount = 1;

        bool operator==(const ProgramKey& other) const
        {
            return parentIsDepth == other.parentIsDepth && writeFurthest == other.writeFurthest &&
                   writeClosest == other.writeClosest && mipLevelCount == other.mipLevelCount;
        }
    };
    struct ProgramKeyHash
    {
        size_t operator()(const ProgramKey& key) const
        {
            return (size_t(key.parentIsDepth) << 0) | (size_t(key.writeFurthest) << 1) | (size_t(key.writeClosest) << 2) |
                   (size_t(key.mipLevelCount) << 3);
        }
    };

    ref<ComputePass> getProgram(const ProgramKey& key);
    void bindAndDispatch(
        RenderContext* pRenderContext,
        const ProgramKey& key,
        ref<Texture> pParent,
        uint32_t parentMip,
        const uint2& srcSize,
        const uint2& dstSize,
        const float2& inputViewportMaxBound,
        ref<Texture> pFurthestHZB,
        ref<Texture> pClosestHZB,
        uint32_t startDestMip,
        uint32_t destMipCount
    );

    std::unordered_map<ProgramKey, ref<ComputePass>, ProgramKeyHash> mPrograms;

    // Set by execute() for debug capture.
    ref<Texture> mpFurthestHZB;
    ref<Texture> mpClosestHZB;
    uint32_t mMipCount = 0;
};
