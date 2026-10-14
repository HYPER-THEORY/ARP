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
#include "HierarchicalZPass.h"

namespace
{
const std::string kProgramFile = "Arctic/HierarchicalZPass/HZBBuild.cs.slang";

const char kInputDepth[] = "sceneDepth";
const char kOutputFurthest[] = "zFurthest";
const char kOutputClosest[] = "zClosest";

/// UE: MAX_MIP_BATCH_SIZE. One dispatch reduces up to this many mip levels.
constexpr uint32_t kMaxMipBatchSize = 4;

/// UE: FPlatformMath::RoundUpToPowerOfTwo, which maps 0 to 0 (not 1).
uint32_t roundUpToPowerOfTwo(uint32_t v)
{
    if (v == 0)
        return 0;
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return v + 1;
}
} // namespace

void HierarchicalZPass::computeHZBDims(const uint2& depthSize, uint2& outHZBSize, uint32_t& outMipCount)
{
    // UE BuildHZB: the pyramid's mip 0 is the depth buffer's size rounded up to
    // a power of two and then halved, not the exact half resolution. The extra
    // power-of-two padding is what lets the last batch reduce cleanly down from
    // a square base. It also means mip 0 covers the depth buffer only in its
    // top-left quadrant of each padded tile; the reduction clamps its reads to
    // the viewport, so the padding sees the edge texels repeated.
    outHZBSize.x = std::max(roundUpToPowerOfTwo(depthSize.x) >> 1, 1u);
    outHZBSize.y = std::max(roundUpToPowerOfTwo(depthSize.y) >> 1, 1u);

    // UE: NumMips = max(floor(log2(max(HZBSize.x, HZBSize.y))), 1). The chain
    // therefore ends at a 2 texel level, not at 1x1 (or at 2x1 for a
    // non-square pyramid, where the shorter axis has already reached 1).
    const uint32_t largerAxis = std::max(outHZBSize.x, outHZBSize.y);
    outMipCount = std::max(static_cast<uint32_t>(std::floor(std::log2(static_cast<float>(largerAxis)))), 1u);
}

HierarchicalZPass::HierarchicalZPass(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice) {}

Properties HierarchicalZPass::getProperties() const
{
    // Nothing configurable: the pyramid's shape is derived from the depth
    // buffer, as in UE.
    return {};
}

RenderPassReflection HierarchicalZPass::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;

    // The pass sizes its outputs from the depth buffer, so it needs the depth
    // dimensions at compile time. connectedResources carries the upstream
    // field's dimensions once the graph has edges; on the first pass (during
    // execution-order resolution) it is empty, and the default texture size is
    // the depth buffer's size anyway. Falling back keeps reflect() usable in
    // both cases instead of throwing or emitting a 1x1 pyramid.
    uint2 depthSize = compileData.defaultTexDims;
    if (const RenderPassReflection::Field* pField = compileData.connectedResources.getField(kInputDepth))
    {
        if (pField->getWidth() && pField->getHeight())
            depthSize = uint2(pField->getWidth(), pField->getHeight());
    }

    // Input: the G-buffer's hardware depth. Dimensions are left at 0 so the field
    // adapts to whatever the producer declares; the format is picked up from the
    // upstream edge (GBufferPass declares D32Float).
    reflector.addInput(kInputDepth, "GBuffer scene depth (D32Float)")
        .format(ResourceFormat::Unknown)
        .bindFlags(ResourceBindFlags::ShaderResource)
        .texture2D(0, 0);

    uint2 hzbSize;
    uint32_t mipCount;
    computeHZBDims(depthSize, hzbSize, mipCount);

    // Both pyramids are declared with their full mip chain. Note this is a
    // stricter contract than most channels: RenderPassReflection::Field::merge
    // rejects a mismatching mip count, so a consumer that connects one of these
    // outputs has to declare the same number of mips.
    reflector.addOutput(kOutputFurthest, "Per-tile max deviceZ (R32Float, mip chain)")
        .format(ResourceFormat::R32Float)
        .bindFlags(ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource)
        .texture2D(hzbSize.x, hzbSize.y, 1, mipCount);

    reflector.addOutput(kOutputClosest, "Per-tile min deviceZ (R32Float, mip chain)")
        .format(ResourceFormat::R32Float)
        .bindFlags(ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource)
        .texture2D(hzbSize.x, hzbSize.y, 1, mipCount);

    return reflector;
}

ref<ComputePass> HierarchicalZPass::getProgram(const ProgramKey& key)
{
    if (auto it = mPrograms.find(key); it != mPrograms.end())
        return it->second;

    ProgramDesc desc;
    desc.addShaderLibrary(kProgramFile).csEntry("csMain");

    DefineList defines;
    defines.add("DIM_MIP_LEVEL_COUNT", std::to_string(key.mipLevelCount));
    defines.add("HZB_PARENT_IS_DEPTH", key.parentIsDepth ? "1" : "0");
    defines.add("HZB_WRITE_FURTHEST", key.writeFurthest ? "1" : "0");
    defines.add("HZB_WRITE_CLOSEST", key.writeClosest ? "1" : "0");

    // No scene modules or type conformances: the pure-depth path never reads
    // camera or scene data, exactly as in UE's VIS_BUFFER_FORMAT == 0 case.
    ref<ComputePass> pPass = ComputePass::create(mpDevice, desc, defines);
    mPrograms.emplace(key, pPass);
    return pPass;
}

void HierarchicalZPass::bindAndDispatch(
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
)
{
    ref<ComputePass> pPass = getProgram(key);
    auto var = pPass->getVars()->getRootVar();

    // UE's ReduceMips parameter setup. DispatchThreadIdToBufferUV converts a
    // destination texel index into the parent UV it covers; the build shader
    // multiplies by 2 because it consumes two parent texels per axis.
    const float2 invSize = 1.0f / float2(srcSize);
    var["PerFrameCB"]["gDispatchThreadIdToBufferUV"] = float4(2.0f * invSize.x, 2.0f * invSize.y, 0.0f, 0.0f);
    var["PerFrameCB"]["gInvSize"] = invSize;
    var["PerFrameCB"]["gParentSize"] = uint2(srcSize);

    // Passed through unchanged, as UE does. The bound is NOT derived from the
    // parent size here: the shader subtracts one texel of its own, and shrinking
    // it a second time would pull the last destination texel's gather onto the
    // previous parent texel. On the closest chain that reads a larger deviceZ
    // than the block actually contains, which is the wrong direction: the
    // closest bound has to stay a lower bound.
    var["PerFrameCB"]["gInputViewportMaxBound"] = inputViewportMaxBound;

    if (key.parentIsDepth)
    {
        // The depth buffer cannot be bound as a UAV, so the first batch reads it
        // as an SRV. This is the only batch where the parent is a different
        // resource from the outputs, which is also why it is the only one that
        // can write both chains in a single dispatch.
        var["gParentTexture"].setSrv(pParent->getSRV(parentMip, 1, 0, 1));
    }
    else
    {
        // Later batches read the previous batch's last mip through a UAV. UE
        // reads it through an SRV, but Falcor tracks resource state per resource
        // rather than per subresource when binding, so an SRV read and a UAV
        // write of one texture would fight over its single state. Reading via
        // UAV keeps the whole resource in the UnorderedAccess state. The shader
        // uses Load, so no sampler is involved and the two forms are equivalent.
        var["gParentTexture"].setUav(pParent->getUAV(parentMip, 0, 1));
    }

    // Bind one UAV per destination mip in this batch. Each is a view of a single
    // mip level, so the writes of a batch never overlap its own parent read.
    for (uint32_t i = 0; i < destMipCount; ++i)
    {
        const uint32_t destMip = startDestMip + i;
        if (key.writeFurthest)
            var["gFurthestHZBOutput"][i].setUav(pFurthestHZB->getUAV(destMip, 0, 1));
        if (key.writeClosest)
            var["gClosestHZBOutput"][i].setUav(pClosestHZB->getUAV(destMip, 0, 1));
    }

    // UE dispatches GetGroupCount(DstSize, 8); ComputePass rounds the thread
    // count up to whole groups itself, so passing the destination size is the
    // same thing.
    pPass->execute(pRenderContext, uint3(dstSize.x, dstSize.y, 1));
}

void HierarchicalZPass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    const ref<Texture> pDepth = renderData.getTexture(kInputDepth);
    const ref<Texture> pFurthest = renderData.getTexture(kOutputFurthest);
    const ref<Texture> pClosest = renderData.getTexture(kOutputClosest);

    mpFurthestHZB = pFurthest;
    mpClosestHZB = pClosest;
    mMipCount = pFurthest ? pFurthest->getMipCount() : 0;

    if (!pDepth || !pFurthest || !pClosest)
        return;

    // Trust the graph's allocation rather than recomputing: reflect() already
    // sized these from the depth buffer, and any disagreement here would mean an
    // alias or an explicitly-sized connection.
    const uint2 hzbSize = uint2(pFurthest->getWidth(0), pFurthest->getHeight(0));
    const uint32_t mipCount = pFurthest->getMipCount();
    if (pClosest->getWidth(0) != hzbSize.x || pClosest->getHeight(0) != hzbSize.y || pClosest->getMipCount() != mipCount)
    {
        logWarning("HierarchicalZPass: the furthest and closest pyramids differ in size or mip count; skipping.");
        return;
    }

    uint2 expectedSize;
    uint32_t expectedMips;
    const uint2 depthSize = uint2(pDepth->getWidth(0), pDepth->getHeight(0));
    computeHZBDims(depthSize, expectedSize, expectedMips);
    if (expectedSize.x != hzbSize.x || expectedSize.y != hzbSize.y || expectedMips != mipCount)
    {
        // Only reachable if the depth buffer changed size without the graph
        // being recompiled, in which case the mip indexing below would run off
        // the end of the chain.
        logWarning(
            "HierarchicalZPass: pyramid is {}x{} ({} mips) but the {}x{} depth buffer implies {}x{} ({} mips); skipping.",
            hzbSize.x,
            hzbSize.y,
            mipCount,
            depthSize.x,
            depthSize.y,
            expectedSize.x,
            expectedSize.y,
            expectedMips);
        return;
    }

    // ---- First batch: depth -> mips [0, min(kMaxMipBatchSize, mipCount)) ----
    //
    // The parent is the depth buffer, so both chains can be produced by one
    // dispatch reading the same 2x2 footprints, exactly as UE does.
    {
        uint2 dstSize = hzbSize;
        const uint32_t destMipCount = std::min(kMaxMipBatchSize, mipCount);
        const uint2 srcSize = uint2(pDepth->getWidth(0), pDepth->getHeight(0));

        // UE: InputViewportMaxBound = (ViewRect.Max - 0.5) / SrcSize. Clamping to
        // the last texel centre keeps the final destination column/row's gather
        // from reading past the viewport when the depth buffer's size is odd.
        const float2 inputViewportMaxBound = float2(
            (float(srcSize.x) - 0.5f) / float(srcSize.x),
            (float(srcSize.y) - 0.5f) / float(srcSize.y));

        ProgramKey key;
        key.parentIsDepth = true;
        key.writeFurthest = true;
        key.writeClosest = true;
        key.mipLevelCount = destMipCount;

        bindAndDispatch(pRenderContext, key, pDepth, 0, srcSize, dstSize, inputViewportMaxBound, pFurthest, pClosest, 0, destMipCount);
    }

    // ---- Remaining batches: previous HZB mip -> the next few mips ----
    //
    // From here on the two chains diverge, because each one reduces its own
    // previous level: they are dispatched separately, as in UE.
    for (uint32_t startDestMip = kMaxMipBatchSize; startDestMip < mipCount; startDestMip += kMaxMipBatchSize)
    {
        const uint2 srcSize = uint2(
            div_round_up(hzbSize.x, 1u << (startDestMip - 1)),
            div_round_up(hzbSize.y, 1u << (startDestMip - 1)));
        const uint2 dstSize = uint2(
            div_round_up(hzbSize.x, 1u << startDestMip),
            div_round_up(hzbSize.y, 1u << startDestMip));
        const uint32_t destMipCount = std::min(kMaxMipBatchSize, mipCount - startDestMip);

        // UE uses exactly (1, 1) for every batch after the first: these parents
        // are pyramids whose edge texels already hold the viewport extreme, so
        // there is nothing to clamp away.
        const float2 inputViewportMaxBound = float2(1.0f, 1.0f);

        for (bool furthestChain : {true, false})
        {
            const ref<Texture> pChain = furthestChain ? pFurthest : pClosest;

            ProgramKey key;
            key.parentIsDepth = false;
            key.writeFurthest = furthestChain;
            key.writeClosest = !furthestChain;
            key.mipLevelCount = destMipCount;

            bindAndDispatch(
                pRenderContext,
                key,
                pChain,
                startDestMip - 1,
                srcSize,
                dstSize,
                inputViewportMaxBound,
                pFurthest,
                pClosest,
                startDestMip,
                destMipCount);
        }
    }
}

void HierarchicalZPass::renderUI(Gui::Widgets& widget)
{
    if (!mpFurthestHZB)
    {
        widget.text("Not executed yet.");
        return;
    }

    widget.text(fmt::format("Mip 0: {}x{}", mpFurthestHZB->getWidth(0), mpFurthestHZB->getHeight(0)));
    widget.text(fmt::format("Mip levels: {}", mMipCount));
}
