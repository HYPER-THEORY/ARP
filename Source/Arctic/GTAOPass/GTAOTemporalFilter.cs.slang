/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the copyright
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

/** GTAO temporal filter — half res, reprojected.

    Direct port of UnrealEngine's GTAOTemporalFilterPSAndCS
    (Private/PostProcessAmbientOcclusion.usf), including
    ReadHistoryClamp and CompareVeloc.

    What UE actually ships here, contrary to what the file's
    GTAO_VARIANCE_CLIPPING define suggests: NeighbourhoodClamp is
    *never called*. The history is bounded by NewAO +/- RangeVal, where RangeVal
    shrinks to 0 as screen velocity grows — under motion the filter trusts the
    new sample almost completely, which is what keeps it from smearing. We port
    the shipped path, not the dead one.

    Two history samples are blended by CompareVeloc: the reprojected one at
    PrevUV, and the non-reprojected one at UV. Where this pixel's velocity
    disagrees with the velocity found at its own reprojection target (a
    disocclusion, or a reprojection that landed on a differently-moving surface)
    the filter falls back to the stationary sample rather than to garbage.

    Deviations from UE:
      - Reprojection comes from MotionVectorPass instead of UE's
        ClipToPrevClip + optional velocity texture; this just avoids duplicating it.
      - Motion is fetched at full-res pixel 2h (the pixel the horizon search
        actually sampled). That is a quarter-of-a-half-texel offset from the
        half-res texel centre used for the UV math — unavoidable at half res.
      - History sampled outside the frame falls back to no accumulation. UE
        leans on its clamped sampler plus the +/-RangeVal bound instead.
*/

// No Scene import: reprojection comes entirely from the motion vector texture.

cbuffer PerFrameCB
{
    uint2 gFullFrameDim;
    uint2 gHalfFrameDim;
    float gBlendWeight;   // UE GTAOParams[4].x — weight of the new frame
    uint  gHasHistory;    // 0 on the first frame after (re)allocation
};

Texture2D<float>  gAOFilteredHalf;  // this frame's spatially filtered AO
Texture2D<float>  gHistoryAO;       // last frame's output of this pass
Texture2D<float2> gMotionVecs;      // full-res, current->previous UV offset

RWTexture2D<float> gAOTemporalHalf;

int2 ClampHalf(int2 px)
{
    return clamp(px, int2(0, 0), int2(gHalfFrameDim) - 1);
}

// Motion vector for half-res pixel h, taken at the full-res pixel it owns.
float2 LoadMotion(int2 halfPx)
{
    int2 fullPx = clamp(ClampHalf(halfPx) * 2, int2(0, 0), int2(gFullFrameDim) - 1);
    return gMotionVecs.Load(int3(fullPx, 0)).xy;
}

// Half-res sample position of texel h, in full-res pixel-centre UV space.
// Texel h holds the AO of full-res pixel 2h (see GTAOPass.cs.slang).
float2 HalfTexelPosUV(int2 halfPx)
{
    return (float2(halfPx) * 2.0f + 0.5f) / float2(gFullFrameDim);
}

// Inverse of the above: full-res UV -> fractional half-res texel coordinate,
// with texel centres at integers. HalfTexelCoord(HalfTexelPosUV(h)) == h.
float2 HalfTexelCoord(float2 fullUV)
{
    return fullUV * float2(gFullFrameDim) * 0.5f - 0.25f;
}

// UE ReadHistoryClamp: manual bilinear where every tap is clamped to
// the acceptable range *before* being weighted, so a single bad tap cannot drag
// the result outside the range the way clamping after the filter would.
float ReadHistoryClamp(float2 t, float MinAO, float MaxAO)
{
    float2 FloorUV = floor(t);
    float2 FracUV = t - FloorUV;

    float BilinearWeights[4];
    BilinearWeights[0] = (1.0f - FracUV.x) * (1.0f - FracUV.y);
    BilinearWeights[1] = (FracUV.x) * (1.0f - FracUV.y);
    BilinearWeights[2] = (1.0f - FracUV.x) * (FracUV.y);
    BilinearWeights[3] = (FracUV.x) * (FracUV.y);

    int2 base = int2(FloorUV);
    int2 offs[4] = { int2(0, 0), int2(1, 0), int2(0, 1), int2(1, 1) };

    float VisHistory = 0.0f;
    [unroll]
    for (int i = 0; i < 4; i++)
    {
        float h = gHistoryAO.Load(int3(ClampHalf(base + offs[i]), 0)).r;
        VisHistory += BilinearWeights[i] * clamp(h, MinAO, MaxAO);
    }
    return VisHistory;
}

// UE CompareVeloc. 1 == velocities agree, 0 == they do not.
float CompareVeloc(float2 V1, float2 V2)
{
    float2 V12 = V1 - V2;
    return 1.0f - saturate(abs(V12.x + V12.y) * 100.0f);
}

[numthreads(8, 8, 1)]
void csMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 halfPixel = dispatchThreadID.xy;
    if (any(halfPixel >= gHalfFrameDim)) return;

    float NewAO = gAOFilteredHalf.Load(int3(halfPixel, 0)).r;

    if (gHasHistory == 0u)
    {
        gAOTemporalHalf[halfPixel] = NewAO;
        return;
    }

    // Reproject in full-res UV space (motion vectors live there), then convert to
    // half-res texel coordinates for the history fetch.
    float2 curUV = HalfTexelPosUV(int2(halfPixel));
    float2 motion = LoadMotion(int2(halfPixel));
    float2 PrevUV = curUV + motion;

    if (any(PrevUV < 0.0f) || any(PrevUV > 1.0f))
    {
        gAOTemporalHalf[halfPixel] = NewAO;
        return;
    }

    float2 prevT = HalfTexelCoord(PrevUV);

    // UE keeps velocity as UV - PrevUV; our motion vector is PrevUV - UV. The
    // sign is global to both operands so CompareVeloc is unaffected.
    float2 PixVelocity = -motion;
    float VelocityMag = saturate(length(PixVelocity) * 100.0f);

    // Velocity at this pixel's own reprojection target.
    int2 prevHalfPx = ClampHalf(int2(floor(prevT + 0.5f)));
    float2 DestVeloc = -LoadMotion(prevHalfPx);

    float VelocCompare = CompareVeloc(PixVelocity, DestVeloc);

    // Acceptable band around the new sample; collapses to 0 under motion.
    float RangeVal = lerp(0.1f, 0.0f, VelocityMag);
    float MinAO = saturate(NewAO - RangeVal);
    float MaxAO = saturate(NewAO + RangeVal);

    float HistoryPrevUV = ReadHistoryClamp(prevT, MinAO, MaxAO);
    float HistoryThisUV = clamp(gHistoryAO.Load(int3(halfPixel, 0)).r, MinAO, MaxAO);

    float HistoryAO = lerp(HistoryThisUV, HistoryPrevUV, VelocCompare);

    gAOTemporalHalf[halfPixel] = saturate(lerp(HistoryAO, NewAO, gBlendWeight));
}
