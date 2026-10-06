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

/** GTAO 5x5 bilateral spatial filter — half res.

    Direct port of UnrealEngine's GTAOSpatialFilterCS
    (Private/PostProcessAmbientOcclusion.usf).

    A plain box blur would bleed AO across depth discontinuities, so each tap is
    weighted against a *plane* fitted through the centre pixel using per-axis
    depth gradients extrapolated from the two neighbours on whichever side is
    more consistent with the centre. Flat-but-slanted surfaces therefore keep
    full weight while silhouettes are cut.

    This is also where UE applies the compensating *PI/2 for the 2/PI in the
    horizon-search pass and the user-facing AO power/intensity,
    so this pass — not the search — is the first one whose
    output is a directly usable AO value.

    A note on the bilateral weight operating on *device* Z with UE's magic 20000:
    that constant transfers to Falcor unchanged even though UE uses reversed-Z
    with an infinite far plane and Falcor uses forward-Z with a finite one. Both
    give |d(deviceZ)/dz| = k*nearZ/z^2 (k = farZ/(farZ-nearZ) ~ 1 for Falcor,
    exactly 1 for UE), so the implied world-space tolerance scales identically
    with distance, and it is expressed relative to nearZ — i.e. it is invariant
    to the cm-vs-m unit difference as well. Exposed as a parameter anyway.
*/

// No Scene import: this filter works entirely on raw device Z, so it never needs
// the camera. (UE computes a ThisZLin here and then never uses it either.)

cbuffer PerFrameCB
{
    uint2 gFullFrameDim;
    uint2 gHalfFrameDim;
    int   gDownsampleFactor;      // full-res pixels per AO pixel (2 at half res)
    int   gFilterMin;             // UE GTAOSpatialFilterWidth.x (-2)
    int   gFilterMax;             // UE GTAOSpatialFilterWidth.y (+2)
    float gSpatialFilterWeight;   // UE's hardcoded 20000
    float gPower;                 // UE ScreenSpaceAOParams[0].x
    float gIntensity;             // UE ScreenSpaceAOParams[0].w
};

Texture2D<float> gAOMapHalf;    // horizon-search output (still 2/PI-scaled)
Texture2D<float> gSceneDepth;   // full-res device Z

RWTexture2D<float> gAOFilteredHalf;

static const float PI = 3.14159265358979f;

// The 5x5 filter works on a 16x8 threadgroup (128 pixels) and needs a 2-pixel
// border, so 20x12 = 240 values are staged in LDS; each thread loads 2.
#define LDS_WIDTH 20
#define LDS_HEIGHT 12

groupshared float AOData[LDS_WIDTH * LDS_HEIGHT];
groupshared float ZData[LDS_WIDTH * LDS_HEIGHT];

int GetLDSLocation(int x, int y)
{
    x += 2; y += 2;
    return (y * LDS_WIDTH) + x;
}

[numthreads(16, 8, 1)]
void csMain(
    int GroupIndex : SV_GroupIndex,
    uint2 GroupId : SV_GroupID,
    uint2 DispatchThreadId : SV_DispatchThreadID,
    uint2 GroupThreadId : SV_GroupThreadID)
{
    int2 GTId = int2(GroupThreadId);
    int2 PixelPos = int2(DispatchThreadId);

    int2 FullGroupOrigin = int2(GroupId.x * 16, GroupId.y * 8);
    int2 FullGroupOriginM2 = FullGroupOrigin - int2(2, 2);

    // ---- Stage the 20x12 neighbourhood (AO at half res, Z at full res) ----
    // pixIdx is always even and LDS_WIDTH is even, so XPos <= 18 and the second
    // write never wraps to the next LDS row.
    int pixIdx = GroupIndex * 2;
    if (pixIdx < LDS_WIDTH * LDS_HEIGHT)
    {
        int XPos = pixIdx % LDS_WIDTH;
        int YPos = pixIdx / LDS_WIDTH;
        int LDSPos = (YPos * LDS_WIDTH) + XPos;

        [unroll]
        for (int k = 0; k < 2; ++k)
        {
            int2 ReadXYAO = FullGroupOriginM2 + int2(XPos + k, YPos);
            // Clamp instead of relying on out-of-bounds Load returning 0, which
            // would feed AO=0 (fully occluded) into the screen-border taps.
            ReadXYAO = clamp(ReadXYAO, int2(0, 0), int2(gHalfFrameDim) - 1);
            int2 ReadXYZ = clamp(ReadXYAO * gDownsampleFactor, int2(0, 0), int2(gFullFrameDim) - 1);

            AOData[LDSPos + k] = gAOMapHalf.Load(int3(ReadXYAO, 0)).r;
            ZData[LDSPos + k] = gSceneDepth.Load(int3(ReadXYZ, 0)).r;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (any(uint2(PixelPos) >= gHalfFrameDim)) return;

    float ThisZ = ZData[GetLDSLocation(GTId.x, GTId.y)];
    int LDSCentre = GetLDSLocation(GTId.x, GTId.y);

    // ---- Per-axis depth gradient, taken from the more consistent side ----
    float2 ZDiff;
    {
        float XM2Z = ZData[LDSCentre - 2];
        float XM1Z = ZData[LDSCentre - 1];
        float XP1Z = ZData[LDSCentre + 1];
        float XP2Z = ZData[LDSCentre + 2];

        // Extrapolate one step inward from each side and keep whichever lands
        // closer to the centre depth — i.e. the side we are probably coplanar with.
        float C1 = abs((XM1Z + (XM1Z - XM2Z)) - ThisZ);
        float C2 = abs((XP1Z + (XP1Z - XP2Z)) - ThisZ);
        ZDiff.x = (C1 < C2) ? (XM1Z - XM2Z) : (XP2Z - XP1Z);
    }
    {
        float YM2Z = ZData[LDSCentre - (2 * LDS_WIDTH)];
        float YM1Z = ZData[LDSCentre - (1 * LDS_WIDTH)];
        float YP1Z = ZData[LDSCentre + (1 * LDS_WIDTH)];
        float YP2Z = ZData[LDSCentre + (2 * LDS_WIDTH)];

        float C1 = abs((YM1Z + (YM1Z - YM2Z)) - ThisZ);
        float C2 = abs((YP1Z + (YP1Z - YP2Z)) - ThisZ);
        ZDiff.y = (C1 < C2) ? (YM1Z - YM2Z) : (YP2Z - YP1Z);
    }

    // ---- Blur, weighting each tap against the fitted plane ----
    float SumAO = 0.0f;
    float SumWeight = 0.0f;

    int LDSBase = GetLDSLocation(GTId.x + gFilterMin, GTId.y + gFilterMin);
    float DepthBase = ThisZ + (ZDiff.x * gFilterMin) + (ZDiff.y * gFilterMin);

    for (int y = gFilterMin; y <= gFilterMax; y++)
    {
        float PlaneZ = DepthBase;
        int LDSLineBase = LDSBase;
        LDSBase += LDS_WIDTH;

        for (int x = gFilterMin; x <= gFilterMax; x++)
        {
            float Sample_AO = AOData[LDSLineBase];
            float SampleZ = ZData[LDSLineBase];
            LDSLineBase++;

            float SampleZDiff = abs(PlaneZ - SampleZ);
            float Weight = 1.0f - saturate(SampleZDiff * gSpatialFilterWeight);

            SumAO += Sample_AO * Weight;
            SumWeight += Weight;

            PlaneZ += ZDiff.x;
        }
        DepthBase += ZDiff.y;
    }

    SumAO /= max(SumWeight, 1e-6f);

    // Undo the 2/PI applied by the horizon search. Paired with
    // that factor this is exactly 1.0 for a fully unoccluded surface.
    SumAO *= (PI / 2.0f);

    // User adjust: intensity == 0 disables AO entirely,
    // power sharpens it. Note UE halves the power exponent.
    SumAO = 1.0f - (1.0f - pow(abs(SumAO), gPower * 0.5f)) * gIntensity;

    gAOFilteredHalf[uint2(PixelPos)] = saturate(SumAO);
}
