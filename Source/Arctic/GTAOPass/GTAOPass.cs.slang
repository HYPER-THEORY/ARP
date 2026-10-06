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

/** Ground-Truth Ambient Occlusion — horizon search + inner integrate, half res.

    Direct port of UnrealEngine's GTAOCombinedPSandCS
    (Private/PostProcessAmbientOcclusion.usf). Runs at half resolution
    reading the full-res GBuffer normal + scene depth.

    Output: gAOMapHalf (R8Unorm), still carrying UE's 2/PI scale — the
    compensating *PI/2 lives in GTAOSpatialFilter.cs.slang, exactly as in UE.
    Do NOT consume this texture directly.

    Notable deviations from UE:
      - No HZB, so we use the non-HZB horizon search (SearchForLargestAngleDual).
        The two variants are numerically equivalent for the same depth source:
        UE's `if (FallOff < 1.0)` early-out is an identity transform when
        FallOff == 1. The real difference is UE's mip escalation, which makes
        each tap a conservative closest-Z over a 2^mip footprint.
      - Half-res pixel h samples the *exact centre* of full-res pixel 2h rather
        than UE's "nudge off the texel boundary by 1/8 texel". Ours is
        unambiguous under point sampling and keeps the reconstructed view
        position consistent with the depth that was fetched.
*/

import Scene.Scene;
import Scene.Camera.Camera;
import Arctic.GBufferPass.GBufferHelpers;

cbuffer PerFrameCB
{
    uint2  gFullFrameDim;      // full-res dims
    uint2  gHalfFrameDim;      // half-res dims
    float  gEffectRadius;      // world-space radius (UE GTAOParams[3].y)
    float  gThicknessBlend;    // UE GTAOParams[1].y == clamp(1 - t*t, 0, 0.99)
    uint   gNumAngles;         // UE GTAOParams[4].y
    float  gSinDeltaAngle;     // UE GTAOParams[4].z == sin(PI / NumAngles)
    float  gCosDeltaAngle;     // UE GTAOParams[4].w == cos(PI / NumAngles)
    float  gTemporalCos;       // UE GTAOParams[0].x
    float  gTemporalSin;       // UE GTAOParams[0].y
    float  gTemporalOffset;    // UE GTAOParams[0].z
    float  gFadeMul;           // UE ScreenSpaceAOParams[4].x == 1 / fadeRadius
    float  gFadeAdd;           // UE ScreenSpaceAOParams[4].y
    float  gMaxDistance;       // UE ScreenSpaceAOParams[4].w — hard early-out
};

Texture2D<float4> gGBufferA;   // full-res: world normal (encoded) + per-object
Texture2D<float>  gSceneDepth; // full-res: device Z (forward-Z: 0=near, 1=far)

RWTexture2D<float> gAOMapHalf;

static const float PI = 3.14159265358979f;
static const float PI_HALF = PI * 0.5f;

// GTAO_NUMTAPS — horizon search samples per direction. 8 == UE's SHADER_QUALITY 2
// ("medium"); UE's high/very-high tiers are 12 and 20.
#define GTAO_NUMTAPS 8
#define GTAO_MAX_PIXEL_SCREEN_RADIUS 256.0f

// ---- Math helpers (ported from UE Common.ush) ----

float acosFast(float x)
{
    float y = abs(x);
    float res = -0.156583f * y + 1.570796f;
    res = res * sqrt(1.0f - y);
    return (x >= 0.0f) ? res : PI - res;
}

float InterleavedGradientNoise(float2 iPos)
{
    return frac(52.9829189f * frac((iPos.x * 0.06711056f) + (iPos.y * 0.00583715f)));
}

// ---- Depth / position reconstruction ----

float LookupDeviceZ(int2 fullPx)
{
    fullPx = clamp(fullPx, int2(0, 0), int2(gFullFrameDim) - 1);
    return gSceneDepth.Load(int3(fullPx, 0)).r;
}

float LookupDeviceZUV(float2 uv)
{
    return LookupDeviceZ(int2(floor(uv * float2(gFullFrameDim))));
}

// Falcor device Z (NDC, forward-Z: 0=near, 1=far) -> linear view depth (positive).
// Falcor builds the projection with math::perspective, which requires a finite
// far plane, so there is no infinite-far case to special-case here.
float ConvertFromDeviceZ(float deviceZ)
{
    float nearZ = gScene.camera.data.nearZ;
    float farZ = gScene.camera.data.farZ;
    return (nearZ * farZ) / max(farZ - deviceZ * (farZ - nearZ), 1e-6f);
}

// UE's View.ScreenToViewSpace: viewXY = (uv * scale + bias) * linearDepth.
// Derivation (Falcor: column-vector convention, RH view space with -Z forward,
// so clip.w == -z_view == linearDepth):
//   clip.x = P00*x + P02*z  and  ndc.x = clip.x / linearDepth
//     => x = (ndc.x + P02) * linearDepth / P00,   ndc.x = uv.x*2 - 1
//   clip.y = P11*y + P12*z
//     => y = (ndc.y + P12) * linearDepth / P11,   ndc.y = 1 - uv.y*2
// P02/P12 are non-zero only under TAA jitter (Camera.cpp applies jitter as a
// clip-space translation), so this stays exact for the jittered depth buffer.
float4 GetScreenToViewSpace()
{
    float P00 = gScene.camera.data.projMat[0][0];
    float P11 = gScene.camera.data.projMat[1][1];
    float P02 = gScene.camera.data.projMat[0][2];
    float P12 = gScene.camera.data.projMat[1][2];
    return float4(2.0f / P00, -2.0f / P11, (P02 - 1.0f) / P00, (1.0f + P12) / P11);
}

float3 ScreenToViewPos(float4 screenToViewSpace, float2 uv, float linearDepth)
{
    float2 p = uv * screenToViewSpace.xy + screenToViewSpace.zw;
    return float3(p * linearDepth, -linearDepth);
}

// View-space normal at an integer full-res pixel.
float3 GetViewSpaceNormal(int2 fullPx)
{
    fullPx = clamp(fullPx, int2(0, 0), int2(gFullFrameDim) - 1);
    float3 worldNormal = decodeNormal(gGBufferA.Load(int3(fullPx, 0)).rgb);
    return normalize(mul((float3x3)gScene.camera.data.viewMat, worldNormal));
}

// ---- Horizon search (UE PostProcessAmbientOcclusion.usf) ----
// Returns (positive-direction horizon, negative-direction horizon) in radians.

float2 SearchForLargestAngleDual(
    uint NumSteps,
    float2 BaseUV,
    float2 ScreenDir,    // in UV units
    float SearchRadius,
    float InitialOffset,
    float3 ViewPos,
    float3 ViewDir,
    float AttenFactor,
    float4 screenToViewSpace)
{
    float2 BestAng = float2(-1.0f, -1.0f);
    float Thickness = gThicknessBlend;

    for (uint i = 0; i < NumSteps; i++)
    {
        float fi = float(i);
        float2 UVOffset = ScreenDir * max(SearchRadius * (fi + InitialOffset), (fi + 1.0f));
        UVOffset.y *= -1.0f;
        float4 UV2 = BaseUV.xyxy + float4(UVOffset.xy, -UVOffset.xy);

        float SceneDepthPos = ConvertFromDeviceZ(LookupDeviceZUV(UV2.xy));
        float SceneDepthNeg = ConvertFromDeviceZ(LookupDeviceZUV(UV2.zw));

        // Positive direction.
        float3 V = ScreenToViewPos(screenToViewSpace, UV2.xy, SceneDepthPos) - ViewPos;
        float LenSq = dot(V, V);
        float OOLen = rsqrt(LenSq + 0.0001f);
        float Ang = dot(V, ViewDir) * OOLen;
        float FallOff = saturate(LenSq * AttenFactor);
        Ang = lerp(Ang, BestAng.x, FallOff);
        BestAng.x = (Ang > BestAng.x) ? Ang : lerp(Ang, BestAng.x, Thickness);

        // Negative direction.
        V = ScreenToViewPos(screenToViewSpace, UV2.zw, SceneDepthNeg) - ViewPos;
        LenSq = dot(V, V);
        OOLen = rsqrt(LenSq + 0.0001f);
        Ang = dot(V, ViewDir) * OOLen;
        FallOff = saturate(LenSq * AttenFactor);
        Ang = lerp(Ang, BestAng.y, FallOff);
        BestAng.y = (Ang > BestAng.y) ? Ang : lerp(Ang, BestAng.y, Thickness);
    }

    BestAng.x = acosFast(clamp(BestAng.x, -1.0f, 1.0f));
    BestAng.y = acosFast(clamp(BestAng.y, -1.0f, 1.0f));
    return BestAng;
}

// ---- Inner integrate (UE PostProcessAmbientOcclusion.usf) ----

float ComputeInnerIntegral(float2 Angles, float2 ScreenDir, float3 ViewDir, float3 ViewSpaceNormal)
{
    // Project the view-space normal onto the plane spanned by the search axis and
    // the view direction, then integrate cosine-weighted visibility between the
    // two horizons. Note PlaneNormal/Perp are handedness-independent: Perp reduces
    // to the normalized projection of ScreenDir onto the plane perpendicular to
    // ViewDir, so Falcor's RH view space needs no sign fixups versus UE's LH one.
    float3 PlaneNormal = normalize(cross(float3(ScreenDir.xy, 0.0f), ViewDir));
    float3 Perp = cross(ViewDir, PlaneNormal);
    float3 ProjNormal = ViewSpaceNormal - PlaneNormal * dot(ViewSpaceNormal, PlaneNormal);

    float LenProjNormal = length(ProjNormal) + 0.000001f;
    float RecipMag = 1.0f / LenProjNormal;

    float CosAng = dot(ProjNormal, Perp) * RecipMag;
    float Gamma = acosFast(CosAng) - PI_HALF;
    float CosGamma = dot(ProjNormal, ViewDir) * RecipMag;
    float SinGamma = CosAng * -2.0f; // == 2*sin(Gamma); the factor 2 is folded in here

    // Clamp both horizons into the normal's hemisphere.
    Angles.x = Gamma + max(-Angles.x - Gamma, -PI_HALF);
    Angles.y = Gamma + min(Angles.y - Gamma, PI_HALF);

    return LenProjNormal * 0.25f * (
        (Angles.x * SinGamma + CosGamma - cos((2.0f * Angles.x) - Gamma)) +
        (Angles.y * SinGamma + CosGamma - cos((2.0f * Angles.y) - Gamma))
    );
}

// ---- Random vector (UE PostProcessAmbientOcclusion.usf) ----

float3 GetRandomVector(uint2 iPos)
{
    iPos.y = 16384 - iPos.y;

    float GradientNoise = InterleavedGradientNoise(float2(iPos));

    float2 RandomTexVec;
    RandomTexVec.x = cos(GradientNoise * PI);
    RandomTexVec.y = sin(GradientNoise * PI);

    float ScaleOffset = (1.0f / 4.0f) * float(((iPos.y - iPos.x) & 3));

    float3 RandomVec;
    RandomVec.x = dot(RandomTexVec.xy, float2(gTemporalCos, -gTemporalSin));
    RandomVec.y = dot(RandomTexVec.xy, float2(gTemporalSin, gTemporalCos));
    RandomVec.z = frac(ScaleOffset + gTemporalOffset);
    return RandomVec;
}

// ---- Main compute ----

[numthreads(8, 8, 1)]
void csMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 halfPixel = dispatchThreadID.xy;
    if (any(halfPixel >= gHalfFrameDim)) return;

    // Half-res pixel h owns the exact centre of full-res pixel 2h. Keeping this
    // one mapping everywhere (here, the spatial/temporal filters and the
    // upsample's bilateral taps) is what makes the chain sub-pixel consistent.
    int2 fullPx = int2(halfPixel) * 2;
    float2 TexUV = (float2(fullPx) + 0.5f) / float2(gFullFrameDim);

    float DeviceZ = LookupDeviceZ(fullPx);
    float SceneDepth = ConvertFromDeviceZ(DeviceZ);

    // Sky / beyond-fade pixels are fully unoccluded. gMaxDistance is
    // kept at/below the point where the distance fade already reaches 1, so this
    // early-out never introduces a visible edge.
    if (SceneDepth > gMaxDistance || DeviceZ >= 1.0f)
    {
        gAOMapHalf[halfPixel] = 1.0f;
        return;
    }

    float4 screenToViewSpace = GetScreenToViewSpace();

    float3 ViewSpacePos = ScreenToViewPos(screenToViewSpace, TexUV, SceneDepth);
    float3 ViewSpaceNormal = GetViewSpaceNormal(fullPx);
    float3 ViewDir = normalize(-ViewSpacePos.xyz);

    // World radius -> full-res pixel radius. A world offset R perpendicular to the
    // view direction at depth z subtends R*P11*H/(2z) pixels vertically, so
    // FOVScale == H*P11/2. (UE uses Extent.y * InvTanHalfFov with a
    // *horizontal* InvTanHalfFov == P00 and carries a `// TODO`; that form is
    // aspect-dependent — 2/aspect off — so we use the exact one.)
    const float WorldRadius = gEffectRadius;
    float FOVScale = float(gFullFrameDim.y) * 0.5f * gScene.camera.data.projMat[1][1];
    float WorldRadiusAdj = WorldRadius * FOVScale;
    float PixelRadius = max(min(WorldRadiusAdj / abs(ViewSpacePos.z), GTAO_MAX_PIXEL_SCREEN_RADIUS), float(GTAO_NUMTAPS));
    float StepRadius = PixelRadius / (float(GTAO_NUMTAPS) + 1.0f);
    float AttenFactor = 2.0f / (WorldRadius * WorldRadius);

    float3 RandomAndOffset = GetRandomVector(halfPixel);
    float2 ScreenDir = RandomAndOffset.xy;
    float Offset = RandomAndOffset.z;

    float2 invFullRes = 1.0f / float2(gFullFrameDim);
    float Sum = 0.0f;

    [loop]
    for (uint Angle = 0; Angle < gNumAngles; Angle++)
    {
        float2 Angles = SearchForLargestAngleDual(
            GTAO_NUMTAPS, TexUV, ScreenDir * invFullRes, StepRadius, Offset,
            ViewSpacePos, ViewDir, AttenFactor, screenToViewSpace);

        Sum += ComputeInnerIntegral(Angles, ScreenDir, ViewDir, ViewSpaceNormal);

        // Rotate the slice and decorrelate the step offset.
        float2 TempScreenDir = ScreenDir.xy;
        ScreenDir.x = (TempScreenDir.x * gCosDeltaAngle) + (TempScreenDir.y * -gSinDeltaAngle);
        ScreenDir.y = (TempScreenDir.x * gSinDeltaAngle) + (TempScreenDir.y * gCosDeltaAngle);
        Offset = frac(Offset + 0.617f);
    }

    float AO = Sum / float(gNumAngles);
    // UE. Undone by *PI/2 in the spatial filter — the pair cancels, so
    // a fully unoccluded surface ends up at exactly 1.0 after that pass.
    AO *= 2.0f / PI;

    // Smooth distance fade, replacing a hard cutoff.
    AO = lerp(AO, 1.0f, saturate(SceneDepth * gFadeMul + gFadeAdd));

    gAOMapHalf[halfPixel] = AO;
}
