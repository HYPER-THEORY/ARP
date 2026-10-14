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
    HZB-accelerated version matching UE's GTAOCombinedCS / SearchForLargestAngleDual_HZB.

    Direct port of UnrealEngine's GTAOCombinedPSandCS
    (Private/PostProcessAmbientOcclusion.usf). Runs at half resolution
    reading the full-res GBuffer normal + scene depth.

    HZB mip staircase (UE SearchForLargestAngleDual_HZB, GTAO_BIASMIPLEVEL per quality).
    UE's loop runs i=1..NumSteps (1-based); the i==0 branch in UE is dead code:
      tap i=1  -> mip GTAO_BIASMIPLEVEL
      tap i=2  -> mip GTAO_BIASMIPLEVEL + 1
      tap i=3  -> mip GTAO_BIASMIPLEVEL        (resets to base, not incremented)
      tap i>=4 -> mip GTAO_BIASMIPLEVEL + 2
    GTAO_BIASMIPLEVEL: Q0=2, Q1=1, Q2/3/4=0.

    Output: gAOMapHalf (R8Unorm), still carrying UE's 2/PI scale.
*/

#include "Utils/Math/MathConstants.slangh"

import Scene.Scene;
import Scene.Camera.Camera;
import Arctic.GBufferPass.GBufferHelpers;

cbuffer PerFrameCB
{
    uint2  gFullFrameDim;
    uint2  gHalfFrameDim;
    float  gEffectRadius;
    float  gThicknessBlend;
    uint   gNumAngles;
    float  gSinDeltaAngle;
    float  gCosDeltaAngle;
    float  gTemporalCos;
    float  gTemporalSin;
    float  gTemporalOffset;
    float  gFadeMul;
    float  gFadeAdd;        // gFadeMul/gFadeAdd: lerp(AO,1,saturate(depth*fadeMul+fadeAdd))
    float  gMaxDistance;    // early-out: pixels beyond this depth write AO=1 (UE ScreenSpaceAOParams[4].w)
    float2 gHzbUVFactor;    // screenUV * gHzbUVFactor = hzbUV (mip-0 coords)
};

Texture2D<float4> gGBufferA;
Texture2D<float>  gSceneDepth;
Texture2D<float>  gZFurthest;    // HZB furthest mip chain (R32Float)
SamplerState      gHZBSampler;   // point clamp

RWTexture2D<float> gAOMapHalf;

#define GTAO_NUMTAPS                 20
#define GTAO_BIASMIPLEVEL            0
#define GTAO_MAX_PIXEL_SCREEN_RADIUS 256.0

// ---- Math helpers ----

float acosFast(float x)
{
    float y = abs(x);
    float res = -0.156583f * y + 1.570796f;
    res = res * sqrt(1.0f - y);
    return (x >= 0.0f) ? res : M_PI - res;
}

float InterleavedGradientNoise(float2 iPos)
{
    return frac(52.9829189f * frac((iPos.x * 0.06711056f) + (iPos.y * 0.00583715f)));
}

// ---- Depth helpers ----

float LookupDeviceZ(int2 fullPx)
{
    fullPx = clamp(fullPx, int2(0, 0), int2(gFullFrameDim) - 1);
    return gSceneDepth.Load(int3(fullPx, 0)).r;
}

float LookupDeviceZUV(float2 uv)
{
    return LookupDeviceZ(int2(floor(uv * float2(gFullFrameDim))));
}

float ConvertFromDeviceZ(float deviceZ)
{
    float nearZ = gScene.camera.data.nearZ;
    float farZ = gScene.camera.data.farZ;
    return (nearZ * farZ) / (farZ - deviceZ * (farZ - nearZ));
}

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

float3 GetViewSpaceNormal(int2 fullPx)
{
    fullPx = clamp(fullPx, int2(0, 0), int2(gFullFrameDim) - 1);
    float3 worldNormal = decodeNormal(gGBufferA.Load(int3(fullPx, 0)).rgb);
    return normalize(mul((float3x3)gScene.camera.data.viewMat, worldNormal));
}

// ---- HZB depth fetch with fixed mip staircase (UE SearchForLargestAngleDual_HZB) ----
// tap index 'i' (1-based as in UE's loop: for i in 1..NumSteps+1).
float GetSceneDepthForTap(float2 uv, uint tapIdx)
{
    // UE SearchForLargestAngleDual_HZB mip staircase (loop runs i=1..NumSteps, 1-based).
    // The i==0 branch in UE's source is dead code: the loop starts at i=1, so that
    // branch never executes. The live logic is:
    //   i=1  -> mip BIASMIPLEVEL
    //   i=2  -> mip BIASMIPLEVEL + 1  (explicit i==2 branch increments)
    //   i=3  -> mip BIASMIPLEVEL      (falls through to the base; no i==3 branch)
    //   i>=4 -> mip BIASMIPLEVEL + 2  (i>3 branch adds 2)
    uint mipLevel;
    if (tapIdx == 1u)
        mipLevel = uint(GTAO_BIASMIPLEVEL);
    else if (tapIdx == 2u)
        mipLevel = uint(GTAO_BIASMIPLEVEL) + 1u;
    else if (tapIdx == 3u)
        mipLevel = uint(GTAO_BIASMIPLEVEL);
    else
        mipLevel = uint(GTAO_BIASMIPLEVEL) + 2u;

    float2 hzbUV  = uv * gHzbUVFactor;
    float deviceZ = gZFurthest.SampleLevel(gHZBSampler, hzbUV, float(mipLevel)).r;
    return ConvertFromDeviceZ(deviceZ);
}

// ---- Horizon search with HZB (UE SearchForLargestAngleDual_HZB) ----
float2 SearchForLargestAngleDual_HZB(
    uint NumSteps,
    float2 BaseUV,
    float2 ScreenDir,
    float SearchRadius,
    float InitialOffset,
    float3 ViewPos,
    float3 ViewDir,
    float AttenFactor,
    float4 screenToViewSpace)
{
    float2 BestAng = float2(-1.0f, -1.0f);
    float Thickness = gThicknessBlend;

    for (uint i = 1u; i <= NumSteps; i++)
    {
        float fi = float(i);
        float2 UVOffset = ScreenDir * max(SearchRadius * (fi + InitialOffset), fi + 1.0f);
        UVOffset.y *= -1.0f;
        float4 UV2 = BaseUV.xyxy + float4(UVOffset.xy, -UVOffset.xy);

        float SceneDepthPos = GetSceneDepthForTap(UV2.xy, i);
        float SceneDepthNeg = GetSceneDepthForTap(UV2.zw, i);

        // Positive direction.
        float3 V = ScreenToViewPos(screenToViewSpace, UV2.xy, SceneDepthPos) - ViewPos;
        float LenSq = dot(V, V);
        float OOLen = rsqrt(LenSq + 0.0001f);
        float FallOff = saturate(LenSq * AttenFactor);
        if (FallOff < 1.0f)
        {
            float Ang = dot(V, ViewDir) * OOLen;
            Ang = lerp(Ang, BestAng.x, FallOff);
            BestAng.x = (Ang > BestAng.x) ? Ang : lerp(Ang, BestAng.x, Thickness);
        }

        // Negative direction.
        V = ScreenToViewPos(screenToViewSpace, UV2.zw, SceneDepthNeg) - ViewPos;
        LenSq = dot(V, V);
        OOLen = rsqrt(LenSq + 0.0001f);
        FallOff = saturate(LenSq * AttenFactor);
        if (FallOff < 1.0f)
        {
            float Ang = dot(V, ViewDir) * OOLen;
            Ang = lerp(Ang, BestAng.y, FallOff);
            BestAng.y = (Ang > BestAng.y) ? Ang : lerp(Ang, BestAng.y, Thickness);
        }
    }

    BestAng.x = acosFast(clamp(BestAng.x, -1.0f, 1.0f));
    BestAng.y = acosFast(clamp(BestAng.y, -1.0f, 1.0f));
    return BestAng;
}

// ---- Inner integrate ----

float ComputeInnerIntegral(float2 Angles, float2 ScreenDir, float3 ViewDir, float3 ViewSpaceNormal)
{
    float3 PlaneNormal = normalize(cross(float3(ScreenDir.xy, 0.0f), ViewDir));
    float3 Perp = cross(ViewDir, PlaneNormal);
    float3 ProjNormal = ViewSpaceNormal - PlaneNormal * dot(ViewSpaceNormal, PlaneNormal);
    float LenProjNormal = length(ProjNormal) + 0.000001f;
    float RecipMag = 1.0f / LenProjNormal;
    float CosAng = dot(ProjNormal, Perp) * RecipMag;
    float Gamma = acosFast(CosAng) - M_PI_2;
    float CosGamma = dot(ProjNormal, ViewDir) * RecipMag;
    float SinGamma = CosAng * -2.0f;

    Angles.x = Gamma + max(-Angles.x - Gamma, -M_PI_2);
    Angles.y = Gamma + min( Angles.y - Gamma,  M_PI_2);

    return LenProjNormal * 0.25f * (
        (Angles.x * SinGamma + CosGamma - cos(2.0f * Angles.x - Gamma)) +
        (Angles.y * SinGamma + CosGamma - cos(2.0f * Angles.y - Gamma)));
}

// ---- Random vector ----

float3 GetRandomVector(uint2 iPos)
{
    iPos.y = 16384 - iPos.y;
    float GradientNoise = InterleavedGradientNoise(float2(iPos));
    float2 RandomTexVec;
    RandomTexVec.x = cos(GradientNoise * M_PI);
    RandomTexVec.y = sin(GradientNoise * M_PI);
    float ScaleOffset = (1.0f / 4.0f) * float(((iPos.y - iPos.x) & 3));
    float3 RandomVec;
    RandomVec.x = dot(RandomTexVec.xy, float2(gTemporalCos, -gTemporalSin));
    RandomVec.y = dot(RandomTexVec.xy, float2(gTemporalSin,  gTemporalCos));
    RandomVec.z = frac(ScaleOffset + gTemporalOffset);
    return RandomVec;
}

// ---- Main ----

[numthreads(8, 8, 1)]
void csMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 halfPixel = dispatchThreadID.xy;
    if (any(halfPixel >= gHalfFrameDim)) return;

    int2 fullPx = int2(halfPixel) * 2;
    float2 TexUV = (float2(fullPx) + 0.5f) / float2(gFullFrameDim);

    float DeviceZ = LookupDeviceZ(fullPx);
    float SceneDepth = ConvertFromDeviceZ(DeviceZ);

    if (SceneDepth > gMaxDistance || DeviceZ >= 1.0f)
    {
        gAOMapHalf[halfPixel] = 1.0f;
        return;
    }

    float4 screenToViewSpace = GetScreenToViewSpace();

    float3 ViewSpacePos = ScreenToViewPos(screenToViewSpace, TexUV, SceneDepth);
    float3 ViewSpaceNormal = GetViewSpaceNormal(fullPx);
    float3 ViewDir = normalize(-ViewSpacePos.xyz);

    float FOVScale = float(gFullFrameDim.y) * 0.5f * gScene.camera.data.projMat[1][1];
    float WorldRadiusAdj = gEffectRadius * FOVScale;
    float PixelRadius = max(min(WorldRadiusAdj / abs(ViewSpacePos.z), GTAO_MAX_PIXEL_SCREEN_RADIUS), float(GTAO_NUMTAPS));
    float StepRadius = PixelRadius / (float(GTAO_NUMTAPS) + 1.0f);
    float AttenFactor = 2.0f / (gEffectRadius * gEffectRadius);

    float3 RandomAndOffset = GetRandomVector(halfPixel);
    float2 ScreenDir = RandomAndOffset.xy;
    float Offset = RandomAndOffset.z;

    float2 invFullRes = 1.0f / float2(gFullFrameDim);
    float Sum = 0.0f;

    [loop]
    for (uint Angle = 0; Angle < gNumAngles; Angle++)
    {
        float2 Angles = SearchForLargestAngleDual_HZB(
            GTAO_NUMTAPS, TexUV, ScreenDir * invFullRes, StepRadius, Offset,
            ViewSpacePos, ViewDir, AttenFactor, screenToViewSpace);

        Sum += ComputeInnerIntegral(Angles, ScreenDir, ViewDir, ViewSpaceNormal);

        float2 TempScreenDir = ScreenDir;
        ScreenDir.x = TempScreenDir.x * gCosDeltaAngle + TempScreenDir.y * (-gSinDeltaAngle);
        ScreenDir.y = TempScreenDir.x * gSinDeltaAngle + TempScreenDir.y * gCosDeltaAngle;
        Offset = frac(Offset + 0.617f);
    }

    float AO = Sum / float(gNumAngles);
    AO *= 2.0f / M_PI;
    AO = lerp(AO, 1.0f, saturate(SceneDepth * gFadeMul + gFadeAdd));

    gAOMapHalf[halfPixel] = AO;
}
