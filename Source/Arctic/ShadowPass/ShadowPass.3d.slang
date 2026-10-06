/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
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
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/

#include "Scene/VertexAttrib.slangh"

import Scene.Raster;
import Utils.Math.MathHelpers;

cbuffer PerFrameCB
{
    float4x4 gLightViewProj;
    float3   gLightPos;
    float3   gLightDir;
    float    gMaxSubjectDepth;
    float    gConstantDepthBias;
    float    gSlopeDepthBias;
    float    gMaxSlopeDepthBias;
};

struct ShadowVSOut
{
    float4 posH : SV_POSITION;
    nointerpolation GeometryInstanceID instanceID : INSTANCE_ID;
    nointerpolation uint materialID : MATERIAL_ID;
    float3 faceNormalW : FACENORMAL;
    float2 texC : TEXCOORD;
    // Light-space linear depth = dot(gLightDir, posW - gLightPos), in METERS.
    // Linearly interpolating this across the triangle is correct because the
    // depth is a linear function of posW (ortho projection). This matches
    // UE's ScreenPosition.z semantics for the non-perspective-correct path.
    float  lightDepth : LIGHTDEPTH;
};

ShadowVSOut vsMain(VSIn vIn)
{
    ShadowVSOut vOut;
    const GeometryInstanceID instanceID = { vIn.instanceID };
    float4x4 worldMat = gScene.getWorldMatrix(instanceID);
    float3 posW = mul(worldMat, float4(vIn.unpack().position, 1.f)).xyz;
    vOut.posH = mul(gLightViewProj, float4(posW, 1.f));
    vOut.instanceID = instanceID;
    vOut.materialID = gScene.getMaterialID(instanceID);
    vOut.faceNormalW = mul((float3x3)gScene.getInverseTransposeWorldMatrix(instanceID), vIn.unpack().normal);
    vOut.texC = vIn.unpack().texCrd;
    vOut.lightDepth = dot(gLightDir, posW - gLightPos);
    return vOut;
}

// Note: [earlydepthstencil] is intentionally NOT used. With early
// depth-stencil, depth is written before the pixel shader runs, so alpha-tested
// (discarded) pixels would still write depth. Downstream passes would then
// sample shadowmap depth where the pixel was actually alpha-clipped, producing
// spurious shadow rectangles on masked geometry.
float psMain(ShadowVSOut vsOut, uint triangleIndex : SV_PrimitiveID) : SV_Target0
{
#if USE_ALPHA_TEST
    VertexData v;
    v.posW = float3(0.f);
    v.faceNormalW = vsOut.faceNormalW;
    v.normalW = vsOut.faceNormalW;
    v.tangentW = float4(0.f);
    v.texC = vsOut.texC;
    v.curveRadius = 0.f;
    v.coneTexLODValue = 0.f;
    let lod = ImplicitLodTextureSampler();
    if (gScene.materials.alphaTest(v, vsOut.materialID, lod))
        discard;
#endif

    // UE SetShadowDepthOutputs (slope-bias plane, ShadowDepthVertexShader.usf):
    //   DepthBias = SlopeDepthBias * slope + ConstantDepthBias
    //   slope = sqrt(1 - NoL^2) / NoL, clamped to MaxSlopeDepthBias.
    // We write the light-space linear depth (meters, dot(lightDir, posW -
    // lightPos)) plus a positive bias that pushes written depth slightly away
    // from the light to suppress self-shadowing acne.
    //
    // UNIT NOTE: UE's bias parameters are DIMENSIONLESS. Its directional
    // shadowmap stores `1 - z_clip * InvMaxSubjectDepth`, i.e. depth normalized
    // over the cascade's range, so UE's numeric defaults do NOT transfer to
    // this pass. Only the formula is ported; our values are in METERS, which
    // keeps them meaningful when the fitted frustum changes size.
    //
    // Like UE, the slope is computed from the interpolated VERTEX normal (UE
    // uses VertexFactoryGetWorldNormal), not a true geometric face normal. For
    // foliage cards whose vertex normals are authored to bulge outward rather
    // than follow the card, NoL is only loosely related to the real incidence
    // angle — expect the slope term to be approximate on such geometry.
    float3 faceNormal = normalize(vsOut.faceNormalW);
    float NoL = abs(dot(gLightDir, faceNormal));
    float slope = (NoL > 0.f)
        ? clamp(sqrt(saturate(1.f - NoL * NoL)) / NoL, 0.f, gMaxSlopeDepthBias)
        : gMaxSlopeDepthBias;
    float depthBias = gSlopeDepthBias * slope + gConstantDepthBias;

    return vsOut.lightDepth + depthBias;
}
