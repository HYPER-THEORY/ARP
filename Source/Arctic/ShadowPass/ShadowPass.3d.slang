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
    float2 texC : TEXCOORD;
    // World position, needed by the alpha test path: evalAlphaTest hashes it for
    // hashed-alpha materials (see AlphaTest.slang). UE interpolates the world
    // position for masked shadows too (INTERPOLATE_POSITION in
    // ShadowDepthCommon.ush).
    float3 posW : POSW;
    // Light-space linear depth in METERS, INCLUDING the depth bias.
    //
    // The bias is applied here rather than in the pixel shader to match UE, which
    // computes it in SetShadowDepthOutputs (ShadowDepthVertexShader.usf) and lets
    // the rasterizer interpolate the result. Depth is a linear function of posW
    // under an orthographic projection, so interpolating the biased value is
    // exact for planar geometry and differs only slightly from a per-pixel
    // evaluation on curved surfaces.
    float  lightDepth : LIGHTDEPTH;
};

ShadowVSOut vsMain(VSIn vIn)
{
    ShadowVSOut vOut;
    const GeometryInstanceID instanceID = { vIn.instanceID };
    float4x4 worldMat = gScene.getWorldMatrix(instanceID);
    float3 posW = mul(worldMat, float4(vIn.unpack().position, 1.f)).xyz;
    float3 normalW = mul((float3x3)gScene.getInverseTransposeWorldMatrix(instanceID), vIn.unpack().normal);

    const float depth = dot(gLightDir, posW - gLightPos);

    // UE SetShadowDepthOutputs (slope-bias plane, ShadowDepthVertexShader.usf):
    //   DepthBias = SlopeDepthBias * slope + ConstantDepthBias
    //   slope = sqrt(1 - NoL^2) / NoL, clamped to MaxSlopeDepthBias.
    //
    // UNIT NOTE: UE's bias parameters are DIMENSIONLESS. Its directional
    // shadowmap stores `1 - z_clip * InvMaxSubjectDepth`, i.e. depth normalized
    // over the cascade's range, so UE's numeric defaults do NOT transfer to
    // this pass. Only the formula is ported; our values are in METERS, which
    // keeps them meaningful when the fitted frustum changes size.
    //
    // Like UE, the slope is computed from the VERTEX normal
    // (VertexFactoryGetWorldNormal in UE), not a true geometric face normal. For
    // foliage cards whose vertex normals are authored to bulge outward rather
    // than follow the card, NoL is only loosely related to the real incidence
    // angle — expect the slope term to be approximate on such geometry.
    float3 faceNormal = normalize(normalW);
    float NoL = abs(dot(gLightDir, faceNormal));
    float slope = (NoL > 0.f)
        ? clamp(sqrt(saturate(1.f - NoL * NoL)) / NoL, 0.f, gMaxSlopeDepthBias)
        : gMaxSlopeDepthBias;
    float depthBias = gSlopeDepthBias * slope + gConstantDepthBias;

    vOut.posH = mul(gLightViewProj, float4(posW, 1.f));

    // UE bClampToNearPlane: pull vertices that land in front of the near plane
    // back onto it, so a triangle straddling the near plane is not partially
    // clipped away and does not leave a hole in the shadowmap. Our clip z is
    // depth * gInvLightDepthRange with 0 at the near plane, so the clamp is a
    // lower bound.
    vOut.posH.z = max(vOut.posH.z, 0.f);

    // NOTE: the depth bias is deliberately NOT added to posH.z. UE does write
    // the biased value to OutPosition.z (ShadowDepthVertexShader.usf), but it can
    // afford to because its bias is a small fraction of the cascade's depth
    // range. Here the bias is expressed in world meters and reaches
    // SlopeDepthBias * MaxSlopeDepthBias + ConstantDepthBias — on Kroken's floor
    // (NoL ~= 0.1, slope clamped) that is about 8.8m against ~1.2m for the window
    // frame facing the light. Biasing the hardware depth would therefore reorder
    // the LessEqual test between differently-oriented casters by ~8 texels, and
    // the frame's stored depth would win texels the floor should have won,
    // erasing the mullion shadow. The hardware depth stays unbiased and purely
    // orders casters; only the color target carries the bias, and that is the
    // value ShadowProjectionPass compares against.

    vOut.instanceID = instanceID;
    vOut.materialID = gScene.getMaterialID(instanceID);
    vOut.posW = posW;
    vOut.texC = vIn.unpack().texCrd;
    vOut.lightDepth = depth + depthBias;
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
    v.posW = vsOut.posW;
    v.faceNormalW = float3(0.f, 0.f, 1.f);
    v.normalW = v.faceNormalW;
    v.tangentW = float4(0.f);
    v.texC = vsOut.texC;
    v.curveRadius = 0.f;
    v.coneTexLODValue = 0.f;
    let lod = ImplicitLodTextureSampler();
    if (gScene.materials.alphaTest(v, vsOut.materialID, lod))
        discard;
#endif

    // The bias was already folded in by the vertex shader.
    return vsOut.lightDepth;
}
