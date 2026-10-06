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
 #    contributors may be used to endorse or promote products derived from
 #    this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 # LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/

// Chaos-style GBuffer pass.
//
// The pixel shader dispatches to a per-material `assembleMaterialParams`
// callback that fills a `MaterialParameters` struct, then a shared
// `encodeMaterialToGbuffer` encoder writes the UE-layout MRTs. This mirrors
// ref/chaos/shaders/core/gbuffer/gbuffer_mr.hlsli's structure and intentionally
// bypasses Falcor's IMaterial/IMaterialInstance dispatch.

#include "Scene/VertexAttrib.slangh"

import Scene.Raster;
import Scene.Material.BasicMaterialData;
import Scene.Material.ShadingUtils;
import Utils.Math.MathHelpers;
import Arctic.GBufferPass.GBufferHelpers;
import Arctic.GBufferPass.GBufferCommon;
import Arctic.Materials.Bush.BushMaterialData;
import Arctic.Materials.Bush.BushGBuffer;
import Arctic.Materials.Standard.StandardGBuffer;

cbuffer PerFrameCB
{
    uint2 gFrameDim;
    // PixelParameters' per-frame half. gIsTemporalEnable reports whether a
    // temporal history exists downstream; materials gate any dithering on it,
    // since a stipple with nothing to integrate it cannot resolve.
    uint gIsTemporalEnable;
    int gJitterFrameIndex;
    float gDitherClipOffset;
    float gTransparencyDistanceThreshold;
    float4 gJitterProjectionOffset;
};

// UE GBuffer MRT output. SV_TARGET index order must match kGBufferChannels
// in GBufferPass.cpp. (Velocity is produced by MotionVectorPass, not here.)
struct GBufferPSOut
{
    float4 gBufferA : SV_TARGET0;   // World normal (xyz, N*0.5+0.5) + per-object data (a)
    float4 gBufferB : SV_TARGET1;   // Metallic, Specular, Roughness, ShadingModelID|SelectiveOutputMask
    float4 gBufferC : SV_TARGET2;   // Base color (rgb, sRGB) + GBufferAO (a)
    float4 gBufferD : SV_TARGET3;   // Custom data (sqrt(subsurfaceColor).rgb + opacity.a)
    float4 gBufferE : SV_TARGET4;   // Precomputed shadow factors (always 0 here)
    float4 emissive : SV_TARGET5;   // Emissive radiance (HDR). Not part of UE's GBuffer.
};

VSOut vsMain(VSIn vsIn)
{
    return defaultVS(vsIn);
}

// Build the material-agnostic per-pixel inputs every assembly callback receives.
// The frame stored here is the interpolated geometric vertex frame: normal
// mapping is the material's job and its result lands in MaterialParameters.
PixelParameters buildPixelParameters(const ShadingData sd, const VSOut vsOut, const bool isFrontFace)
{
    PixelParameters pixelParams;
    pixelParams.uv = sd.uv;
    pixelParams.worldNormal = float4(sd.frame.N, 0.f);
    pixelParams.worldTangent = sd.tangentW;
    pixelParams.tbn = float3x3(sd.frame.T, sd.frame.B, sd.frame.N);
    pixelParams.worldPosition = float4(sd.posW, 1.f);
    pixelParams.svPosition = vsOut.posH;
    // The rasterizer delivers 1/w_clip in SV_Position.w, so the reciprocal is the
    // view-space depth.
    pixelParams.linearDepth = 1.f / vsOut.posH.w;
    pixelParams.viewDirection = sd.V;
    pixelParams.isTemporalEnable = gIsTemporalEnable != 0;
    pixelParams.jitterFrameIndex = gJitterFrameIndex;
    pixelParams.ditherClipOffset = gDitherClipOffset;
    pixelParams.transparencyDistanceThreshold = gTransparencyDistanceThreshold;
    pixelParams.jitterProjectionOffset = gJitterProjectionOffset;
    pixelParams.isFrontFaceFloat = isFrontFace ? 1.f : -1.f;
    pixelParams.screenUVZ = float3(vsOut.posH.xy / (float2)gFrameDim, vsOut.posH.z);
    pixelParams.sampler = gScene.materials.getDefaultTextureSampler(sd.materialID);
    return pixelParams;
}

// Note: [earlydepthstencil] is intentionally NOT used here. With early
// depth-stencil, depth is written before the pixel shader runs, so alpha-tested
// (discarded) pixels would still write depth. Downstream passes (DIPass) then
// see geometry depth where the pixel was actually alpha-clipped, read zeroed
// GBuffer values, and shade the pixel black — producing the "black rectangle
// patches" symptom. Without early depth-stencil, discarded pixels also skip
// depth write, so they correctly appear as background (sky) in later passes.
GBufferPSOut psMain(
    VSOut vsOut,
    uint triangleIndex: SV_PrimitiveID,
    float3 barycentrics: SV_Barycentrics,
    bool isFrontFace: SV_IsFrontFace)
{
    float3 faceNormal = gScene.getFaceNormalW(vsOut.instanceID, triangleIndex);
    VertexData v = prepareVertexData(vsOut, faceNormal);

    const float3 viewDir = normalize(gScene.camera.getPosition() - v.posW);
    ShadingData sd = gScene.materials.prepareShadingData(v, vsOut.materialID, viewDir);

    const PixelParameters pixelParams = buildPixelParameters(sd, vsOut, isFrontFace);
    const MaterialHeader materialHeader = gScene.materials.getMaterialHeader(sd.materialID);
    const UnsupportedMaterialData unsupportedData = {};

    MaterialParameters mp = (MaterialParameters)0;
    const MaterialType materialType = gScene.materials.getMaterialType(sd.materialID);

    if (materialType == MaterialType::Bush)
    {
        const BushMaterialData md = reinterpret<BushMaterialData, MaterialPayload>(
            gScene.materials.getMaterialDataBlob(sd.materialID).payload);
        assembleMaterialParams_Bush(materialHeader, md, pixelParams, mp);
        customClip_Bush(materialHeader, md, pixelParams, mp);
    }
    else if (materialType == MaterialType::Standard)
    {
        const BasicMaterialData md = gScene.materials.getBasicMaterialData(sd.materialID);
        // UE's GBuffer is a metal-rough encoding. SpecGloss stores (F0.rgb,
        // glossiness) in the same field and has no lossless mapping onto it, so
        // it takes the unsupported-material path rather than being mis-unpacked.
        if (md.getShadingModel() == ShadingModel::MetalRough)
        {
            assembleMaterialParams_Standard(materialHeader, md, pixelParams, mp);
            customClip_Standard(materialHeader, md, pixelParams, mp);
        }
        else
        {
            assembleMaterialParams_Unsupported(materialHeader, unsupportedData, pixelParams, mp);
        }
    }
    else
    {
        assembleMaterialParams_Unsupported(materialHeader, unsupportedData, pixelParams, mp);
    }

    mp.normal = normalize(mp.normal);

    GBufferPSOut psOut = {};
    encodeMaterialToGbuffer(
        mp, psOut.gBufferA, psOut.gBufferB, psOut.gBufferC, psOut.gBufferD, psOut.gBufferE, psOut.emissive);

    return psOut;
}
