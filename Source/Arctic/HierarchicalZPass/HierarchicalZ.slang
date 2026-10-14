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

/** Sampling helpers for the HZB produced by HierarchicalZPass.

    Equivalent of UnrealEngine's Private/HZB.ush for the subset that concerns
    reading the pyramid. Two deliberate differences from UE:

      - UE declares the HZB textures, samplers and the coordinate transforms as
        globals, because a UE shader includes HZB.ush and the RDG binds them for
        it. Falcor's `import` makes a module's globals part of the importing
        program's root signature too, but then every importer is forced to bind
        them. The transforms are therefore passed in explicitly through
        HZBParams, and only the textures/sampler are left to the caller so a
        pass can choose which views to bind.

      - UE's HZBTracing.ush applies `TileZ *= lerp(.99, 1, saturate(t * 10))` to
        the sampled tile depth to make thin-surface hits conservative. That
        factor assumes reverse-Z, where a larger deviceZ is nearer. With
        Falcor's forward-Z the same intent needs the factor to move the value in
        the opposite direction (toward the camera means *smaller* deviceZ), so
        the constant is not reproduced here. A consumer that wants it should use
        `lerp(1.0, 1.01, saturate(t * 10))` and re-derive it against its own
        hit test rather than copying UE's number.

    Coordinate spaces
    -----------------
    The pyramid's mip 0 is half resolution (or the power-of-two-rounded half),
    so HZB UV != screen UV. `hzbUVToScreenUV` / `screenUVToHZBUV` convert, using
    the same mip-0-relative factors UE stores in HZBUvFactorAndInvFactor:

        hzbUV         = screenUV * viewSize / (2 * hzbSize)
        screenUV      = hzbUV * (2 * hzbSize) / viewSize

    Sampling a specific mip: use `sampleHZBFurthest` / `sampleHZBSamplerLevel`
    with an explicit mip index. The pyramid is a max/min reduction, so a coarser
    mip is a conservative bound for the whole tile: the furthest pyramid never
    reports a surface nearer than the true nearest one, and the closest pyramid
    never reports a surface further than the true furthest one.
*/

#pragma once

/** Coordinate transforms for the HZB, mirroring UE's FHZBParameters.

    All fields are in "mip 0" terms. `viewSize` is the viewport size the depth
    buffer was rendered at (in Falcor this is the depth texture's size, since the
    pass builds the pyramid over the whole texture).
*/
struct HZBParams
{
    float2 hzbSize;                     ///< Mip 0 dimensions, in texels.
    float2 viewSize;                    ///< Viewport dimensions, in texels.
    float2 hzbBaseTexelSize;            ///< 1 / hzbSize.
    float4 hzbUvFactorAndInvFactor;     ///< xy = viewSize / (2 * hzbSize), zw = its reciprocal.
    uint   hzbMipCount;                 ///< Number of mip levels in the pyramid.
    uint   isFurthestHZBValid;          ///< 0 if the furthest pyramid was not produced.
    uint   isClosestHZBValid;           ///< 0 if the closest pyramid was not produced.
};

/** Builds HZBParams from the mip 0 size, the view size and the mip count. */
HZBParams makeHZBParams(uint2 hzbSize, uint2 viewSize, uint mipCount)
{
    HZBParams p;
    p.hzbSize = float2(hzbSize);
    p.viewSize = float2(viewSize);
    p.hzbBaseTexelSize = 1.0f / float2(hzbSize);
    float2 factor = p.viewSize / (2.0f * p.hzbSize);
    p.hzbUvFactorAndInvFactor = float4(factor, 1.0f / factor);
    p.hzbMipCount = mipCount;
    p.isFurthestHZBValid = 1;
    p.isClosestHZBValid = 1;
    return p;
}

/** Screen UV -> HZB UV (UE: ViewportUVToHZBBufferUV). */
float2 screenUVToHZBUV(float2 screenUV, HZBParams params)
{
    return screenUV * params.hzbUvFactorAndInvFactor.xy;
}

/** HZB UV -> screen UV. */
float2 hzbUVToScreenUV(float2 hzbUV, HZBParams params)
{
    return hzbUV * params.hzbUvFactorAndInvFactor.zw;
}

/** Screen UV -> the centre of the HZB texel covering it (UE: SamplePixelToHZBUV). */
float2 screenUVToHZBTexelCentreUV(float2 screenUV, HZBParams params)
{
    const float2 hzbUV = screenUVToHZBUV(screenUV, params);
    return (floor(hzbUV * params.hzbSize) + 0.5f) * params.hzbBaseTexelSize;
}

/** Point-samples the furthest pyramid at an explicit mip level. */
float sampleHZBFurthestTexture(Texture2D<float> hzb, SamplerState s, float2 hzbUV, float mipLevel)
{
    return hzb.SampleLevel(s, hzbUV, mipLevel).r;
}

/** Point-samples the closest pyramid at an explicit mip level. */
float sampleHZBSamplerLevel(Texture2D<float> hzb, SamplerState s, float2 hzbUV, float mipLevel)
{
    return hzb.SampleLevel(s, hzbUV, mipLevel).r;
}

/** Depth range covered by one HZB texel of the given mip, along both axes.

    With forward-Z the furthest pyramid reports the largest deviceZ in the tile
    and the closest pyramid the smallest, so `[closest, furthest]` is a
    conservative bracket of every deviceZ the tile contains.
*/
float2 hzbTexelDepthRange(
    Texture2D<float> furthest,
    Texture2D<float> closest,
    SamplerState s,
    float2 hzbUV,
    float mipLevel)
{
    return float2(
        sampleHZBSamplerLevel(closest, s, hzbUV, mipLevel),
        sampleHZBFurthestTexture(furthest, s, hzbUV, mipLevel));
}

/** Returns the mip level whose texel footprint is at least `texelsWide` texels of mip 0. */
float hzbMipForTexelFootprint(float texelsWide)
{
    return max(log2(max(texelsWide, 1.0f)), 0.0f);
}
