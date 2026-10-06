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

/** Half-res dither smoothing pass.

    Port of UE's MainSSAOSmoothCS (Private/PostProcessAmbientOcclusion.usf).
    The SSAO compute pass uses a 4x4-tiled random rotation, producing a
    visible 4x4 dither pattern in the half-res AO. This pass averages a
    stride-2 2x2 box (the four samples are one 4x4-tile apart) to remove the
    dither without blurring across silhouette edges any more than necessary.

    Inputs:  gAOMapHalf (half-res RGBA32Float)
    Outputs: gAOMapHalfSmoothed (half-res RGBA32Float)
*/

import Scene.Scene;
import Scene.Camera.Camera;

cbuffer PerFrameCB
{
    uint2 gHalfFrameDim;
};

Texture2D<float4> gAOMapHalf;
RWTexture2D<float4> gAOMapHalfSmoothed;

[numthreads(8, 8, 1)]
void csMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int2 pixel = int2(dispatchThreadID.xy);
    if (any(pixel >= int2(gHalfFrameDim))) return;

    // 2x2 box at stride 2 — covers the 4x4 random tile (UE's pattern).
    float4 s00 = gAOMapHalf.Load(int3(clamp(pixel + int2(0, 0), int2(0, 0), int2(gHalfFrameDim) - 1), 0));
    float4 s10 = gAOMapHalf.Load(int3(clamp(pixel + int2(2, 0), int2(0, 0), int2(gHalfFrameDim) - 1), 0));
    float4 s01 = gAOMapHalf.Load(int3(clamp(pixel + int2(0, 2), int2(0, 0), int2(gHalfFrameDim) - 1), 0));
    float4 s11 = gAOMapHalf.Load(int3(clamp(pixel + int2(2, 2), int2(0, 0), int2(gHalfFrameDim) - 1), 0));

    gAOMapHalfSmoothed[pixel] = (s00 + s10 + s01 + s11) * 0.25f;
}
