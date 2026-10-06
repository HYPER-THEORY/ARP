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

// UE-style standalone MotionVector pass.
//
// Per-pixel velocity is camera-only and can be reconstructed cheaply from depth
// + the previous frame's view-projection matrix — UE's "hinge" trick
// (GetCameraMotionVelocity / ComputeStaticVelocity). This avoids a per-object
// velocity raster and removes the velocity MRT from GBufferPass.
//
// The output convention is byte-for-byte identical to the motion vectors
// GBufferPass used to write (calcMotionVector(pixelPos, prevPosH, dim) + jitter),
// so the downstream TAAPass can consume it unchanged.

import Scene.Scene;
import Scene.Camera.Camera;
import Utils.Math.MathHelpers;

cbuffer PerFrameCB
{
    uint2 gFrameDim;
};

Texture2D<float> gSceneDepth;     // GBufferPass.depth (device Z: 0=near, 1=far)
RWTexture2D<float2> gMotionVecs;  // RG16Float: current->previous UV offset

[numthreads(8, 8, 1)]
void csMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadID.xy;
    if (any(pixel >= gFrameDim)) return;

    float deviceZ = gSceneDepth.Load(int3(pixel, 0)).r;

    // NOTE: background / sky pixels (deviceZ == 1) are NOT special-cased. UE does
    // not either: ComputeStaticVelocity runs ClipToPrevClip on the far-plane depth
    // and yields the correct parallax-free, rotation-only motion, which is exactly
    // what TAA needs to accumulate the sky while the camera turns. Zeroing it here
    // would leave the sky's history misaligned under rotation, so it would get no
    // temporal accumulation at all. Falcor uses a finite far plane (math::perspective),
    // so unprojecting deviceZ == 1 gives a finite point at farZ with a non-zero w;
    // the residual translation error is ~posDelta/farZ, which is negligible.

    // Reconstruct the TRUE current-frame world position of this pixel. The depth
    // buffer was rasterized with the jittered projection; unprojecting the
    // pixel-center NDC with the (jittered) invViewProj yields the same world
    // point GBufferPass would have interpolated for this pixel. Falcor's jitter
    // is a pure x/y clip translation, so deviceZ is unaffected.
    float2 ndc;
    ndc.x = (float(pixel.x) + 0.5f) / float(gFrameDim.x) * 2.0f - 1.0f;
    ndc.y = 1.0f - (float(pixel.y) + 0.5f) / float(gFrameDim.y) * 2.0f;
    float4 clip = float4(ndc, deviceZ, 1.0f);
    float4 worldH = mul(gScene.camera.data.invViewProj, clip);
    float3 worldPos = worldH.xyz / worldH.w;

    // Reproject into the previous frame's (un-jittered) clip space.
    float4 prevPosH = mul(gScene.camera.data.prevViewProjMatNoJitter, float4(worldPos, 1.0f));

    float2 pixelPos = float2(pixel) + float2(0.5f, 0.5f);
    float2 motion = calcMotionVector(pixelPos, prevPosH, gFrameDim)
                  + float2(gScene.camera.data.jitterX, -gScene.camera.data.jitterY);

    gMotionVecs[pixel] = motion;
}
