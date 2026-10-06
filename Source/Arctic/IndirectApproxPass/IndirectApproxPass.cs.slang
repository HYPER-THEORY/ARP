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

/** Real-time >= 2-bounce indirect lighting for a wheat canopy.

    Per pixel: reconstruct the world position, look up the column's LAI from the
    vertical profile and the lateral density map, solve PROSAIL's 4SAIL transport
    at that depth in closed form, and gather the result through the leaf's
    two-sided BSDF.

    See FourSail.slang for the model, how the interior fluxes are obtained from
    PROSAIL's own layer operators, and the bounce accounting. This file is only
    the entry point and the GBuffer plumbing.

    Note that the view direction is deliberately unused. The interior multiple
    scattering field is treated as isotropic within each hemisphere, which is a
    good approximation precisely because it has scattered several times; the
    directional effects that do survive (the hot spot) are single scattering and
    belong to DIPass. This is also why 4SAIL's ko / sob / sof / hotspot machinery
    is absent: it exists to produce view-direction radiance.
*/

import Scene.Scene;
import Scene.Camera.Camera;
import Arctic.GBufferPass.GBufferHelpers;
import Arctic.IndirectApproxPass.FourSail;

cbuffer PerFrameCB
{
    uint2 gFrameDim;
    uint  gDebugView; ///< 0 = off, see kDebug* below.
};

// Debug views. Kept in sync with IndirectApproxPass::renderUI().
static const uint kDebugOff = 0u;
static const uint kDebugEDown = 1u;
static const uint kDebugEUp = 2u;
static const uint kDebugDensity = 3u;
static const uint kDebugTau = 4u;
static const uint kDebugLai = 5u;
static const uint kDebugUncollidedShare = 6u;

Texture2D<float> gSceneDepth;
Texture2D<float4> gGBufferA;
Texture2D<float4> gGBufferB;
Texture2D<float4> gGBufferC;
Texture2D<float4> gGBufferD;

RWTexture2D<float4> gIndirect;
RWTexture2D<float4> gFieldDebug;

float3 reconstructWorldPos(uint2 pixel, float deviceZ)
{
    float2 ndc;
    ndc.x = (float(pixel.x) + 0.5f) / float(gFrameDim.x) * 2.f - 1.f;
    ndc.y = 1.f - (float(pixel.y) + 0.5f) / float(gFrameDim.y) * 2.f;
    // Falcor matrices are column-vector: v' = M * v, so mul(M, v) not mul(v, M).
    float4 worldH = mul(gScene.camera.data.invViewProj, float4(ndc, deviceZ, 1.f));
    return worldH.xyz / worldH.w;
}

[numthreads(8, 8, 1)]
void csMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadID.xy;
    if (any(pixel >= gFrameDim)) return;

    gIndirect[pixel] = float4(0.f, 0.f, 0.f, 1.f);
    gFieldDebug[pixel] = float4(0.f, 0.f, 0.f, 1.f);

    float deviceZ = gSceneDepth.Load(int3(pixel, 0)).r;
    if (deviceZ >= 1.f) return; // background

    // Foliage only. Ground and other surfaces are left to AmbientPass.
    float4 b = gGBufferB.Load(int3(pixel, 0));
    if (decodeShadingModelId(b.a) != SHADINGMODELID_TWOSIDED_FOLIAGE) return;

    float3 worldPos = reconstructWorldPos(pixel, deviceZ);

    CanopyFluxes f = solveCanopyField(worldPos);

    float3 N = normalize(decodeNormal(gGBufferA.Load(int3(pixel, 0)).rgb));
    float3 baseColor = gGBufferC.Load(int3(pixel, 0)).rgb;
    // GBufferD.rgb is sqrt-encoded on the 8-bit target (UE ExtractSubsurfaceColor).
    float3 subsurfaceColor = extractSubsurfaceColor(gGBufferD.Load(int3(pixel, 0)).rgb);

    // The leaf's own scattering happens here, which is what makes the total
    // >= 2 bounces. Deliberately NOT multiplied by AO: AO models occlusion of
    // the uncollided sky, and applying it would darken the very term that is
    // supposed to lift the shadow side.
    float3 indirect = gatherTwoSidedFoliage(f, N, baseColor, subsurfaceColor);

    gIndirect[pixel] = float4(indirect, 1.f);

    // ---- Debug ----
    float density = xyDensityScale(worldPos.xy);
    float lai = gLai * density;
    float tau = clamp(laiAbove(worldPos.z) * density, 0.f, lai);
    gFieldDebug[pixel] = float4(density, tau, lai, 1.f);

    if (gDebugView != kDebugOff)
    {
        float3 v = float3(0.f);
        switch (gDebugView)
        {
        case kDebugEDown:
            v = f.eDown;
            break;
        case kDebugEUp:
            v = f.eUp;
            break;
        case kDebugDensity:
            v = float3(density);
            break;
        case kDebugTau:
            v = float3(tau);
            break;
        case kDebugLai:
            v = float3(lai);
            break;
        case kDebugUncollidedShare:
            // How much of the raw downward flux survived the uncollided
            // subtraction. Near 0 means the collided term is a small difference
            // of large numbers, i.e. the regime where that subtraction has to be
            // quadrature-consistent or it dominates the answer.
            v = f.eDown / max(f.eDown + kPi * gSkyRadiance, 1e-6f);
            break;
        }
        gIndirect[pixel] = float4(v, 1.f);
    }
}
