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

import Scene.Scene;
import Scene.Camera.Camera;
import Utils.Math.MathHelpers;

cbuffer PerFrameCB
{
    uint2 gFrameDim;
    float gLod;
    float gSunAngularRadius;
    float gSunHaloMultiplier;
};

Texture2D<float> gSceneDepth;
RWTexture2D<float4> gSkyboxColor;

[numthreads(8, 8, 1)]
void csMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadID.xy;
    if (any(pixel >= gFrameDim)) return;

    float deviceZ = gSceneDepth.Load(int3(pixel, 0)).r;

    // Pixels with depth == 1 (cleared, no geometry written) are background.
    if (deviceZ >= 1.f)
    {
        // Reconstruct view direction from pixel and sample the environment map.
        // Use Falcor's standard pinhole ray direction generator (cameraU/V/W
        // based) to avoid matrix multiply convention pitfalls with invViewProj.
        float3 dir = normalize(gScene.camera.computeNonNormalizedRayDirPinhole(pixel, gFrameDim, false));
        float3 sky = gScene.envMap.eval(dir, gLod);

        // Additive sun disk driven by the first directional light. DirectionalLight
        // has no subtended angle, so the disk half-angle is a UI-controlled radius.
        // The disk is placed at -dirW (the direction the light comes FROM).
        if (gScene.getLightCount() > 0)
        {
            LightData light = gScene.getLight(0);
            if (light.type == uint(LightType::Directional))
            {
                float3 sunDir = -light.dirW;
                float cosTheta = dot(dir, sunDir);
                float cosInner = cos(gSunAngularRadius);
                float cosOuter = cos(gSunAngularRadius * gSunHaloMultiplier);
                // smoothstep(cosOuter, cosInner, cosTheta): 1 inside disk, 0 outside halo.
                float alpha = smoothstep(cosOuter, cosInner, cosTheta);
                sky += light.intensity * alpha;
            }
        }

        gSkyboxColor[pixel] = float4(sky, 1.f);
    }
    else
    {
        gSkyboxColor[pixel] = float4(0.f, 0.f, 0.f, 0.f);
    }
}
