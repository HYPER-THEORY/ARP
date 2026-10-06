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
 #    from this software without specific written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/

// Debug buffer viewer. Forwards one of the candidate input textures to the
// output, optionally restricted to a single R/G/B/A channel. Selection is a
// uniform update — no graph rebuild. Indices must match DebugViewPass.cpp.

import Arctic.GBufferPass.GBufferHelpers;

cbuffer PerFrameCB
{
    uint  gSelection;
    uint  gChannel;   // 0=RGB, 1=R, 2=G, 3=B, 4=A
    uint2 gFrameDim;
};

Texture2D<float4> gFinal;
Texture2D<float4> gGBufferA;
Texture2D<float4> gGBufferB;
Texture2D<float4> gGBufferC;
Texture2D<float4> gGBufferD;
Texture2D<float4> gGBufferE;
Texture2D<float>  gDepth;
Texture2D<float4> gEmissive;
Texture2D<float2> gShadowMask;
Texture2D<float4> gAOMap;
Texture2D<float4> gDIColor;
Texture2D<float4> gAmbient;
Texture2D<float4> gIndirect;
Texture2D<float4> gComposite;
Texture2D<float4> gFieldDebug;

RWTexture2D<float4> gOutput;

[numthreads(8, 8, 1)]
void csMain(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x >= gFrameDim.x || dtid.y >= gFrameDim.y) return;

    float4 color = float4(0.f, 0.f, 0.f, 1.f);
    [forcecase] switch (gSelection)
    {
        case 0:  color = gFinal[dtid.xy]; break;
        case 1:  color = gGBufferA[dtid.xy]; break;
        case 2:  color = gGBufferB[dtid.xy]; break;
        case 3:  color = gGBufferC[dtid.xy]; break;
        case 4:  color = gGBufferD[dtid.xy]; break;
        case 5:  color = gGBufferE[dtid.xy]; break;
        case 6:  color = gDepth[dtid.xy].xxxx; break;
        case 7:  color = gEmissive[dtid.xy]; break;
        case 8:  color = float4(gShadowMask[dtid.xy], 0.f, 1.f); break;
        case 9:  color = gAOMap[dtid.xy].xxxx; break;
        case 10: color = gDIColor[dtid.xy]; break;
        case 11: color = gAmbient[dtid.xy]; break;
        case 12: color = gIndirect[dtid.xy]; break;
        case 13: color = gComposite[dtid.xy]; break;
        case 14: color = gFieldDebug[dtid.xy]; break;
        // Shading model, false-coloured with UE's own debug palette
        // (ShadingCommon.ush). Dark blue = Unlit, green = DefaultLit,
        // red = Subsurface, blue = TwoSidedFoliage, white = unrecognised.
        case 15: color = float4(getShadingModelColor(decodeShadingModelId(gGBufferB[dtid.xy].a)), 1.f); break;
    }

    float3 outRgb;
    [forcecase] switch (gChannel)
    {
        case 1: outRgb = color.rrr; break;
        case 2: outRgb = color.ggg; break;
        case 3: outRgb = color.bbb; break;
        case 4: outRgb = color.aaa; break;
        default: outRgb = color.rgb; break;
    }

    gOutput[dtid.xy] = float4(outRgb, 1.f);
}
