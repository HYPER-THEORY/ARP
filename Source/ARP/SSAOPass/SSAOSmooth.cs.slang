/// Copyright (C) 2026-2027 The ARP Project. All Rights Reserved.

cbuffer CB
{
    uint2 frameDim;
};

Texture2D<float>   gAOMap;
RWTexture2D<float> gSmoothedAOMap;

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 pixelPos = dispatchThreadID.xy;

    if (any(pixelPos >= frameDim))
    {
        return;
    }

    int2 min = int2(0, 0);
    int2 max = int2(frameDim) - 1;

    float s00 = gAOMap.Load(int3(clamp(pixelPos + int2(0, 0), min, max), 0));
    float s10 = gAOMap.Load(int3(clamp(pixelPos + int2(2, 0), min, max), 0));
    float s01 = gAOMap.Load(int3(clamp(pixelPos + int2(0, 2), min, max), 0));
    float s11 = gAOMap.Load(int3(clamp(pixelPos + int2(2, 2), min, max), 0));

    gSmoothedAOMap[pixelPos] = (s00 + s10 + s01 + s11) * 0.25;
}
