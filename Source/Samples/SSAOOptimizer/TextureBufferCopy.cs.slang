/// Copyright (C) 2026-2027 ARP (Auto Render Pipeline). All Rights Reserved.

/** Copies between an R32Float texture and a row-major float buffer.

    ASSAOPass takes and produces textures, and PyTorch reaches GPU memory only
    through shared structured buffers. These two dispatches move an image
    between the two forms on the device, so the AO image, the target and
    dL/dAO never pass through the host.

    The buffer side is tightly packed, index = y * frameDim.x + x, which is the
    layout Buffer.to_torch() maps as an [height, width] tensor.

      - `textureToBuffer` : gSrcTexture -> gDstBuffer
      - `bufferToTexture` : gSrcBuffer  -> gDstTexture
*/

cbuffer CB
{
    uint2 frameDim;
};

Texture2D<float>          gSrcTexture;
RWStructuredBuffer<float> gDstBuffer;

StructuredBuffer<float>   gSrcBuffer;
RWTexture2D<float>        gDstTexture;

[numthreads(16, 16, 1)]
void textureToBuffer(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 pixelPos = dispatchThreadID.xy;
    if (any(pixelPos >= frameDim))
        return;

    gDstBuffer[pixelPos.y * frameDim.x + pixelPos.x] = gSrcTexture[pixelPos];
}

[numthreads(16, 16, 1)]
void bufferToTexture(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 pixelPos = dispatchThreadID.xy;
    if (any(pixelPos >= frameDim))
        return;

    gDstTexture[pixelPos] = gSrcBuffer[pixelPos.y * frameDim.x + pixelPos.x];
}
