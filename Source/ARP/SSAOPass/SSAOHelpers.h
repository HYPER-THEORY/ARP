/// Copyright (C) 2026-2027 ARP (Auto Render Pipeline). All Rights Reserved.

#pragma once

#include "Falcor.h"

using namespace Falcor;

namespace SSAOHelpers
{
/// Sampler for the tiled random-rotation normal texture: point filtering,
/// wrapping, so the shader's `temporalOffset + uv * frameDim / 64` lookup tiles
/// seamlessly.
ref<Sampler> createRandomNormalSampler(ref<Device> pDevice);

/// The 64x64 RG8Unorm texture that rotates the occlusion sample pattern. It is
/// a 4x4 tile of 16 unit-ish vectors at different angles and lengths, expanded
/// to 64x64 so a bilinear-free point sample of `uv * frameDim / 64` lands on a
/// different rotation per 64-pixel block.
ref<Texture> createRandomNormalTexture(ref<Device> pDevice);

} // namespace SSAOHelpers
