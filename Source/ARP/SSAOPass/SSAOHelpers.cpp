/// Copyright (C) 2026-2027 ARP (Auto Render Pipeline). All Rights Reserved.

#include "SSAOHelpers.h"

namespace SSAOHelpers
{

ref<Sampler> createRandomNormalSampler(ref<Device> pDevice)
{
    Sampler::Desc desc;
    desc.setFilterMode(TextureFilteringMode::Point, TextureFilteringMode::Point, TextureFilteringMode::Point);
    desc.setAddressingMode(TextureAddressingMode::Wrap, TextureAddressingMode::Wrap, TextureAddressingMode::Wrap);
    return pDevice->createSampler(desc);
}

ref<Texture> createRandomNormalTexture(ref<Device> pDevice)
{
    constexpr int kReorder[16] = {0, 11, 7, 3, 10, 4, 15, 12, 6, 8, 1, 14, 13, 2, 9, 5};

    using RG = math::vector<uint8_t, 2>;

    RG bases[16];

    for (int pos = 0; pos < 16; ++pos)
    {
        float w = kReorder[pos];
        float ww = w / 16.0f * M_PI;
        float lenm = 1.0f - (std::sin(w * 1.98f) * 0.5f + 0.5f) * 0.23f;
        float c = std::cos(ww) * lenm;
        float s = std::sin(ww) * lenm;
        bases[pos].r = uint8_t((c * 0.5f + 0.5f) * 255.f + 0.5f);
        bases[pos].g = uint8_t((s * 0.5f + 0.5f) * 255.f + 0.5f);
    }

    const uint32_t kExtent = 64;

    std::vector<RG> dest(kExtent * kExtent);

    for (int y = 0; y < kExtent; ++y)
    {
        for (int x = 0; x < kExtent; ++x)
        {
            dest[y * kExtent + x] = bases[(x % 4) + (y % 4) * 4];
        }
    }

    return pDevice->createTexture2D(
        kExtent, kExtent, ResourceFormat::RG8Unorm, 1, 1, dest.data(), ResourceBindFlags::ShaderResource);
}

} // namespace SSAOHelpers
