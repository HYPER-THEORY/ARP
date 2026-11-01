/// Copyright (C) 2026-2027 ARP (Auto Render Pipeline). All Rights Reserved.

[Differentiable]
T aBilinearSample<T : __BuiltinFloatingPointType>(no_diff Texture2D<T> texture, float2 location, no_diff uint level)
{
    int3 dimensions;
    texture.GetDimensions(level, dimensions.x, dimensions.y, dimensions.z);

    float2 coord = location * dimensions.xy;

    int2 coord0 = int2(floor(coord - 0.5));
    float2 a = coord - 0.5 - coord0;

    T v00 = texture.Load(int3(clamp(coord0             , int2(0), dimensions.xy - 1), level));
    T v10 = texture.Load(int3(clamp(coord0 + int2(1, 0), int2(0), dimensions.xy - 1), level));
    T v01 = texture.Load(int3(clamp(coord0 + int2(0, 1), int2(0), dimensions.xy - 1), level));
    T v11 = texture.Load(int3(clamp(coord0 + int2(1, 1), int2(0), dimensions.xy - 1), level));

    return lerp(lerp(v00, v10, T(a.x)), lerp(v01, v11, T(a.x)), T(a.y));
}

[Differentiable]
T aTrilinearSample<T : __BuiltinFloatingPointType>(no_diff Texture2D<T> texture, float2 location, float level)
{
    int3 dimensions;
    texture.GetDimensions(0, dimensions.x, dimensions.y, dimensions.z);

    int level0 = int(floor(level));
    float a = level - level0;

    return lerp(
        aBilinearSample(texture, location, clamp(level0    , 0, dimensions.z - 1)),
        aBilinearSample(texture, location, clamp(level0 + 1, 0, dimensions.z - 1)), T(a));
}
