add_falcor_executable(SSAOOptimizer)

target_sources(SSAOOptimizer PRIVATE
    SSAOOptimizer.cpp
    SSAOOptimizer.h
    TextureBufferCopy.cs.slang
    View.ps.slang
)

target_link_libraries(SSAOOptimizer PRIVATE Arctic ARP)

target_copy_shaders(SSAOOptimizer Samples/SSAOOptimizer)

target_source_group(SSAOOptimizer "Samples")
