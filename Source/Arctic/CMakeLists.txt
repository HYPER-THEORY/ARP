add_library(Arctic STATIC)

target_sources(Arctic PRIVATE
    Materials/Bush/BushMaterial.cpp
    Materials/Bush/BushMaterial.h
    Materials/Bush/BushMaterialData.slang
    Materials/Bush/BushCommon.slang
    Materials/Bush/BushGBuffer.slang
    Materials/Bush/BushMaterial.slang
    Materials/Bush/BushMaterialInstance.slang
    Materials/Bush/BushBRDF.slang
    Materials/Standard/StandardGBuffer.slang
    GBufferPass/GBufferPass.cpp
    GBufferPass/GBufferPass.h
    GBufferPass/GBufferPass.3d.slang
    GBufferPass/GBufferHelpers.slang
    GBufferPass/GBufferCommon.slang
    ShadowPass/ShadowPass.cpp
    ShadowPass/ShadowPass.h
    ShadowPass/ShadowViewData.h
    ShadowPass/ShadowPass.3d.slang
    ShadowProjectionPass/ShadowProjectionPass.cpp
    ShadowProjectionPass/ShadowProjectionPass.h
    ShadowProjectionPass/ShadowProjectionPass.3d.slang
    SkyboxPass/SkyboxPass.cpp
    SkyboxPass/SkyboxPass.h
    SkyboxPass/SkyboxPass.3d.slang
    DIPass/DIPass.cpp
    DIPass/DIPass.h
    DIPass/DIPass.3d.slang
    GTAOPass/GTAOPass.cpp
    GTAOPass/GTAOPass.h
    GTAOPass/GTAOPass.cs.slang
    GTAOPass/GTAOSpatialFilter.cs.slang
    GTAOPass/GTAOTemporalFilter.cs.slang
    GTAOPass/GTAOUpsample.cs.slang
    SSAOPass/SSAOPass.cpp
    SSAOPass/SSAOPass.h
    SSAOPass/SSAOPass.cs.slang
    SSAOPass/SSAOSmooth.cs.slang
    SSAOPass/SSAOUpsample.cs.slang
    MotionVectorPass/MotionVectorPass.cpp
    MotionVectorPass/MotionVectorPass.h
    MotionVectorPass/MotionVectorPass.cs.slang
    TAAPass/TAAPass.cpp
    TAAPass/TAAPass.h
    TAAPass/TAAPass.cs.slang
    AmbientPass/AmbientPass.cpp
    AmbientPass/AmbientPass.h
    AmbientPass/AmbientPass.3d.slang
    CompositePass/CompositePass.cpp
    CompositePass/CompositePass.h
    CompositePass/CompositePass.cs.slang
    IndirectApproxPass/IndirectApproxPass.cpp
    IndirectApproxPass/IndirectApproxPass.h
    IndirectApproxPass/ProsailLidf.cpp
    IndirectApproxPass/ProsailLidf.h
    IndirectApproxPass/FourSail.slang
    IndirectApproxPass/IndirectApproxPass.cs.slang
    DebugViewPass/DebugViewPass.cpp
    DebugViewPass/DebugViewPass.h
    DebugViewPass/DebugViewPass.3d.slang
)

target_link_libraries(Arctic PUBLIC Falcor)

target_include_directories(Arctic PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

set_target_properties(Arctic PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_copy_shaders(Arctic Arctic)

target_source_group(Arctic "Arctic")
