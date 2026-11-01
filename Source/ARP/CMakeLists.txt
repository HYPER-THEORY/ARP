add_library(ARP STATIC)

target_sources(ARP PRIVATE
    SSAOPass/ACommon.slang
    SSAOPass/ASSAOPass.cpp
    SSAOPass/ASSAOPass.h
    SSAOPass/ASSAO.cs.slang
    SSAOPass/SSAOHelpers.cpp
    SSAOPass/SSAOHelpers.h
    SSAOPass/SSAOPass.cpp
    SSAOPass/SSAOPass.h
    SSAOPass/SSAO.cs.slang
    SSAOPass/SSAOSmooth.cs.slang
)

target_link_libraries(ARP PUBLIC Falcor)

target_include_directories(ARP PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

set_target_properties(ARP PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_copy_shaders(ARP ARP)

target_source_group(ARP "ARP")
