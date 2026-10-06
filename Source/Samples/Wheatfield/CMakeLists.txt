add_falcor_executable(Wheatfield)

target_sources(Wheatfield PRIVATE
    Wheatfield.cpp
    Wheatfield.h
)

target_link_libraries(Wheatfield PRIVATE Arctic)

target_source_group(Wheatfield "Samples")
