add_falcor_executable(Kroken)

target_sources(Kroken PRIVATE
    Kroken.cpp
    Kroken.h
)

target_link_libraries(Kroken PRIVATE Arctic)

target_source_group(Kroken "Samples")
