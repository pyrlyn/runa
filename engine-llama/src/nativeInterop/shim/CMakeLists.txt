# A C API over the part of llama.cpp that has none: its `common` library, which renders a model's
# Jinja chat template, turns tools and JSON schemas into grammars and parses what the model answers.
# Kotlin/Native binds C only, so runa_llama.cpp wraps it. Built by the cmakeLlama* tasks of
# build.gradle.kts, together with llama.cpp itself.
cmake_minimum_required(VERSION 3.14)
project(runa-llama C CXX)

set(RUNA_LLAMA_DIR "" CACHE PATH "The llama.cpp sources")
add_subdirectory("${RUNA_LLAMA_DIR}" llama.cpp)

add_library(runa-llama STATIC runa_llama.cpp)
target_include_directories(runa-llama PUBLIC .)
target_link_libraries(runa-llama PRIVATE llama-common llama)
target_compile_features(runa-llama PRIVATE cxx_std_17)
