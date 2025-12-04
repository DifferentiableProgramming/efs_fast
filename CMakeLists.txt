cmake_minimum_required(VERSION 3.15)
project(efs_fast C)

include(FetchContent)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Find OpenMP on macOS (Homebrew) if standard lookup fails
if(APPLE)
    file(GLOB_RECURSE OPENMP_HINT_PATHS /opt/homebrew/opt/libomp/lib/cmake/*)
    list(APPEND CMAKE_PREFIX_PATH ${OPENMP_HINT_PATHS})
endif()

find_package(OpenMP REQUIRED)

set(CMAKE_C_STANDARD 99)

# --- 2. Architecture Detection & SIMDe ---
# Check if we are on ARM64 (Apple Silicon)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    message(STATUS "Detected ARM64/Apple Silicon. Fetching SIMDe for AVX2 emulation.")
    
    FetchContent_Declare(
        simde
        GIT_REPOSITORY https://github.com/simd-everywhere/simde.git
        GIT_TAG        v0.8.0
    )
    FetchContent_MakeAvailable(simde)
    
    # Flag to tell core.c to include simde headers
    add_compile_definitions(USE_SIMDE)
    
    # Standard optimizations for ARM64 (NEON is implied)
    add_compile_options(-O3 -funroll-loops -fomit-frame-pointer -fPIC)
    
    # Specific include for SIMDe
    include_directories(${simde_SOURCE_DIR})

elseif(MSVC)
    # Windows x86/x64
    add_compile_options(/O2 /arch:AVX2 /fp:fast)
else()
    # Linux/Intel Mac x86_64
    add_compile_options(-O3 -mavx2 -mfma -ffast-math -funroll-loops -fomit-frame-pointer -fPIC)
endif()

# --- 3. Build Target ---
add_library(efs_lib MODULE src/efs_fast/core.c)

set_target_properties(efs_lib PROPERTIES PREFIX "")

if(WIN32)
    set_target_properties(efs_lib PROPERTIES SUFFIX ".pyd")
endif()

# Link SIMDe if on ARM
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    # SIMDe is header-only, but we link the target to inherit include paths
    target_link_libraries(efs_lib PRIVATE simde)
endif()

target_link_libraries(efs_lib PRIVATE OpenMP::OpenMP_C Python::Module)

install(TARGETS efs_lib DESTINATION efs_fast)