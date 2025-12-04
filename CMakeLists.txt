cmake_minimum_required(VERSION 3.15)
project(efs_fast C)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

set(CMAKE_C_STANDARD 99)

if(MSVC)
    add_compile_options(/O2 /arch:AVX2 /fp:fast)
else()
    add_compile_options(-O3 -mavx2 -mfma -ffast-math -funroll-loops -fomit-frame-pointer -fPIC)
endif()

find_package(OpenMP REQUIRED)

add_library(efs_lib MODULE src/efs_fast/core.c)

target_link_libraries(efs_lib PRIVATE OpenMP::OpenMP_C Python::Module)

install(TARGETS efs_lib DESTINATION efs_fast)