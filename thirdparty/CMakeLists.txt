# Copyright 2024 Alişah Özcan
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# Developer: Alişah Özcan

execute_process(
    COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/build.sh
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    RESULT_VARIABLE BUILD_SCRIPT_RESULT
    OUTPUT_VARIABLE BUILD_SCRIPT_OUTPUT
    ERROR_VARIABLE BUILD_SCRIPT_ERROR
)

if(NOT BUILD_SCRIPT_RESULT EQUAL 0)
    message(FATAL_ERROR "Thirdparty build script failed:\n${BUILD_SCRIPT_ERROR}")
endif()

message(STATUS "Thirdparty build completed successfully")

include(FetchContent)

FetchContent_Declare(
    rmm
    GIT_REPOSITORY https://github.com/rapidsai/rmm
    GIT_TAG        branch-25.08
    SOURCE_SUBDIR  cpp
)

set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(rmm)

# GPU-FFT and GPU-NTT are consumed from build-dir copies populated by
# heongpu_apply_patch() in the parent CMakeLists.txt; this keeps the source
# submodules clean and decouples patch application from source location.
# Fall back to the in-tree submodule when the patched copy variable is unset
# (e.g. someone includes thirdparty/ directly in another build).
if(DEFINED HEONGPU_GPU_FFT_PATCHED_DIR)
    add_subdirectory("${HEONGPU_GPU_FFT_PATCHED_DIR}" GPU-FFT)
else()
    add_subdirectory(GPU-FFT)
endif()
if(DEFINED HEONGPU_GPU_NTT_PATCHED_DIR)
    add_subdirectory("${HEONGPU_GPU_NTT_PATCHED_DIR}" GPU-NTT)
else()
    add_subdirectory(GPU-NTT)
endif()
add_subdirectory(RNGonGPU)

