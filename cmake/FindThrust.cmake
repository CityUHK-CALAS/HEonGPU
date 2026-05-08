# Locate Thrust library and set THRUST_INCLUDE_DIR
#
# Supports both pre-CUDA-13 and CUDA 13+ layouts:
#   - Pre-13: <cuda>/include/thrust/version.h
#   - CUDA 13+: <cuda>/include/cccl/thrust/version.h
#             (NVIDIA consolidated Thrust/CUB/libcu++ under cccl/)

find_package(CUDAToolkit QUIET)

find_path(THRUST_INCLUDE_DIR
  NAMES thrust/version.h
  HINTS
    /usr/include/cuda
    /usr/local/include
    /usr/local/cuda/include
    ${CUDA_INCLUDE_DIRS}
    ${CUDA_TOOLKIT_ROOT_DIR}
    ${CUDA_SDK_ROOT_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CUDAToolkit_TARGET_DIR}
  PATH_SUFFIXES
    cccl
    include
    include/cccl
    targets/x86_64-linux/include
    targets/x86_64-linux/include/cccl
)

if (THRUST_INCLUDE_DIR)
  list(REMOVE_DUPLICATES THRUST_INCLUDE_DIR)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Thrust REQUIRED_VARS THRUST_INCLUDE_DIR)

if(Thrust_FOUND AND NOT TARGET Thrust)
  add_library(Thrust INTERFACE)
  target_include_directories(Thrust INTERFACE ${THRUST_INCLUDE_DIR})
endif()

mark_as_advanced(THRUST_INCLUDE_DIR)
