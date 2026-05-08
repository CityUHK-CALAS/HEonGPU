# HEonGPU helper: apply a unified-diff patch to a build-dir copy of a
# thirdparty source tree. Used to carry CUDA 13 compatibility fixes against
# thirdparty/GPU-NTT (and the small-N FFT extension against thirdparty/GPU-FFT)
# until upstream releases them.
#
# Usage:
#   heongpu_apply_patch(<patch_file> <source_dir> <out_dir_var>)
#     <patch_file>  Absolute path to the .patch file.
#     <source_dir>  Absolute path to the read-only thirdparty source.
#     <out_dir_var> Variable that receives the path to the patched copy
#                   inside the build directory; the caller passes that
#                   path to add_subdirectory().
#
# Behavior:
#   - Copies <source_dir> to ${CMAKE_BINARY_DIR}/_thirdparty/<basename>,
#     excluding ".git" so we are independent of whether the source is a git
#     working tree, a vendored snapshot, or a read-only mount. Source
#     submodules are never modified.
#   - Initializes a one-shot git repo on the copy so `git apply` has a tree
#     to operate against, then applies the patch idempotently.
#   - Re-running configure is a no-op when the patch is already applied to
#     the copy (verified via `git apply --reverse --check`).

find_package(Git REQUIRED)

function(heongpu_apply_patch patch_file source_dir out_dir_var)
    if(NOT EXISTS "${patch_file}")
        message(FATAL_ERROR "Patch file not found: ${patch_file}")
    endif()
    if(NOT EXISTS "${source_dir}")
        message(FATAL_ERROR "Source directory not found: ${source_dir}")
    endif()

    get_filename_component(_target_name "${source_dir}" NAME)
    set(_copy_dir "${CMAKE_BINARY_DIR}/_thirdparty/${_target_name}")

    if(NOT EXISTS "${_copy_dir}/.git")
        file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/_thirdparty")
        if(EXISTS "${_copy_dir}")
            file(REMOVE_RECURSE "${_copy_dir}")
        endif()
        # Trailing slash on source: copy contents into _copy_dir directly.
        file(COPY "${source_dir}/"
             DESTINATION "${_copy_dir}"
             PATTERN ".git" EXCLUDE
             PATTERN "build" EXCLUDE)
        execute_process(
            COMMAND ${GIT_EXECUTABLE} init --quiet --initial-branch=main
            WORKING_DIRECTORY "${_copy_dir}"
            RESULT_VARIABLE _git_init_rc
            OUTPUT_QUIET ERROR_QUIET)
        if(NOT _git_init_rc EQUAL 0)
            message(FATAL_ERROR
                "git init failed for patched thirdparty copy at ${_copy_dir}")
        endif()
        execute_process(
            COMMAND ${GIT_EXECUTABLE} -C "${_copy_dir}"
                    -c user.email=heongpu@build -c user.name=heongpu-build
                    add -A
            OUTPUT_QUIET ERROR_QUIET)
        execute_process(
            COMMAND ${GIT_EXECUTABLE} -C "${_copy_dir}"
                    -c user.email=heongpu@build -c user.name=heongpu-build
                    commit --quiet --allow-empty -m "snapshot"
            OUTPUT_QUIET ERROR_QUIET)
    endif()

    execute_process(
        COMMAND ${GIT_EXECUTABLE} -C "${_copy_dir}"
                apply --reverse --check ${patch_file}
        RESULT_VARIABLE _already_applied
        OUTPUT_QUIET ERROR_QUIET)
    if(_already_applied EQUAL 0)
        message(STATUS "Patch already applied (build copy): ${patch_file}")
        set(${out_dir_var} "${_copy_dir}" PARENT_SCOPE)
        return()
    endif()

    execute_process(
        COMMAND ${GIT_EXECUTABLE} -C "${_copy_dir}"
                apply --check ${patch_file}
        RESULT_VARIABLE _can_apply
        OUTPUT_QUIET
        ERROR_VARIABLE _check_err)
    if(NOT _can_apply EQUAL 0)
        message(FATAL_ERROR
            "Cannot apply patch ${patch_file} to ${_copy_dir}.\n"
            "  Source likely drifted from the patch's base revision.\n"
            "  git stderr: ${_check_err}")
    endif()

    execute_process(
        COMMAND ${GIT_EXECUTABLE} -C "${_copy_dir}" apply ${patch_file}
        RESULT_VARIABLE _apply_rc
        ERROR_VARIABLE _apply_err)
    if(NOT _apply_rc EQUAL 0)
        message(FATAL_ERROR
            "Failed to apply patch ${patch_file}: ${_apply_err}")
    endif()
    message(STATUS "Applied patch: ${patch_file} (build copy at ${_copy_dir})")
    set(${out_dir_var} "${_copy_dir}" PARENT_SCOPE)
endfunction()
