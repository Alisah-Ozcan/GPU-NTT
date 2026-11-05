# Copyright 2023-2026 Alişah Özcan
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# Developer: Alişah Özcan

# A project that uses an installed GPU-NTT, to check the installed package.
# Configure it with CMAKE_PREFIX_PATH set to the GPU-NTT install prefix.

cmake_minimum_required(VERSION 3.26)
project(gpuntt_package_test LANGUAGES CXX CUDA)

find_package(GPUNTT REQUIRED)

add_executable(package_test main.cu)
target_link_libraries(package_test PRIVATE GPUNTT::ntt)
set_target_properties(package_test PROPERTIES
    CUDA_ARCHITECTURES "${GPUNTT_CUDA_ARCHITECTURES}"
)
