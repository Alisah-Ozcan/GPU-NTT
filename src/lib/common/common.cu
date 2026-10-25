// Copyright 2023-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "gpuntt/common/common.cuh"

namespace gpuntt
{
    void customAssert(bool condition, const std::string& errorMessage)
    {
        if (!condition)
        {
            throw std::invalid_argument(errorMessage);
        }
    }

} // namespace gpuntt
