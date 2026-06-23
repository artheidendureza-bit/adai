#pragma once

// Main header for adaicpp library
// Includes all public headers

#include "adaicpp/device.h"
#include "adaicpp/tensor.h"
#include "adaicpp/memory_pool.h"
#include "adaicpp/node.h"
#include "adaicpp/graph.h"
#include "adaicpp/op_registry.h"

namespace adaicpp {

// Initialize the library (registers standard operations)
inline void initialize() {
    register_standard_operations();
}

} // namespace adaicpp
