#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

namespace mmopt {

// A and B may share a dtype while the final output has another dtype. The
// accumulator is explicit because Split-K workspaces and L0C store FP32 even
// when the returned tensor is FP16 or BF16.
struct TypeSignature {
    std::string a;
    std::string b;
    std::string c;
    std::string accumulator;
};

inline std::uint64_t ElementBytes(const std::string& dtype) {
    if (dtype == "fp16" || dtype == "bf16") return 2;
    if (dtype == "fp32") return 4;
    throw std::invalid_argument("unsupported element dtype: " + dtype);
}

inline TypeSignature SameTypeSignature(const std::string& dtype) {
    return {dtype, dtype, dtype, "fp32"};
}

}  // namespace mmopt
