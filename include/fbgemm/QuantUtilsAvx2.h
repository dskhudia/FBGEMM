#pragma once

#include <cstdint>
#include "FbgemmBuild.h"

namespace fbgemm {

// Copied from gemmlowp
//
// A structure to hold quantization parameters 'scale' and 'zero_point'.
// The meaning of these values is as the constants in the quantization equation
//
//   real_value = scale * (quantized_value - zero_point)
//
// In other words, 'zero_point' is the quantized value that corresponds
// to the real value 0, and 'scale' is the difference of real values
// corresponding to consecutive quantized values.
struct FBGEMM_API TensorQuantizationParams {
  float scale;
  std::int32_t zero_point;
  int precision;
  float Min() const;
  float Max() const;
};

// Parameters when we scale from int32 intermediate matrix multiplication
// results to 8-bit integers
struct FBGEMM_API RequantizationParams {
  // For floating-point requantization
  float real_multiplier;

  // For fixed-point requantization
  std::int32_t multiplier;
  int right_shift;

  TensorQuantizationParams target_qparams;
};

////////////////////////////////////////////////////////////////////////////////
// Utility functions

void QuantizeAvx2(
    const float* src,
    std::uint8_t* dst,
    int len,
    const TensorQuantizationParams& qparams);

/**
 * @brief Find the min and max value in a float matrix.
 */
void FBGEMM_API FindMinMax(const float* m, float* min, float* max, int len);

void RequantizeFixedPointAvx2(
    const std::int32_t* src,
    std::uint8_t* dst,
    int len,
    const RequantizationParams& params);

void RequantizeAvx2(
    const std::int32_t* src,
    std::uint8_t* dst,
    int len,
    const RequantizationParams& params);

} // namespace fbgemm
