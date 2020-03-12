/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "bench/BenchUtils.h"
#include "fbgemm/FbgemmSpMM.h"
#include "src/RefImplementations.h"

#include <iostream>

using namespace std;
using namespace fbgemm;

int main(int, char**) {

  vector<vector<unsigned>> shapes = {{128, 1024, 1024}};

  // C is MxN -> CT is NxM
  // A is MxK -> AT is KxM
  // B is KxN -> BT is NxK

  for (auto const& s : shapes) {
    int m = s[0];
    int n = s[1];
    int k = s[2];

    if ((k % 4) != 0) {
      cout << "Skipping shape " << m << ", " << n << ", " << k;
      cout << " as K is not a multiple of 4" << endl;
      continue;
    }

    for (float fnz = 0.99; fnz >= 0.009999; fnz -= 0.01) {
      auto aData = getRandomSparseVector(m * k / 4);
      auto bData = getRandomSparseVector(k * n / 4, fnz);
      auto cData = getRandomSparseVector(m * n);

      auto aptr = reinterpret_cast<uint8_t*>(aData.data());
      auto bptr = reinterpret_cast<const int8_t*>(bData.data());
      auto cptr = reinterpret_cast<int32_t*>(cData.data());

      for (int i = 0; i < k * m; ++i) {
        aptr[i] &= 0x7F;
      }

      // We calculate C^T = B^T x A^T
      // B matrix is sparse and passed in as first matrix to generateSpMM
      int ldat = m;
      int ldbt = k;
      int ldct = m;

      aligned_vector<float> atData(k / 4 * m);
      aligned_vector<float> btData(n * k);
      aligned_vector<float> ctData(n * m);

      auto atptr = reinterpret_cast<const uint8_t*>(atData.data());
      auto btptr = reinterpret_cast<int8_t*>(btData.data());
      auto ctptr = reinterpret_cast<int32_t*>(ctData.data());

      // Transpose as if A is float so 4 columns are interleaved
      transpose_matrix(m, k / 4, aData.data(), k / 4, atData.data(), ldat);
      transpose_matrix(k, n, bptr, n, btptr, ldbt);

      auto fn = generateSpMM<int32_t>(n, m, k, btptr, ldbt, ldat, ldct);
      auto fn_varying_n = generateSpMM<int32_t>(n, k, btptr, ldbt);

      double FLOPs = m * n * k * 2;

      constexpr int NWARMUP = 5;
      constexpr int NITER = 32;
      auto secs = measureWithWarmup(
          [&]() { fn(atptr, ctptr, 0); },
          NWARMUP,
          NITER,
          [&]() {
            cache_evict(atData);
            cache_evict(btData);
            cache_evict(ctData);
          });

      auto secs_varying_n = measureWithWarmup(
          [&]() {
            fn_varying_n(
                atptr,
                ctptr,
                m,
                ldat /* ldb */,
                ldct /* ldc */,
                0 /* accum_flag */);
          },
          NWARMUP,
          NITER,
          [&]() {
            cache_evict(atData);
            cache_evict(btData);
            cache_evict(ctData);
          });

      cout << fnz << "," << (FLOPs / secs / 1e9) << ","
           << (fnz * FLOPs / secs / 1e9) << ","
           << (FLOPs / secs_varying_n / 1e9) << ","
           << (fnz * FLOPs / secs_varying_n / 1e9) << endl;
    }
  }
}
