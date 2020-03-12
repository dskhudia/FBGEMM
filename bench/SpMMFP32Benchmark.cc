/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "bench/BenchUtils.h"
#include "fbgemm/FbgemmSpMM.h"

#include <iostream>

using namespace std;
using namespace fbgemm;

int main(int, char**) {

  vector<vector<int>> shapes = {{128, 1024, 1024}};

  // C is MxN -> CT is NxM
  // A is MxK -> AT is KxM
  // B is KxN -> BT is NxK

  // for (int s = 64; s <= 128; s *= 2)
  for (auto const& s : shapes) {
    int m = s[0];
    int n = s[1];
    int k = s[2];

    for (float fnz = 0.99; fnz >= 0.009999; fnz -= 0.01) {
      auto aData = getRandomSparseVector(m * k);
      auto bData = getRandomSparseVector(k * n, fnz);
      auto cData = getRandomSparseVector(m * n);

      aligned_vector<float> atData(k * m);
      aligned_vector<float> btData(n * k);
      aligned_vector<float> ctData(n * m);

      transpose_matrix(m, k, aData.data(), k, atData.data(), m);
      transpose_matrix(k, n, bData.data(), n, btData.data(), k);

      // We calculate C^T = B^T x A^T
      // B matrix is sparse and passed in as first matrix to generateSpMM
      int ldat = m;
      int ldbt = k;
      int ldct = m;
      auto fn = generateSpMM<float>(n, m, k, btData.data(), ldbt, ldat, ldct);
      auto fn_varying_n = generateSpMM<float>(n, k, btData.data(), ldbt);

      double effective_flop = m * n * k * 2;

      constexpr int NWARMUP = 5;
      constexpr int NITER = 32;
      auto secs = measureWithWarmup(
          [&]() { fn(atData.data(), ctData.data(), 0); },
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
                atData.data(),
                ctData.data(),
                m,
                ldat, /* ldat */
                ldct, /* ldct */
                0 /* accum_flag */);
          },
          NWARMUP,
          NITER,
          [&]() {
            cache_evict(atData);
            cache_evict(btData);
            cache_evict(ctData);
          });

      double effective_gflops = effective_flop / secs / 1e9;
      double effective_gflops_varying_n = effective_flop / secs_varying_n / 1e9;
      cout << fnz << "," << effective_gflops << "," << fnz * effective_gflops
           << "," << effective_gflops_varying_n << ","
           << fnz * effective_gflops_varying_n << endl;
    }
  }
}
