#ifndef GEMMGEN_REFERENCE_CSA_H
#define GEMMGEN_REFERENCE_CSA_H

#include "typedef.h"
#include <iostream>

namespace gemmgen {
  namespace reference {
    enum class LayoutType {
      Trans, NoTrans
    };

    void singleCsa(LayoutType TypeA,
                    LayoutType TypeB,
                    int M, int N, int K,
                    real Alpha, real *A, int Lda,
                    real *B, int Ldb, real Beta);
  }
}

#endif //GEMMGEN_REFERENCE_GEMM_H