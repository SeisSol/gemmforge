#include "csa.h"

using namespace gemmgen::reference;

namespace gemmgen {
  namespace reference {

    void singleCsa(LayoutType TypeA,
                    LayoutType TypeB,
                    int M, int N, int K,
                    real Alpha, real *A, int Lda,
                    real *B, int Ldb,
                    real Beta) {

          for (int n = 0; n < N; ++n) {
            for (int m = 0; m < M; ++m) {
              B[m + n * Ldb] = Alpha * A[m + n * Lda] + Beta * B[m + n * Ldb];
            }
          }

        }
      }
    }



