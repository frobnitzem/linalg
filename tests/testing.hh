#include <linalg.hh>
#include <cmath>
#include <bits/stdc++.h>
#include <vector>

#define setup(m,n,k) \
    int64_t m = 100, n = 200, k = 50; \
    if(argc == 4) { \
        m = atol(argv[1]); \
        n = atol(argv[2]); \
        k = atol(argv[3]); \
    }

void print_times(std::vector<double> &results, double gflop) {
    sort(results.begin(), results.end());
    double sum = 0.0;
    int N = (results.size()+1)/2;
    // average of fastest times
    for(int i=0; i < N; i++) {
        sum += results[i];
    }
    printf("GFLOPS: %f\n", N*gflop/sum);
}

namespace Linalg {

template <typename T, typename U>
double nrm(TileP<T> A, TileP<U> B) {
    blas_error_if_msg(A->m != B->m || A->n != B->n,
                      "nrm requires identical tile dimensions");
    double ans = 0.0;
    #pragma omp parallel for reduction(max : ans)
    for(int64_t j=0; j<A->n; j++) {
        for(int64_t i=0; i<A->m; i++) {
            double u = std::norm( A->at(i,j) - B->at(i,j) );
            if(u > ans) ans = u;
        }
    }
    return std::sqrt(ans);
}

#define roundup(x,y) ( ( ((x) + (y) - 1) / (int64_t)(y) ) * (int64_t)(y) )

}
