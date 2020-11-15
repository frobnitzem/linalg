#include <linalg.hh>
#include <stdio.h>
#include <stdlib.h>
#include <memory>

int main(int argc, char *argv[]) {
    using T = double;

    if(argc != 4) {
        printf("Usage: %s <m> <n> <k>\n", argv[0]);
        return 1;
    }
    int m = atol(argv[1]);
    int n = atol(argv[2]);
    int k = atol(argv[3]);

    Linalg::Context c;
    //auto c = Linalg::Context();
    auto A = std::make_shared<Linalg::Tile<T> >(m, k, m, Linalg::Place::CUDA);
    auto B = std::make_shared<Linalg::Tile<T> >(k, n, k, Linalg::Place::CUDA);
    auto C = std::make_shared<Linalg::Tile<T> >(m, n, m, Linalg::Place::CUDA);

    c.gemm(-1.0, A, B, 1.0, C);
    return 0;
}
