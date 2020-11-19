#include <linalg.hh>

int main(int argc, char *argv[]) {
    Linalg::Context c;
    Linalg::Tile<float> T(50, 50, 50, Linalg::Place::Host);
    return 0;
}
