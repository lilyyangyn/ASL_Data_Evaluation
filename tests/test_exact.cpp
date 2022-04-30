#include "acutest.h"
#include "exact.h"
#include <cmath>

const double eps = 1e-6;

static void cmp(const Matrix* m, const double* v) {
    auto M = m->getM();
    auto N = m->getN();

    for (size_t i = 0 ; i < M; i ++) {
        for (size_t j = 0; j < N; j ++) {
            TEST_CHECK(std::fabs(m->getElement(i, j) - v[i * N + j]) < eps);
        }
    }
}

static void test_simple_array() {
    double a1[][2] = {
        {0.47069075, 0.06548475},
        {0.12246441, 0.57838926},
        {0.98473347, 0.55588644}
    };
    double a2[][2] = {
        {0.72012919, 0.04385545},
        {0.42811407, 0.74712948},
        {0.23655954, 0.43509146},
        {0.12540547, 0.9914887}
    };
    double a3[][5] = {
        {1.0, 2.0, 3.0, 4.0, 5.0}
    };
    double r_gt[][3] = {
        {0, 2, 1},
        {1, 2, 0},
        {1, 0, 2},
        {1, 2, 0}
    };
    double r_sp[][3] = {
        { 1.0,          0.0,          0.0        },
        { 0.0,          1.0,          0.0        },
        {-0.16666667, -0.16666667,  0.33333333},
        { 0.0,          0.0,          0.0        }
    };
    Matrix x_train((const double*)a1, 3, 2);
    Matrix x_test((const double*)a2, 4, 2);
    Matrix gt(4, 3);
    std::vector<double> mid(3);
    Matrix y_train((const double*)a3, 1, 5);
    Matrix y_test((const double*)a3, 1, 5);
    Matrix sp(4, 3);

    compute_sp_plain(&x_train, &x_test, &y_train, &y_test, 1, mid, &gt, &sp);

    cmp(&gt, (const double*)r_gt);
    cmp(&sp, (const double*)r_sp);

    return;
}

TEST_LIST = {
    { "test_simple_array", test_simple_array },
    { NULL, NULL }
};