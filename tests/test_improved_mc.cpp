#include "acutest.h"
#include "improved_mc.h"
#include "matrix.h"
#include <assert.h>
#include <cmath>

const double eps = 1e-6;

static void cmp(const Matrix* m, const double* v) {
    auto M = m->getM();
    auto N = m->getN();

    for (size_t i = 0 ; i < M; i ++) {
        for (size_t j = 0; j < N; j ++) {
            TEST_CHECK(std::fabs(m->getElement(i, j) - v[i * N + j]) < eps);
            TEST_MSG("Wrong (%zd, %zd): %f != %f", i, j, m->getElement(i, j), v[i*N + j]);
        }
    }
}

static void test_simple_array() {
    double a1[][2] = {
        {40.41239336, 41.58393483},
        {3.88269482, 14.27480309},
        {22.16561151, 23.42790303},
        {21.75263250, 39.93422261}
    };
    double a2[][2] = {
        {22.53411032, 34.51625690},
        {22.16561151, 23.42790303},
        {22.16561151, 23.42790303}
    };
    double a3[][4] = {
        {16.08934229, 47.33126170, 87.35825235, 98.53587074}
    };
    double a4[][3] = {
        {92.37973453, 87.35825235, 87.35825235}
    };

    double a5[][4] = {
        {0.00000000, 0.00000000, 0.00000000, 0.00000000},
        {0.00000000, 0.00000000, 1.00000000, 0.00000000},
        {0.00000000, 0.00000000, 1.00000000, 0.00000000}
    };

    Matrix x_train((const double*)a1, 4, 2);
    Matrix x_test((const double*)a2, 3, 2);
    Matrix y_train((const double*)a3, 1, 4);
    Matrix y_test((const double*)a4, 1, 3);


    // TODO: seems that only  num_permutes = 1 works?
    uint64_t num_permutes = 1;
    uint64_t K = 1;
    Matrix permutations(num_permutes, 4);
    Matrix point_dists(3, 4);
    Matrix sp(3, 4);
    
    random_permute(&permutations, x_train.getM(), num_permutes);
    point_distances(&x_train, &x_test, &point_dists);
    improved_single_unweighted_knn_class_shapley(&y_train, &y_test, &permutations, &point_dists, K, num_permutes, &sp);
    
    cmp(&sp, (const double*)a5);

    return;
}

TEST_LIST = {
    { "test_simple_array", test_simple_array },
    { NULL, NULL }
};