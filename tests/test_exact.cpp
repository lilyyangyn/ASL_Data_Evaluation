#include "acutest.h"
#include "exact.h"
#include "lazycsv.hpp"
#include "matrix.h"
#include <cmath>
#include <filesystem>
#include <assert.h>


const double eps = 1e-6;
const int num_test_sets = 1;

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

static void cmp_matrix(const Matrix* a, const Matrix* b) {
    auto M = a->getM();
    auto N = a->getN();
    auto b_M = b->getM();
    auto b_N = b->getN();
    // printf("M %lu, %lu", M, b_M);
    // printf("N %lu, %lu", N, b_N);
    TEST_ASSERT(M == b_M);
    TEST_ASSERT(N == b_N);

    for (size_t i = 0 ; i < M; i ++) {
        for (size_t j = 0; j < N; j ++) {
            // printf("%lu, %lu\n", i, j);
            // printf("%f, %f\n", a->getElement(i, j), b->getElement(i, j));
            TEST_CHECK(std::fabs(a->getElement(i, j) - b->getElement(i, j)) < eps);
            TEST_MSG("Wrong (%zd, %zd): %f != (%zd, %zd) %f\n", i, j, a->getElement(i, j), i, j, b->getElement(i, j));
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

// Allow cmake to inject this path
#ifndef TEST_DATA_DIR
#define TEST_DATA_DIR "../tests/"
#endif

#define TEST_SET_N(N) \
    static void test_set_##N() { \
        std::filesystem::path input_directory = std::string(TEST_DATA_DIR); \
        input_directory = input_directory / std::string("test_set_" #N); \
        TEST_ASSERT(std::filesystem::exists(input_directory)); \
        auto data = load_exact_data(input_directory);\
        Matrix gt(data->x_test.getM(), data->x_train.getM());\
        Matrix sp(gt.getM(), gt.getN());\
        std::vector<double> mid;\
        mid.resize(data->x_train.getM());\
        compute_sp_plain(&data->x_train, &data->x_test, &data->y_train, &data->y_test, 1, mid, &gt, &sp);\
        size_t r_gt_m, r_gt_n, r_sp_m, r_sp_n;\
        auto v_gt = read_csv(input_directory / "knn_gt.csv", r_gt_m, r_gt_n);\
        auto v_sp = read_csv(input_directory / "sp_gt.csv", r_sp_m, r_sp_n);\
        Matrix r_gt(v_gt, r_gt_m, r_gt_n);\
        Matrix r_sp(v_sp, r_sp_m, r_sp_n);\
        write_csv(input_directory / "my_knn_gt.csv", &gt);\
        write_csv(input_directory / "my_sp_gt.csv", &sp);\
        cmp_matrix(&gt, &r_gt);\
        cmp_matrix(&sp, &r_sp);\
    } \

TEST_SET_N(0)
TEST_SET_N(1)
TEST_SET_N(2)
TEST_SET_N(3)
TEST_SET_N(4)
TEST_SET_N(5)
TEST_SET_N(6)
TEST_SET_N(7)
TEST_SET_N(8)
TEST_SET_N(9)
TEST_SET_N(10)
TEST_SET_N(11)
TEST_SET_N(12)

#define TEST_SET_N_ENTRY(N) "test_set_" #N, test_set_##N

TEST_LIST = {
    { "test_simple_array", test_simple_array },
    // { TEST_SET_N_ENTRY(0) },
    // { TEST_SET_N_ENTRY(1) },
    // { TEST_SET_N_ENTRY(2) },
    // { TEST_SET_N_ENTRY(3) },
    // { TEST_SET_N_ENTRY(4) },
    // { TEST_SET_N_ENTRY(5) },
    // { TEST_SET_N_ENTRY(6) },
    // { TEST_SET_N_ENTRY(7) },
    // { TEST_SET_N_ENTRY(8) },
    // { TEST_SET_N_ENTRY(9) },
    // { TEST_SET_N_ENTRY(10) },
    // { TEST_SET_N_ENTRY(11) },
    // { TEST_SET_N_ENTRY(12) },
    { NULL, NULL }
};