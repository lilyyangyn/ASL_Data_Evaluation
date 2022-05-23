#include "acutest.h"
#include "improved_mc.h"
#include "matrix.h"
#include <assert.h>
#include <cmath>

const double eps = 1e-6;
// const double eps = 0.6;

static void cmp(const Matrix* m, const double* v) {
    auto M = m->getM();
    auto N = m->getN();

    for (size_t i = 0 ; i < M; i ++) {
        for (size_t j = 0; j < N; j ++) {
            TEST_CHECK(std::fabs(double(m->getElement(i, j) - v[i * N + j])) < eps);
            TEST_MSG("Wrong (%zd, %zd): %f != %f", i, j, double(m->getElement(i, j)), v[i*N + j]);
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
            TEST_CHECK(std::fabs(double(a->getElement(i, j) - b->getElement(i, j))) < eps);
            TEST_MSG("Wrong (%zd, %zd): %f != (%zd, %zd) %f\n", i, j, double(a->getElement(i, j)), i, j, double(b->getElement(i, j)));
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

    std::vector<uint64_t> mid1(4);
    std::vector<double> mid2(4);
    Matrix phi(num_permutes, 4);
    FixedSizeKNNHeap H(K); 

    compute_sp_improved_mc(&x_train, &x_test, &y_train, &y_test, K, num_permutes, 
        &permutations, &point_dists, &sp, mid1, mid2, &phi, &H);
    
    cmp(&sp, (const double*)a5);

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
        uint64_t num_permutes = 100;\
        uint64_t K = 1;\
        Matrix permutations(num_permutes, data->x_train.getM());\
        Matrix point_dists(data->x_test.getM(), data->x_train.getM());\
        Matrix sp(data->x_test.getM(), data->x_train.getM());\
        std::vector<uint64_t> mid1;\
        mid1.resize(data->x_train.getM());\
        std::vector<double> mid2;\
        mid2.resize(data->x_train.getM());\
        Matrix phi(num_permutes, data->x_train.getM());\
        FixedSizeKNNHeap H(K);\
        compute_sp_improved_mc(&data->x_train, &data->x_test, &data->y_train, &data->y_test, K, num_permutes, &permutations, &point_dists, &sp, mid1, mid2, &phi, &H);\
        size_t r_sp_m, r_sp_n;\
        auto v_sp = read_csv(input_directory / "sp_gt.csv", r_sp_m, r_sp_n);\
        Matrix r_sp(v_sp, r_sp_m, r_sp_n);\
        write_csv(input_directory / "my_sp_gt_2.csv", &sp);\
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
TEST_SET_N(13)
TEST_SET_N(14)
TEST_SET_N(15)
TEST_SET_N(16)
TEST_SET_N(17)
TEST_SET_N(18)
TEST_SET_N(19)
TEST_SET_N(20)
TEST_SET_N(21)
TEST_SET_N(22)

#define TEST_SET_N_ENTRY(N) "test_set_" #N, test_set_##N

TEST_LIST = {
    // { "test_simple_array", test_simple_array },
    { TEST_SET_N_ENTRY(0) },
    // { TEST_SET_N_ENTRY(1) },
    { TEST_SET_N_ENTRY(2) },
    { TEST_SET_N_ENTRY(3) },
    { TEST_SET_N_ENTRY(4) },
    { TEST_SET_N_ENTRY(5) },
    { TEST_SET_N_ENTRY(6) },
    { TEST_SET_N_ENTRY(7) },
    { TEST_SET_N_ENTRY(8) },
    { TEST_SET_N_ENTRY(9) },
    { TEST_SET_N_ENTRY(10) },
    { TEST_SET_N_ENTRY(11) },
    { TEST_SET_N_ENTRY(12) },
    { TEST_SET_N_ENTRY(13) },
    { TEST_SET_N_ENTRY(14) },
    { TEST_SET_N_ENTRY(15) },
    { TEST_SET_N_ENTRY(16) },
    { TEST_SET_N_ENTRY(17) },
    { TEST_SET_N_ENTRY(18) },
    { TEST_SET_N_ENTRY(19) },
    { TEST_SET_N_ENTRY(20) },
    { TEST_SET_N_ENTRY(21) },
    { TEST_SET_N_ENTRY(22) },
    { NULL, NULL }
};