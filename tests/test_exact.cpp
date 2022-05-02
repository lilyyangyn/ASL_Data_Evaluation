#include "acutest.h"
#include "exact.h"
#include "lazycsv.hpp"
#include "matrix.h"
#include <cmath>
#include <filesystem>
#include <fstream>


const double eps = 1e-6;
const int num_test_sets = 1;

static void cmp(const Matrix* m, const double* v) {
    auto M = m->getM();
    auto N = m->getN();

    for (size_t i = 0 ; i < M; i ++) {
        for (size_t j = 0; j < N; j ++) {
            TEST_CHECK(std::fabs(m->getElement(i, j) - v[i * N + j]) < eps);
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
            TEST_CHECK(std::fabs(a->getElement(i, j) - b->getElement(i, j)) < eps);
        }
    }
}

typedef struct InputData {
    Matrix x_train;
    Matrix x_test;
    Matrix y_train;
    Matrix y_test;

    InputData(
            const double* x_train_data, size_t x_train_m, size_t x_train_n,
            const double* x_test_data, size_t x_test_m, size_t x_test_n,
            const double* y_train_data, size_t y_train_m, size_t y_train_n,
            const double* y_test_data, size_t y_test_m, size_t y_test_n) :
            x_train(x_train_data, x_train_m, x_train_n),
            x_test(x_test_data, x_test_m, x_test_n),
            y_train(y_train_data, y_train_m, y_train_n),
            y_test(y_test_data, y_test_m, y_test_n) {}
} InputData;

static std::vector<double> read_csv(const std::filesystem::path& path, size_t& M, size_t& N) {
    std::vector<double> data;
    lazycsv::parser<lazycsv::mmap_source, lazycsv::has_header<false>> p {path.string()};
    M = 0;
    N = 0;

    for (const auto row : p) {
        M++;
        N = 0;
        for (const auto cell : row) {
            N++;
            auto it = cell.raw();
            data.push_back(std::stod(std::string(it)));
        }
    }

    return data;
}

static std::unique_ptr<InputData> load_data(const std::filesystem::path& input_directory) {
    size_t x_train_m, x_train_n, x_test_m, x_test_n, y_train_m, y_train_n, y_test_m, y_test_n;

    auto x_train = read_csv(input_directory / "x_train.csv", x_train_m, x_train_n);
    auto x_test = read_csv(input_directory / "x_test.csv", x_test_m, x_test_n);
    auto y_train = read_csv(input_directory / "y_train.csv", y_train_m, y_train_n);
    auto y_test = read_csv(input_directory / "y_test.csv", y_test_m, y_test_n);

    assert(x_train_n * x_train_m == x_train.size());
    assert(x_test_n * x_test_m == x_test.size());
    assert(y_train_n * y_train_m == y_train.size());
    assert(y_test_n * y_test_m == y_test.size());

    return std::make_unique<InputData>(
            &x_train[0], x_train_m, x_train_n,
            &x_test[0], x_test_m, x_test_n,
            &y_train[0], y_train_m, y_train_n,
            &y_test[0], y_test_m, y_test_n);
}

static void write_csv(const std::filesystem::path& path, Matrix* matrix) {
    std::ofstream out(path);

    size_t M = matrix->getM();
    size_t N = matrix->getN();
    // double* val = matrix->getVal();
    for (size_t m = 0; m < M; m++) {
        for (size_t n = 0; n < N - 1; n++) {
            out << std::fixed << std::setprecision(8) << matrix->getElement(m, n) << ", ";
        }
        out << std::fixed << std::setprecision(8) << matrix->getElement(m, N-1);
        if (m != M-1) {
            out << "\n";
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

static void test_arrays() {
    int i = 2;
    while (true) {
        std::filesystem::path input_directory = "../tests/test_set_" + std::to_string(i);
        if (!std::filesystem::exists(input_directory)) {
            break;
        }

        auto data = load_data(input_directory);
        Matrix gt(data->x_test.getM(), data->x_train.getM());
        Matrix sp(gt.getM(), gt.getN());
        std::vector<double> mid;
        mid.resize(data->x_train.getM());

        compute_sp_plain(&data->x_train, &data->x_test, &data->y_train, &data->y_test, 1, mid, &gt, &sp);

        size_t r_gt_m, r_gt_n, r_sp_m, r_sp_n;
        Matrix r_gt(&read_csv(input_directory / "knn_gt.csv", r_gt_m, r_gt_n)[0], r_gt_m, r_gt_n);
        Matrix r_sp(&read_csv(input_directory / "sp_gt.csv", r_sp_m, r_sp_n)[0], r_sp_m, r_sp_n);

        write_csv(input_directory / "my_knn_gt.csv", &gt);
        write_csv(input_directory / "my_sp_gt.csv", &sp);

        cmp_matrix(&gt, &r_gt);
        cmp_matrix(&sp, &r_sp);

        i++;
    }
    
    return;
    
}

TEST_LIST = {
    { "test_simple_array", test_simple_array },
    { "test_arrays", test_arrays },
    { NULL, NULL }
};