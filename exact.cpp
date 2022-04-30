#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include <assert.h>
#include <sstream>
#include <exception>
#include <memory>
#include <filesystem>

#include "argparse.hpp"
#include "lazycsv.hpp"
#include "benchmark.h"


// A mxn matrix
// Simple wrapper, no check and easy to overflow, be cautious
class Matrix {
private:
    uint64_t m;
    uint64_t n;
    double* val;
public:
    Matrix(uint64_t vm, uint64_t vn) : n(vn), m(vm) {
        size_t sz = sizeof(double) * n * m;
        this->val = new double[sz];
        memset(this->val, 0, sz);
    }

    Matrix(const double* d, uint64_t vm, uint64_t vn) : Matrix(vm, vn) {
        memcpy(this->val, d, sizeof(double) * vn * vm);
    }

    Matrix(const Matrix&) = delete;
    Matrix(Matrix&&) = delete;

    ~Matrix() {
        delete[] this->val;
    }

    double* getVal() const {
        return this->val;
    }

    double getElement(size_t i, size_t j) const {
        return this->val[i * this->n + j];
    }

    void setElement(size_t i, size_t j, double d) {
        this->val[i*this->n + j] = d;
    }

    size_t getN() const {
        return this->n;
    }

    size_t getM() const {
        return this->m;
    }

    void pprint(const char* matrix_name = nullptr) const {
        if (matrix_name) {
            printf("%s:\n", matrix_name);
        }
        printf("--------[%ldx%ld]\n", this->m, this->n);
        for (size_t i = 0; i < this->m; i ++) {
            for (size_t j = 0; j < this->n; j ++) {
                printf("%f ", this->getElement(i, j));
            }
            printf("\n");
        }
        printf("--------\n");
    }
};

static std::vector<size_t> argsort(const std::vector<double>& mid) {
    std::vector<size_t> v(mid.size());
    for (size_t i = 0; i < mid.size(); i ++) {
        v[i] = i;
    }

    std::sort(v.begin(), v.end(), [&mid](const size_t& lhs, const size_t& rhs){
        return mid[lhs] < mid[rhs];
    });

    return v;
}



static void KNN(const Matrix* x_train, const Matrix* x_test, Matrix* gt, std::vector<double>& mid) {
    auto x_train_M = x_train->getM();
    auto x_train_N = x_train->getN();
    auto x_test_M = x_test->getM();
    auto x_test_N = x_test->getN();
    auto N1 = x_train_M;
    auto N2 = x_test_M;
    // bn == an

#ifndef NDEBUG
    x_train->pprint("x_train");
    x_test->pprint("x_test");
#endif

    assert(x_test_N == x_train_N);
    assert(gt->getM() == N2);
    assert(gt->getN() == N1);

    for (size_t i = 0; i < N2; i++) {
        for (size_t j = 0; j < N1; j++) {
            mid[j] = 0;
        }
        for (size_t j = 0; j < N1; j++) {
            for (size_t k = 0; k < x_train_N; k ++) {
                mid[j] += (x_train->getElement(j, k) - x_test->getElement(i, k)) * (x_train->getElement(j, k) - x_test->getElement(i, k));
            }
            mid[j] = std::sqrt(mid[j]);
        }
        auto sorted = argsort(mid);
        for (size_t k = 0; k < N2; k ++) {
            gt->setElement(i, k, sorted[k]);
        }
    }

#ifndef NDEBUG
    gt->pprint("gt");
#endif
}

static void compute_single_unweighted_knn_class_shapley(
    const Matrix* x_train, const Matrix* y_train, 
    const Matrix* gt, const Matrix* y_test,
    uint64_t K, Matrix* result) {
    auto N1 = x_train->getM();
    auto N2 = gt->getM();
    auto gtN = gt->getN();

#ifndef NDEBUG
    x_train->pprint("x_train");
    y_train->pprint("y_train");
    gt->pprint("gt");
    y_test->pprint("y_test");
#endif

    assert(y_train->getM() == 1);
    assert(y_test->getM() == 1);
    assert(result->getN() == N1);
    assert(result->getM() == N2);

    for (size_t j = 0; j < N2; j++) {
        result->setElement(j, gt->getElement(j, gtN - 1), int(y_train->getElement(0, gt->getElement(j, gtN - 1)) == y_test->getElement(0, j)) / double(N1));
        for (size_t i = N1 - 2; i <= N1 -2 ; i --) {
            result->setElement(j, gt->getElement(j, i), 
                result->getElement(j, gt->getElement(j, i+ 1)) + (
                    double(int(y_train->getElement(0, gt->getElement(j, i)) == y_test->getElement(0, j)) -
                    int(y_train->getElement(0, gt->getElement(j, i + 1)) == y_test->getElement(0, j))) / double(K) * std::min(K, i + 1) / double(i + 1)
                ));
        }
    }

#ifndef NDEBUG
    result->pprint("sp");
#endif
}

static void compute_sp_plain(
    const Matrix* x_train, const Matrix* x_test, const Matrix* y_train, 
    const Matrix* y_test, uint64_t K, std::vector<double>& mid,
    Matrix* gt, Matrix* sp) {
    KNN(x_train, x_test, gt, mid);
    compute_single_unweighted_knn_class_shapley(x_train, y_train, gt, y_test, K, sp);
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

// a = np.array([[0.47069075, 0.06548475],
//     [0.12246441, 0.57838926],
//     [0.98473347, 0.55588644]])
// b = np.array([[0.72012919, 0.04385545],
//         [0.42811407, 0.74712948],
//         [0.23655954, 0.43509146],
//         [0.12540547, 0.9914887 ]])

int main(int argc, char** argv) {
    argparse::ArgumentParser p("exact");

    p.add_argument("-j", "--json").default_value(false).implicit_value(true).help("Json output");
    p.add_argument("-p", "--print").default_value(false).implicit_value(true).help("Print input data and result");
    p.add_argument("-i", "--input").required().help("The input directory");

    try {
        p.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::stringstream ss;

        ss << err.what();
        ss << p;

        printf("%s\n", ss.str().c_str());
        exit(-1);
    }

    auto data = load_data(p.get<std::string>("-i"));
    Matrix gt(data->x_test.getM(), data->x_train.getM());
    Matrix sp(gt.getM(), gt.getN());
    std::vector<double> mid;
    mid.resize(data->x_train.getM());

    if (p.get<bool>("-p")) {
        data->x_train.pprint("x_train");
        data->x_test.pprint("x_test");
        data->y_train.pprint("y_train");
        data->y_test.pprint("y_test");
    }

    benchmark::Register("exact_sp_plain", std::bind(compute_sp_plain, &data->x_train, &data->x_test, &data->y_train, &data->y_test, 1, mid, &gt, &sp));

    benchmark::Run(p.get<bool>("-j"));

    if (p.get<bool>("-p")) {
        gt.pprint("gt");
        sp.pprint("sp");
    }

    return 0;
}