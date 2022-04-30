#include <chrono>
#include <functional>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include <assert.h>

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

    double getElement(int i, int j) const {
        return this->val[i * this->n + j];
    }

    void setElement(int i, int j, double d) {
        this->val[i*this->n + j] = d;
    }

    int getN() const {
        return this->n;
    }

    int getM() const {
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

typedef struct Benchmark {
    const char* name;
    std::function<void()> func;
} Benchmark;

constexpr const int max_bench = 1024;

static Benchmark bench[max_bench];
static size_t bench_count = 0;

static double measure_function(const Benchmark& bench) {
    auto start = std::chrono::steady_clock::now();

    bench.func();

    auto end = std::chrono::steady_clock::now();

    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

static void register_benchmark(const char* name, std::function<void()> f) {
    bench[bench_count].name = name;
    bench[bench_count].func = f;
    bench_count++;
}

static void run_benchmark(bool json_output) {
    double result[max_bench];
    for (size_t i = 0 ; i < bench_count; i ++) {
        result[i] = measure_function(bench[i]);
    }

    if (json_output) {
        printf("{\n");
        for (size_t i = 0; i < bench_count; i ++) {
            printf("  \"%s\": {\n", bench[i].name);
            printf("    \"ns\": \"%f\"\n", result[i]);
            printf("  }");

            if (i != bench_count  - 1) {
                printf(",");
            }

            printf("\n");
        }
        printf("}\n");
    } else {
        for (size_t i = 0; i < bench_count; i ++) {
            printf("%s: %f nanoseconds (%f seconds)\n", bench[i].name, result[i], result[i] / 1e9);
        }
    }
}

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
    double* pa = x_train->getVal();
    double* pb = x_test->getVal();
    double* pc = gt->getVal();
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

static void compute_sp(
    const Matrix* x_train, const Matrix* x_test, const Matrix* y_train, 
    const Matrix* y_test, uint64_t K, std::vector<double>& mid,
    Matrix* gt, Matrix* sp) {
    KNN(x_train, x_test, gt, mid);
    compute_single_unweighted_knn_class_shapley(x_train, y_train, gt, y_test, K, sp);
}

// a = np.array([[0.47069075, 0.06548475],
//     [0.12246441, 0.57838926],
//     [0.98473347, 0.55588644]])
// b = np.array([[0.72012919, 0.04385545],
//         [0.42811407, 0.74712948],
//         [0.23655954, 0.43509146],
//         [0.12540547, 0.9914887 ]])

int main(int argc, char** argv) {
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
    Matrix x_train((const double*)a1, 3, 2);
    Matrix x_test((const double*)a2, 4, 2);
    Matrix gt(4, 3);
    std::vector<double> mid(3);
    Matrix y_train((const double*)a3, 1, 5);
    Matrix y_test((const double*)a3, 1, 5);
    Matrix sp(4, 3);
    bool json_output = false;
    
    if (argc == 2 && !strcmp(argv[1], "-j")) {
        json_output = true;
    }

    register_benchmark("exact_sp", std::bind(compute_sp, &x_train, &x_test, &y_train, &y_test, 1, mid, &gt, &sp));

    run_benchmark(json_output);

    return 0;
}