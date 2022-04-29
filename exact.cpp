#include <chrono>
#include <functional>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>

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
        printf("--------[%ldx%ld]\n", this->m, this->m);
        for (size_t i = 0; i < this->m; i ++) {
            for (size_t j = 0; j < this->n; j ++) {
                printf("%f ", this->getElement(i, j));
            }
            printf("\n");
        }
        printf("--------\n");
    }
};

static double measure_function(std::function<void()> f) {
    auto start = std::chrono::steady_clock::now();

    f();

    auto end = std::chrono::steady_clock::now();

    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
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

static void KNN_Simple(const Matrix* a, const Matrix* b, Matrix* c, std::vector<double>& mid) {
    double* pa = a->getVal();
    double* pb = b->getVal();
    double* pc = c->getVal();
    auto aM = a->getM();
    auto aN = a->getN();
    auto bM = b->getM();
    auto bN = b->getN();
    auto N1 = aM;
    auto N2 = bM;
    // bn == an

    for (size_t i = 0; i < N2; i++) {
        for (size_t j = 0; j < N1; j++) {
            mid[j] = 0;
        }
        for (size_t j = 0; j < N1; j++) {
            for (size_t k = 0; k < aN; k ++) {
                mid[j] += (a->getElement(j, k) - b->getElement(i, k)) * (a->getElement(j, k) - b->getElement(i, k));
            }
            mid[j] = std::sqrt(mid[j]);
        }
        auto sorted = argsort(mid);
        for (size_t k = 0; k < N2; k ++) {
            c->setElement(i, k, sorted[k]);
        }
    }
}

// a = np.array([[0.47069075, 0.06548475],
//     [0.12246441, 0.57838926],
//     [0.98473347, 0.55588644]])
// b = np.array([[0.72012919, 0.04385545],
//         [0.42811407, 0.74712948],
//         [0.23655954, 0.43509146],
//         [0.12540547, 0.9914887 ]])

int main() {
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
    Matrix a((const double*)a1, 3, 2);
    Matrix b((const double*)a2, 4, 2);
    Matrix c(4, 3);
    std::vector<double> mid(3);
    

    printf("KNN_Simple: %f ns\n", measure_function(std::bind(KNN_Simple, &a, &b, &c, mid)));

    return 0;
}