#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include <assert.h>
#include <sstream>
#include <memory>
#include <filesystem>

#include "exact.h"
#include "x86intrin.h"

static std::vector<size_t> argsort(const std::vector<double>& mid) {
    std::vector<size_t> v(mid.size());
    for (size_t i = 0; i < mid.size(); i ++) {
        v[i] = i;
    }

    std::stable_sort(v.begin(), v.end(), [&mid](const size_t& lhs, const size_t& rhs){
#ifdef FLOPS
        getCounter()->Increase(1);
#endif
        return mid[lhs] < mid[rhs];
    });

    return v;
}

void KNN_unroll4(const Matrix* x_train, const Matrix* x_test, Matrix* gt, std::vector<double>& mid) {
    auto x_train_M = x_train->getM();
    auto x_train_N = x_train->getN();
    auto x_test_M = x_test->getM();
    auto x_test_N = x_test->getN();
    auto N1 = x_train_M;
    auto N2 = x_test_M;
    // bn == an

    assert(x_test_N == x_train_N);
    assert(gt->getM() == N2);
    assert(gt->getN() == N1);

    for (size_t i = 0; i < N2; i++) {
        size_t j = 0;
        // optimized as memset by gcc
        for (j = 0; j < N1; j++) {
            mid[j] = 0;
        }
        for (j = 0; j < N1 - 3; j+=4) {
            for (size_t k = 0; k < x_train_N; k ++) {
                auto val = (x_train->getElement(j, k) - x_test->getElement(i, k)); // avx-ed by gcc
                auto val1 = (x_train->getElement(j + 1, k) - x_test->getElement(i, k));
                auto val2 = (x_train->getElement(j + 2, k) - x_test->getElement(i, k));
                auto val3 = (x_train->getElement(j + 3, k) - x_test->getElement(i, k));
                mid[j] += val * val;
                mid[j + 1] += val1 * val1;
                mid[j + 2] += val2 * val2;
                mid[j + 3] += val3 * val3;
            }
            mid[j] = std::sqrt(mid[j]); // optmized by sqrt by gcc
            mid[j + 1] = std::sqrt(mid[j + 1]);
            mid[j + 2] = std::sqrt(mid[j + 2]);
            mid[j + 3] = std::sqrt(mid[j + 3]);
        }
        for (; j < N1; j++) {
            for (size_t k = 0; k < x_train_N; k ++) {
                auto val = (x_train->getElement(j, k) - x_test->getElement(i, k));
                mid[j] +=  val * val;
            }
            mid[j] = std::sqrt(mid[j]);
        }
        auto sorted = argsort(mid);
        for (size_t k = 0; k < N1; k ++) {
            gt->setElement(i, k, sorted[k]);
        }
#ifndef NDEBUG
        fprintf(stderr, "%zd:", i);
        for (auto& it : sorted) {
            fprintf(stderr, "%zd ", it);
        }
        fprintf(stderr, "\n");
#endif
    }
}

void KNN(const Matrix* x_train, const Matrix* x_test, Matrix* gt, std::vector<double>& mid) {
    auto x_train_M = x_train->getM();
    auto x_train_N = x_train->getN();
    auto x_test_M = x_test->getM();
    auto x_test_N = x_test->getN();
    auto N1 = x_train_M;
    auto N2 = x_test_M;
    // bn == an

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
        for (size_t k = 0; k < N1; k ++) {
            gt->setElement(i, k, sorted[k]);
        }
#ifndef NDEBUG
        fprintf(stderr, "%zd:", i);
        for (auto& it : sorted) {
            fprintf(stderr, "%zd ", it);
        }
        fprintf(stderr, "\n");
#endif
    }
}

void compute_single_unweighted_knn_class_shapley(
    const Matrix* x_train, const Matrix* y_train, 
    const Matrix* gt, const Matrix* y_test,
    uint64_t K, Matrix* result) {
    auto N1 = x_train->getM();
    auto N2 = gt->getM();
    auto gtN = gt->getN();

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
                    int(y_train->getElement(0, gt->getElement(j, i + 1)) == y_test->getElement(0, j))) / double(K) * std::min(size_t(K), i + 1) / double(i + 1)
                ));
        }
    }
}

void compute_sp_knn_unroll4(
    const Matrix* x_train, const Matrix* x_test, const Matrix* y_train, 
    const Matrix* y_test, uint64_t K, std::vector<double>& mid,
    Matrix* gt, Matrix* sp) {
    KNN_unroll4(x_train, x_test, gt, mid);
    compute_single_unweighted_knn_class_shapley(x_train, y_train, gt, y_test, K, sp);
}

void compute_sp_plain(
    const Matrix* x_train, const Matrix* x_test, const Matrix* y_train, 
    const Matrix* y_test, uint64_t K, std::vector<double>& mid,
    Matrix* gt, Matrix* sp) {
    KNN(x_train, x_test, gt, mid);
    compute_single_unweighted_knn_class_shapley(x_train, y_train, gt, y_test, K, sp);
}
