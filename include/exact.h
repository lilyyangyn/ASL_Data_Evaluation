#pragma once

#include "matrix.h"
#include "lazycsv.hpp"
#include <vector>
#include <memory>
#include <assert.h>

typedef struct ExactInputData {
    Matrix x_train;
    Matrix x_test;
    Matrix y_train;
    Matrix y_test;

    ExactInputData(
            const double* x_train_data, size_t x_train_m, size_t x_train_n,
            const double* x_test_data, size_t x_test_m, size_t x_test_n,
            const double* y_train_data, size_t y_train_m, size_t y_train_n,
            const double* y_test_data, size_t y_test_m, size_t y_test_n) :
            x_train(x_train_data, x_train_m, x_train_n),
            x_test(x_test_data, x_test_m, x_test_n),
            y_train(y_train_data, y_train_m, y_train_n),
            y_test(y_test_data, y_test_m, y_test_n) {}
} ExactInputData;

static inline std::unique_ptr<ExactInputData> load_exact_data(const std::filesystem::path& input_directory) {
    size_t x_train_m, x_train_n, x_test_m, x_test_n, y_train_m, y_train_n, y_test_m, y_test_n;

    auto x_train = read_csv(input_directory / "x_train.csv", x_train_m, x_train_n);
    auto x_test = read_csv(input_directory / "x_test.csv", x_test_m, x_test_n);
    auto y_train = read_csv(input_directory / "y_train.csv", y_train_m, y_train_n);
    auto y_test = read_csv(input_directory / "y_test.csv", y_test_m, y_test_n);

    assert(x_train_n * x_train_m == x_train.size());
    assert(x_test_n * x_test_m == x_test.size());
    assert(y_train_n * y_train_m == y_train.size());
    assert(y_test_n * y_test_m == y_test.size());

    return std::make_unique<ExactInputData>(
            &x_train[0], x_train_m, x_train_n,
            &x_test[0], x_test_m, x_test_n,
            &y_train[0], y_train_m, y_train_n,
            &y_test[0], y_test_m, y_test_n);
}

void KNN(const Matrix* x_train, const Matrix* x_test, Matrix* gt, std::vector<double>& mid);

void compute_single_unweighted_knn_class_shapley(
    const Matrix* x_train, const Matrix* y_train, 
    const Matrix* gt, const Matrix* y_test,
    uint64_t K, Matrix* result);

void compute_sp_plain(
    const Matrix* x_train, const Matrix* x_test, const Matrix* y_train, 
    const Matrix* y_test, uint64_t K, std::vector<double>& mid,
    Matrix* gt, Matrix* sp);