#pragma once

#include "matrix.h"
#include <vector>

void KNN(const Matrix* x_train, const Matrix* x_test, Matrix* gt, std::vector<double>& mid);

void compute_single_unweighted_knn_class_shapley(
    const Matrix* x_train, const Matrix* y_train, 
    const Matrix* gt, const Matrix* y_test,
    uint64_t K, Matrix* result);

void compute_sp_plain(
    const Matrix* x_train, const Matrix* x_test, const Matrix* y_train, 
    const Matrix* y_test, uint64_t K, std::vector<double>& mid,
    Matrix* gt, Matrix* sp);