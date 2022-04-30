#pragma once

#include <cstdint>
#include <cstring>
#include <cstdio>

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