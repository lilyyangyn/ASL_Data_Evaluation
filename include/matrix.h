#pragma once

#include <cstdint>
#include <cstring>
#include <cstdio>
#include <vector>
#include <filesystem>
#include "lazycsv.hpp"
#include <assert.h>
#include <fstream>
#include "flops.h"

class DoubleProxy {
    double d;

public:
    DoubleProxy(double val) :  d(val) {}

    double get() {
        return d;
    }

    operator double() {
        return this->get();
    }

    DoubleProxy& operator*(const DoubleProxy& other) {
        return this->operator*(other.d);
    }

    DoubleProxy& operator*(double other) {
        getCounter()->Increase(1);
        d *= other;
        return *this;
    }

    DoubleProxy& operator+(const DoubleProxy& other) {
        return this->operator+(other.d);
    }

    DoubleProxy& operator+(double other) {
        getCounter()->Increase(1);
        d += other;
        return *this;
    }

    DoubleProxy& operator-() {
        // Count this???
        getCounter()->Increase(1);
        d = -d;
        return *this;
    }

    DoubleProxy& operator-(const DoubleProxy& other) {
        return this->operator-(other.d);
    }

    DoubleProxy& operator-(double other) {
        getCounter()->Increase(1);
        d -= other;
        return *this;
    }

    DoubleProxy& operator/(const DoubleProxy& other) {
        return this->operator/(other.d);
    }

    DoubleProxy& operator/(double other) {
        getCounter()->Increase(1);
        d /= other;
        return *this;
    }

};

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

    Matrix(const std::vector<double>& d, uint64_t vm, uint64_t vn) : Matrix(vm, vn) {
        assert(d.size() >= vm * vn);
        memcpy(this->val, &d[0], sizeof(double) * vn * vm);
    }

    Matrix(const Matrix&) = delete;
    Matrix(Matrix&&) = delete;

    ~Matrix() {
        delete[] this->val;
    }

    double* getVal() const {
        return this->val;
    }

#ifndef FLOPS
    double getElement(size_t i, size_t j) const {
        return this->val[i * this->n + j];
    }

    void setElement(size_t i, size_t j, double d) {
        this->val[i*this->n + j] = d;
    }
#else
    DoubleProxy getElement(size_t i, size_t j) const {
        return DoubleProxy(this->val[i * this->n + j]);
    }

    void setElement(size_t i, size_t j, DoubleProxy d) {
        this->val[i*this->n + j] = d.get();
    }

    void setElement(size_t i, size_t j, double d) {
        this->val[i*this->n + j] = d;
    }
#endif
    size_t getN() const {
        return this->n;
    }

    size_t getM() const {
        return this->m;
    }

    void resetVal() {
        size_t sz = sizeof(double) * n * m;
        memset(this->val, 0, sz);
    }

    void pprint(const char* matrix_name = nullptr) const {
        if (matrix_name) {
            printf("%s:\n", matrix_name);
        }
        printf("--------[%ldx%ld]\n", this->m, this->n);
        for (size_t i = 0; i < this->m; i ++) {
            for (size_t j = 0; j < this->n; j ++) {
                printf("%f ", double(this->getElement(i, j)));
            }
            printf("\n");
        }
        printf("--------\n");
    }
};

static inline std::vector<double> read_csv(const std::filesystem::path& path, size_t& M, size_t& N) {
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

static void write_csv(const std::filesystem::path& path, Matrix* matrix) {
    std::ofstream out(path);

    size_t M = matrix->getM();
    size_t N = matrix->getN();
    // double* val = matrix->getVal();
    for (size_t m = 0; m < M; m++) {
        for (size_t n = 0; n < N - 1; n++) {
            out << std::fixed << std::setprecision(8) << matrix->getElement(m, n) << ",";
        }
        out << std::fixed << std::setprecision(8) << matrix->getElement(m, N-1);
        if (m != M-1) {
            out << "\n";
        }
    }
}