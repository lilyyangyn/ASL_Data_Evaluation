#include <cmath>
#include <algorithm>
#include <random>
#include <immintrin.h>

#include "improved_mc.h"

static const double eps = 1e-6;

static double knn_utility_unroll4(const Matrix* y_train, double y_test_point, uint64_t K, const std::vector<KNNPoint>& k_nearest_points) {
    auto size = k_nearest_points.size();

    int sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
    size_t i = 0;
    for (i = 0; i + 3 < size; i+=4) {
        sum0 += int( std::fabs(y_train->getElement(0, k_nearest_points[i].idx) - y_test_point ) < eps);
        sum1 += int( std::fabs(y_train->getElement(0, k_nearest_points[i+1].idx) - y_test_point ) < eps);
        sum2 += int( std::fabs(y_train->getElement(0, k_nearest_points[i+2].idx) - y_test_point ) < eps);
        sum3 += int( std::fabs(y_train->getElement(0, k_nearest_points[i+3].idx) - y_test_point ) < eps);
    }
    for (; i < size; i++) {
        sum0 += int( std::fabs(y_train->getElement(0, k_nearest_points[i].idx) - y_test_point ) < eps);
    }
#ifdef FLOPS
    getCounter()->Increase(size * 3 + 4); // 1 -, 1 <, 1 + per loop 
                                          // and 3 adds and 1 div
                                          // fabs flops???
#endif
    return double(sum0 + sum1 + sum2 + sum3)/size;
}

static double knn_utility(const Matrix* y_train, double y_test_point, uint64_t K, const std::vector<KNNPoint>& k_nearest_points) {
    auto size = k_nearest_points.size();

    int sum = 0;
    for (size_t i = 0; i < size; i++) {
        sum += int( std::fabs(y_train->getElement(0, k_nearest_points[i].idx) - y_test_point ) < eps);
    }
#ifdef FLOPS
    getCounter()->Increase(size * 3 + 1); // 1 -, 1 <, 1 + per loop and 1 div
                                          // fabs flops???
#endif
    return double(sum)/size;
}

void random_permute(Matrix* permutations, uint64_t x_train_M, uint64_t num_permute, std::vector<uint64_t>& mid) {
    auto permutations_M = permutations->getM();
    auto permutations_N = permutations->getN();

    assert(permutations_M == num_permute);
    assert(permutations_N == x_train_M);

    for (size_t i = 0; i < x_train_M; i++) {
        mid[i] = i;
    }
    auto rng = std::default_random_engine {};
    for (size_t i = 0; i < num_permute; i++) {
        std::shuffle(std::begin(mid), std::end(mid), rng);
        for (size_t j = 0; j < x_train_M; j++) {
            permutations->setElement(i, j, mid[j]);
        }
    }
}

void point_distances_simd(const Matrix* x_train, const Matrix* x_test, Matrix *result, std::vector<double>& mid) {
    auto x_train_M = x_train->getM();
    auto x_train_N = x_train->getN();
    auto x_test_M = x_test->getM();
    auto x_test_N = x_test->getN(); 
    auto N1 = x_train_M;
    auto N2 = x_test_M;
    assert(x_test_N == x_train_N);
    assert(result->getM() == N2);
    assert(result->getN() == N1);

    __m256d zero_vec = _mm256_set1_pd(0);
    __m256d x_test_vec, val_vec, val1_vec, val2_vec, val3_vec, mid_vec, mid1_vec, mid2_vec, mid3_vec, midsum_vec, midsum1_vec;
    double* x_train_val = x_train->getVal();
    double* x_test_val = x_test->getVal();
    for (size_t i = 0; i < N2; i++) {
        size_t j = 0;      
        for (j = 0; j + 3 < N1; j+=4) {
            size_t k = 0;
            mid_vec = _mm256_set1_pd(0);
            mid1_vec = _mm256_set1_pd(0);
            mid2_vec = _mm256_set1_pd(0);
            mid3_vec = _mm256_set1_pd(0);
            for (k = 0; k+3 < x_train_N; k+=4) {
                double * idx = x_train_val + j*x_train_N + k;
                x_test_vec = _mm256_load_pd(x_test_val + i*x_test_N + k);
                val_vec = _mm256_sub_pd(_mm256_load_pd(idx), x_test_vec);
                val1_vec = _mm256_sub_pd(_mm256_load_pd(idx + x_train_N), x_test_vec);
                val2_vec = _mm256_sub_pd(_mm256_load_pd(idx + 2*x_train_N), x_test_vec);
                val3_vec = _mm256_sub_pd(_mm256_load_pd(idx + 3*x_train_N), x_test_vec);

                mid_vec = _mm256_fmadd_pd(val_vec, val_vec, mid_vec);
                mid1_vec = _mm256_fmadd_pd(val1_vec, val1_vec, mid1_vec);
                mid2_vec = _mm256_fmadd_pd(val2_vec, val2_vec, mid2_vec);
                mid3_vec = _mm256_fmadd_pd(val3_vec, val3_vec, mid3_vec);
            }
            midsum_vec = _mm256_add_pd(mid_vec, mid1_vec);
            midsum1_vec = _mm256_add_pd(mid2_vec, mid3_vec);
            _mm256_store_pd(&mid[j], _mm256_add_pd(midsum_vec, midsum1_vec));

            for (; k < x_train_N; k ++) {
                auto x_val = x_test->getElement(i, k);
                auto val = (x_train->getElement(j, k) - x_val); // avx-ed by gcc
                auto val1 = (x_train->getElement(j + 1, k) - x_val);
                auto val2 = (x_train->getElement(j + 2, k) - x_val);
                auto val3 = (x_train->getElement(j + 3, k) - x_val);
                mid[j] += val * val;
                mid[j + 1] += val1 * val1;
                mid[j + 2] += val2 * val2;
                mid[j + 3] += val3 * val3;
            }

            mid_vec = _mm256_load_pd(&mid[j]);
            _mm256_store_pd(&mid[j], _mm256_sqrt_pd(mid_vec));
#ifdef FLOPS
            getCounter()->Increase(4 + x_train_N * 3 * 4);
#endif
        }
        for (; j < N1; j++) {
            // size_t k = 0;
            // mid_vec = _mm256_set1_pd(0);
            // for (k = 0; k+3 < x_train_N; k+=4) {
            //     double * idx = x_train_val + j*x_train_N + k;
            //     x_test_vec = _mm256_load_pd(x_test_val + i*x_test_N + k);
            //     val_vec = _mm256_sub_pd(_mm256_load_pd(idx), x_test_vec);
            //     mid_vec = _mm256_fmadd_pd(val_vec, val_vec, mid_vec);
            // }
            // _mm256_store_pd(&mid[j], mid_vec);

            // for (; k < x_train_N; k ++) {
            //     auto x_val = x_test->getElement(i, k);
            //     auto val = (x_train->getElement(j, k) - x_val); // avx-ed by gcc
            //     mid[j] += val * val;
            // }
            for (size_t k = 0; k < x_train_N; k ++) {
                auto val = (x_train->getElement(j, k) - x_test->getElement(i, k));
                mid[j] +=  val * val;
            }
            mid[j] = std::sqrt(mid[j]);
#ifdef FLOPS
            getCounter()->Increase(1 + x_train_N * 3);
#endif
        }
        for (size_t j = 0; j < N1; j++) {
            result->setElement(i, j, mid[j]);
        }
    }
}

void point_distances_unroll4(const Matrix* x_train, const Matrix* x_test, Matrix *result, std::vector<double>& mid) {
    auto x_train_M = x_train->getM();
    auto x_train_N = x_train->getN();
    auto x_test_M = x_test->getM();
    auto x_test_N = x_test->getN(); 
    auto N1 = x_train_M;
    auto N2 = x_test_M;
    assert(x_test_N == x_train_N);
    assert(result->getM() == N2);
    assert(result->getN() == N1);

    for (size_t i = 0; i < N2; i++) {
        size_t j = 0;
        // optimized as memset by gcc
        for (j = 0; j < N1; j++) {
            mid[j] = 0;
        }
        for (j = 0; j + 3 < N1; j+=4) {
            for (size_t k = 0; k < x_train_N; k ++) {
                auto x_val = x_test->getElement(i, k);
                auto val = (x_train->getElement(j, k) - x_val); // avx-ed by gcc
                auto val1 = (x_train->getElement(j + 1, k) - x_val);
                auto val2 = (x_train->getElement(j + 2, k) - x_val);
                auto val3 = (x_train->getElement(j + 3, k) - x_val);
                mid[j] += val * val;
                mid[j + 1] += val1 * val1;
                mid[j + 2] += val2 * val2;
                mid[j + 3] += val3 * val3;
            }
            mid[j] = std::sqrt(mid[j]); // optmized by sqrt by gcc
            mid[j + 1] = std::sqrt(mid[j + 1]);
            mid[j + 2] = std::sqrt(mid[j + 2]);
            mid[j + 3] = std::sqrt(mid[j + 3]);
#ifdef FLOPS
            getCounter()->Increase(4 + x_train_N * 3 * 4);
#endif
        }
        for (; j < N1; j++) {
            for (size_t k = 0; k < x_train_N; k ++) {
                auto val = (x_train->getElement(j, k) - x_test->getElement(i, k));
                mid[j] +=  val * val;
            }
            mid[j] = std::sqrt(mid[j]);
#ifdef FLOPS
            getCounter()->Increase(1 + x_train_N * 3);
#endif
        }
        for (size_t j = 0; j < N1; j++) {
            result->setElement(i, j, mid[j]);
        }
    }
}

void point_distances(const Matrix* x_train, const Matrix* x_test, Matrix *result, std::vector<double>& mid) {
    auto x_train_M = x_train->getM();
    auto x_train_N = x_train->getN();
    auto x_test_M = x_test->getM();
    auto x_test_N = x_test->getN(); 
    auto N1 = x_train_M;
    auto N2 = x_test_M;
    assert(x_test_N == x_train_N);
    assert(result->getM() == N2);
    assert(result->getN() == N1);

    for (size_t i = 0; i < N2; i++) {
        for (size_t j = 0; j < N1; j++) {
            mid[j] = 0;
        }
        for (size_t j = 0; j < N1; j++) {
            for (size_t k = 0; k < x_train_N; k ++) {
                double scalar = x_train->getElement(j, k) - x_test->getElement(i, k);
                mid[j] += scalar * scalar;
            }
            mid[j] = std::sqrt(mid[j]);
#ifdef FLOPS
            getCounter()->Increase(x_train_N * 3 + 1);
#endif
        }
        for (size_t j = 0; j < N1; j++) {
            result->setElement(i, j, mid[j]);
        }
    }
}

void improved_single_unweighted_knn_class_shapley_unroll4(
    const Matrix* y_train, const Matrix* y_test,
    const Matrix* permutations, const Matrix* distances, 
    uint64_t K, uint64_t num_permute, Matrix* result, 
    Matrix* phi, FixedSizeKNNHeap* H) { 
    auto N1 = y_train->getN();
    auto N2 = y_test->getN();

    assert(result->getM() == N2);
    assert(result->getN() == N1);
    assert(phi->getM() == num_permute);
    assert(phi->getN() == N1);
    assert(H->getMaxSize() == K);

    for (size_t k = 0; k < N2; k++) {
        phi->resetVal();
        for (size_t t = 0; t < num_permute; t++) {
            H->popAll();
            double prev_utility = 0;
            for (size_t i = 0; i < N1; i++) {
                // insert permutation to the heap
                auto cur_idx = permutations->getElement(t, i);
                bool modified = H->push(KNNPoint(cur_idx, distances->getElement(k, cur_idx)));
                if (modified) {
                    double utility = knn_utility_unroll4(y_train, y_test->getElement(0, k), K, H->getAllItem());
                    phi->setElement(t, cur_idx, utility - prev_utility);
                    prev_utility = utility;
                } else {
                    phi->setElement(t, cur_idx, 
                        phi->getElement(t, permutations->getElement(t, i-1)));
                }
            }
        }

        for (size_t i = 0; i < N1; i++) {
            double sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
            size_t t = 0;
            for (t = 0; t + 3 < num_permute; t+=4) {
                sum0 += phi->getElement(t, i);
                sum1 += phi->getElement(t, i+1);
                sum2 += phi->getElement(t, i+2);
                sum3 += phi->getElement(t, i+3);
            }
            for (; t < num_permute; t++) {
                sum0 += phi->getElement(t, i);
            }
            result->setElement(k, i, (sum0+sum1+sum2+sum3)/num_permute);
        } 
    }
}

void improved_single_unweighted_knn_class_shapley(
    const Matrix* y_train, const Matrix* y_test,
    const Matrix* permutations, const Matrix* distances, 
    uint64_t K, uint64_t num_permute, Matrix* result, 
    Matrix* phi, FixedSizeKNNHeap* H) { 
    auto N1 = y_train->getN();
    auto N2 = y_test->getN();

    assert(result->getM() == N2);
    assert(result->getN() == N1);
    assert(phi->getM() == num_permute);
    assert(phi->getN() == N1);
    assert(H->getMaxSize() == K);

    for (size_t k = 0; k < N2; k++) {
        phi->resetVal();
        for (size_t t = 0; t < num_permute; t++) {
            H->popAll();
            double prev_utility = 0;
            for (size_t i = 0; i < N1; i++) {
                // insert permutation to the heap
                auto cur_idx = permutations->getElement(t, i);
                bool modified = H->push(KNNPoint(cur_idx, distances->getElement(k, cur_idx)));
                if (modified) {
                    double utility = knn_utility(y_train, y_test->getElement(0, k), K, H->getAllItem());
                    phi->setElement(t, cur_idx, utility - prev_utility);
                    prev_utility = utility;
                } else {
                    phi->setElement(t, cur_idx, 
                        phi->getElement(t, permutations->getElement(t, i-1)));
                    // phi->setElement(t, cur_idx, 
                    //     phi->getElement(t, 0));
                }
            }
        }

        for (size_t i = 0; i < N1; i++) {
            double sum = 0;
            for (size_t t = 0; t < num_permute; t++) {
                sum += phi->getElement(t, i);
            }
            result->setElement(k, i, sum/num_permute);
        } 
    }
}

void compute_sp_improved_mc(
    const Matrix* x_train, const Matrix* x_test, const Matrix* y_train, 
    const Matrix* y_test, uint64_t K, uint64_t num_permutes, Matrix* permutations, Matrix* point_dists, 
    Matrix* sp, std::vector<uint64_t>& mid1, std::vector<double>& mid2, Matrix* phi, FixedSizeKNNHeap* H) {
    random_permute(permutations, x_train->getM(), num_permutes, mid1);
    point_distances(x_train, x_test, point_dists, mid2);
    improved_single_unweighted_knn_class_shapley(y_train, y_test, permutations, point_dists, K, num_permutes, sp, phi, H);
}

void compute_sp_improved_mc_unroll4(
    const Matrix* x_train, const Matrix* x_test, const Matrix* y_train, 
    const Matrix* y_test, uint64_t K, uint64_t num_permutes, Matrix* permutations, Matrix* point_dists, 
    Matrix* sp, std::vector<uint64_t>& mid1, std::vector<double>& mid2, Matrix* phi, FixedSizeKNNHeap* H) {
    random_permute(permutations, x_train->getM(), num_permutes, mid1);
    point_distances_unroll4(x_train, x_test, point_dists, mid2);
    improved_single_unweighted_knn_class_shapley_unroll4(y_train, y_test, permutations, point_dists, K, num_permutes, sp, phi, H);
}

void compute_sp_improved_mc_simd(
    const Matrix* x_train, const Matrix* x_test, const Matrix* y_train, 
    const Matrix* y_test, uint64_t K, uint64_t num_permutes, Matrix* permutations, Matrix* point_dists, 
    Matrix* sp, std::vector<uint64_t>& mid1, std::vector<double>& mid2, Matrix* phi, FixedSizeKNNHeap* H) {
    random_permute(permutations, x_train->getM(), num_permutes, mid1);
    point_distances_simd(x_train, x_test, point_dists, mid2);
    improved_single_unweighted_knn_class_shapley_unroll4(y_train, y_test, permutations, point_dists, K, num_permutes, sp, phi, H);
}