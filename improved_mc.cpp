#include <cmath>

#include "improved_mc.h"

static double knn_utility(const Matrix* y_train, double y_test_point, uint64_t K, std::vector<KNNPoint> k_nearest_points) {
    auto size = k_nearest_points.size();

    int sum = 0;
    for (size_t i = 0; i < size; i++) {
        sum += int(y_train->getElement(0, k_nearest_points[i].idx) == y_test_point);
    }
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
    for (size_t i = 0; i < num_permute; i++) {
        std::next_permutation(mid.begin(), mid.end());
        for (size_t j = 0; j < x_train_M; j++) {
            permutations->setElement(i, j, mid[j]);
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
                mid[j] += (x_train->getElement(j, k) - x_test->getElement(i, k)) * (x_train->getElement(j, k) - x_test->getElement(i, k));
            }
            mid[j] = std::sqrt(mid[j]);
        }
        for (size_t j = 0; j < N1; j++) {
            result->setElement(i, j, mid[j]);
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

    for (size_t k = 0; k < N1; k++) {
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