#pragma once

#include <vector>

#include "exact.h"

struct KNNPoint {
    uint64_t idx;
    double distance;

    KNNPoint(uint64_t idx, double distance) : idx(idx), distance(distance) {}

    bool operator<(const KNNPoint& rhs) const
    {
        return distance < rhs.distance;
    }
    // bool operator>(const KNNPoint& rhs) const
    // {
    //     return distance > rhs.distance;
    // }
};

class FixedSizeKNNHeap {
    public:
        FixedSizeKNNHeap() : max_size(0) {}
        FixedSizeKNNHeap(uint64_t size) : max_size(size) {}

        bool push(const KNNPoint item) {
            bool modified = true;
            if (myheap.size() < max_size) {
                myheap.push_back(item);
                std::push_heap(myheap.begin(), myheap.end(), cmp);
            } else {
                if (item.distance < myheap.front().distance) {
                    std::pop_heap(myheap.begin(), myheap.end(), cmp);
                    myheap.pop_back();
                    myheap.push_back(item);
                    std::push_heap(myheap.begin(), myheap.end(), cmp);
                } else {
                    modified = false;
                }
            }
            return modified;
        }

        void pop() {
            if (myheap.empty()) {
                return;
            }
            std::pop_heap(myheap.begin(), myheap.end(), cmp);
            myheap.pop_back();
        }

        std::vector<KNNPoint> getAllItem() {
            return myheap;
        }

    
    protected:
        uint64_t max_size;
        // std::priority_queue<KNNPoint, std::vector<KNNPoint>, std::greater<KNNPoint> > min_heap;
        std::vector<KNNPoint> myheap;
        std::less<KNNPoint> cmp;
};

void random_permute(Matrix* permutations, uint64_t x_train_M, uint64_t num_permute);

void point_distances(const Matrix* x_train, const Matrix* x_test, Matrix *result);

void improved_single_unweighted_knn_class_shapley(
    const Matrix* y_train, const Matrix* y_test,
    const Matrix* permutations, const Matrix* distances, 
    uint64_t K, uint64_t num_permute, Matrix* result);

void compute_sp_improved_mc(
    const Matrix* x_train, const Matrix* x_test, const Matrix* y_train, 
    const Matrix* y_test, uint64_t K, uint64_t num_permutes, Matrix* permutations, Matrix* point_dists, 
    Matrix* sp);