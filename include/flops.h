#pragma once

#include <cstdint>

class FlopsCounter {
    uint64_t ctr;

public:
    FlopsCounter() : ctr(0) {}

    void Increase(uint64_t v) {
        this->ctr += v;
    }

    void Reset() {
        this->ctr = 0;
    }

    uint64_t Get() {
        return this->ctr;
    }
};


FlopsCounter* getCounter();