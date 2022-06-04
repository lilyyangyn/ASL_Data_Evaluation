#pragma once

#include <cstdint>

class FlopsCounter {
    uint64_t ctr;
    uint64_t read;
    uint64_t write;

public:
    FlopsCounter() : ctr(0), read(0), write(0) {}

    void Increase(uint64_t v) {
        this->ctr += v;
    }

    void Reset() {
        this->ctr = 0;
    }

    uint64_t Get() {
        return this->ctr;
    }

    void IncreaseRead(uint64_t v) {
        this->read += v;
    }

    void IncreaseWrite(uint64_t v) {
        this->write += v;
    }

    void ResetRW() {
        this->read = 0;
        this->write = 0;
    }

    uint64_t GetRead() {
        return this->read;
    }

    uint64_t GetWrite() {
        return this->write;
    }
};


FlopsCounter* getCounter();