#pragma once
#include <functional>

namespace benchmark {
    void Run(bool json_output, size_t repeat);

    void Register(const char* name, std::function<void()> f);
}