#pragma once
#include <functional>
#include <vector>
#include <string>

namespace benchmark {

    void List();

    void Run(bool json_output, size_t repeat, const std::vector<std::string>& tests);

    void Register(const char* name, std::function<void()> f);
}