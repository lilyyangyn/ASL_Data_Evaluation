#include <chrono>
#include <functional>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace benchmark {

typedef struct Benchmark {
    const char* name;
    std::function<void()> func;
    double result;

    Benchmark(const char* n, std::function<void()> f, double r) : name(n), func(f), result(r) {}
} Benchmark;


static std::vector<Benchmark> bench;

static double Measure(const Benchmark& bench) {
    auto start = std::chrono::steady_clock::now();

    bench.func();

    auto end = std::chrono::steady_clock::now();

    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

void Register(const char* name, std::function<void()> f) {
    bench.emplace_back(name, f, 0);
}

void Run(bool json_output) {
    for (auto& v : bench) {
        v.result = Measure(v);
    }

    if (json_output) {
        printf("{\n");
        for (auto it = bench.begin(); it != bench.end(); it ++) {
            printf("  \"%s\": {\n", it->name);
            printf("    \"ns\": \"%f\"\n", it->result);
            printf("  }");

            if (it + 1 != bench.end()) {
                printf(",");
            }

            printf("\n");
        }
        printf("}\n");
    } else {
        for (auto& v : bench) {
            printf("%s: %f nanoseconds (%f seconds)\n", v.name, v.result, v.result / 1e9);
        }
    }
}

}