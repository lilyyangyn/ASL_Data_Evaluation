#include <functional>
#include <cstdint>
#include <cstdio>
#include <cinttypes>
#include <vector>
#include <x86intrin.h>

namespace benchmark {

typedef struct Benchmark {
    const char* name;
    std::function<void()> func;
    double result;

    Benchmark(const char* n, std::function<void()> f, double r) : name(n), func(f), result(r) {}
} Benchmark;


static std::vector<Benchmark> bench;
static unsigned int dummy;

static uint64_t Measure(const Benchmark& bench) {
    auto start = __rdtscp(&dummy);

    bench.func();

    auto end = __rdtscp(&dummy);

    return end - start;
}

void Register(const char* name, std::function<void()> f) {
    bench.emplace_back(name, f, 0);
}

void Run(bool json_output, size_t repeat) {
    for (auto& v : bench) {
        v.result = 0;
        for (size_t i = 0; i < repeat; i ++ ) {
            v.result += Measure(v) / double(repeat);
        }
    }

    if (json_output) {
        printf("{\n");
        for (auto it = bench.begin(); it != bench.end(); it ++) {
            printf("  \"%s\": {\n", it->name);
            printf("    \"cycles\": \"%f\"\n", it->result);
            printf("  }");

            if (it + 1 != bench.end()) {
                printf(",");
            }

            printf("\n");
        }
        printf("}\n");
    } else {
        for (auto& v : bench) {
            printf("%s: %f cycles (repeat %zd times)\n", v.name, v.result, repeat);
        }
    }
}

}