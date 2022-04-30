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
    uint64_t result;

    Benchmark(const char* n, std::function<void()> f, uint64_t r) : name(n), func(f), result(r) {}
} Benchmark;


static std::vector<Benchmark> bench;

static uint64_t Measure(const Benchmark& bench) {
    unsigned int dummy;
    auto start = __rdtscp(&dummy);

    bench.func();

    auto end = __rdtscp(&dummy);

    return end - start;
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
            printf("    \"cycles\": \"%" PRIu64 "\"\n", it->result);
            printf("  }");

            if (it + 1 != bench.end()) {
                printf(",");
            }

            printf("\n");
        }
        printf("}\n");
    } else {
        for (auto& v : bench) {
            printf("%s: %" PRIu64 " cycles\n", v.name, v.result);
        }
    }
}

}