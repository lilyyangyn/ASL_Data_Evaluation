#include <functional>
#include <cstdint>
#include <cstdio>
#include <cinttypes>
#include <vector>
#include <x86intrin.h>
#include <string>
#include "flops.h"

namespace benchmark {

typedef struct Benchmark {
    const char* name;
    std::function<void()> func;
    double result;
#ifdef FLOPS
    uint64_t flops;
#endif

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

void List() {
    printf("All compiled test:\n");
    for (auto& v : bench) {
        printf("%s\n", v.name);
    }
}

void Run(bool json_output, size_t repeat, const std::vector<std::string>& tests) {
    std::vector<Benchmark> run;

    if (tests.size() == 0) {
        run.assign(bench.begin(), bench.end());
    } else {
        bool found = false;
        for (auto& v : bench) {
            for (auto& t : tests) {
                if (v.name == t) {
                    run.emplace_back(v);
                    break;
                }
            }
        }
    }

    for (auto& v : run) {
        v.result = 0;
#ifdef FLOPS
        getCounter()->Reset();
#endif
        for (size_t i = 0; i < repeat; i ++ ) {

            v.result += Measure(v) / double(repeat);
        }
#ifdef FLOPS
        v.flops = getCounter()->Get();
#endif
    }

    if (json_output) {
        printf("{\n");
        for (auto it = run.begin(); it != run.end(); it ++) {
            printf("  \"%s\": {\n", it->name);
            printf("    \"cycles\": \"%f\"", it->result);
#ifdef FLOPS
            printf(",\n");
            printf("    \"flops\": \"%ld\"\n", it->flops);
#else
            printf("\n");
#endif
            printf("  }");

            if (it + 1 != run.end()) {
                printf(",");
            }

            printf("\n");
        }
        printf("}\n");
    } else {
        for (auto& v : run) {
#ifdef FLOPS
            printf("%s: %f cycles (repeat %zd times) %ld flops\n", v.name, v.result, repeat, v.flops);
#else
            printf("%s: %f cycles (repeat %zd times)\n", v.name, v.result, repeat);
#endif
        }
    }
}

}