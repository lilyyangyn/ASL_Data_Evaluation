#include <chrono>
#include <functional>
#include <cstdint>
#include <cstdio>

namespace benchmark {

typedef struct Benchmark {
    const char* name;
    std::function<void()> func;
} Benchmark;

constexpr const int max_bench = 1024;

static Benchmark bench[max_bench];
static size_t bench_count = 0;

static double Measure(const Benchmark& bench) {
    auto start = std::chrono::steady_clock::now();

    bench.func();

    auto end = std::chrono::steady_clock::now();

    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

void Register(const char* name, std::function<void()> f) {
    bench[bench_count].name = name;
    bench[bench_count].func = f;
    bench_count++;
}

void Run(bool json_output) {
    double result[max_bench];
    for (size_t i = 0 ; i < bench_count; i ++) {
        result[i] = Measure(bench[i]);
    }

    if (json_output) {
        printf("{\n");
        for (size_t i = 0; i < bench_count; i ++) {
            printf("  \"%s\": {\n", bench[i].name);
            printf("    \"ns\": \"%f\"\n", result[i]);
            printf("  }");

            if (i != bench_count  - 1) {
                printf(",");
            }

            printf("\n");
        }
        printf("}\n");
    } else {
        for (size_t i = 0; i < bench_count; i ++) {
            printf("%s: %f nanoseconds (%f seconds)\n", bench[i].name, result[i], result[i] / 1e9);
        }
    }
}

}