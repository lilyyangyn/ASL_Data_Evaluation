#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include <assert.h>
#include <sstream>
#include <exception>
#include <memory>
#include <filesystem>

#include "argparse.hpp"
#include "lazycsv.hpp"
#include "benchmark.h"
#include "matrix.h"
#include "exact.h"


typedef struct InputData {
    Matrix x_train;
    Matrix x_test;
    Matrix y_train;
    Matrix y_test;

    InputData(
            const double* x_train_data, size_t x_train_m, size_t x_train_n,
            const double* x_test_data, size_t x_test_m, size_t x_test_n,
            const double* y_train_data, size_t y_train_m, size_t y_train_n,
            const double* y_test_data, size_t y_test_m, size_t y_test_n) :
            x_train(x_train_data, x_train_m, x_train_n),
            x_test(x_test_data, x_test_m, x_test_n),
            y_train(y_train_data, y_train_m, y_train_n),
            y_test(y_test_data, y_test_m, y_test_n) {}
} InputData;

static std::vector<double> read_csv(const std::filesystem::path& path, size_t& M, size_t& N) {
    std::vector<double> data;
    lazycsv::parser<lazycsv::mmap_source, lazycsv::has_header<false>> p {path.string()};
    M = 0;
    N = 0;

    for (const auto row : p) {
        M++;
        N = 0;
        for (const auto cell : row) {
            N++;
            auto it = cell.raw();
            data.push_back(std::stod(std::string(it)));
        }
    }

    return data;
}

static std::unique_ptr<InputData> load_data(const std::filesystem::path& input_directory) {
    size_t x_train_m, x_train_n, x_test_m, x_test_n, y_train_m, y_train_n, y_test_m, y_test_n;

    auto x_train = read_csv(input_directory / "x_train.csv", x_train_m, x_train_n);
    auto x_test = read_csv(input_directory / "x_test.csv", x_test_m, x_test_n);
    auto y_train = read_csv(input_directory / "y_train.csv", y_train_m, y_train_n);
    auto y_test = read_csv(input_directory / "y_test.csv", y_test_m, y_test_n);

    assert(x_train_n * x_train_m == x_train.size());
    assert(x_test_n * x_test_m == x_test.size());
    assert(y_train_n * y_train_m == y_train.size());
    assert(y_test_n * y_test_m == y_test.size());

    return std::make_unique<InputData>(
            &x_train[0], x_train_m, x_train_n,
            &x_test[0], x_test_m, x_test_n,
            &y_train[0], y_train_m, y_train_n,
            &y_test[0], y_test_m, y_test_n);
}

// a = np.array([[0.47069075, 0.06548475],
//     [0.12246441, 0.57838926],
//     [0.98473347, 0.55588644]])
// b = np.array([[0.72012919, 0.04385545],
//         [0.42811407, 0.74712948],
//         [0.23655954, 0.43509146],
//         [0.12540547, 0.9914887 ]])

int main(int argc, char** argv) {
    argparse::ArgumentParser p("exact");

    p.add_argument("-j", "--json").default_value(false).implicit_value(true).help("Json output");
    p.add_argument("-r", "--repeat").default_value(1UL).scan<'u', size_t>().help("Repeat times");
    p.add_argument("-v", "--verbose").default_value(false).implicit_value(true).help("Print input data and result");
    p.add_argument("-i", "--input").required().help("The input directory");

    try {
        p.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::stringstream ss;

        ss << err.what();
        ss << p;

        printf("%s\n", ss.str().c_str());
        exit(-1);
    }
    auto verbose = p.get<bool>("-v");
    auto data = load_data(p.get<std::string>("-i"));
    Matrix gt(data->x_test.getM(), data->x_train.getM());
    Matrix sp(gt.getM(), gt.getN());
    std::vector<double> mid;
    mid.resize(data->x_train.getM());

    if (verbose) {
        data->x_train.pprint("x_train");
        data->x_test.pprint("x_test");
        data->y_train.pprint("y_train");
        data->y_test.pprint("y_test");
    }

    benchmark::Register("exact_sp_plain", std::bind(compute_sp_plain, &data->x_train, &data->x_test, &data->y_train, &data->y_test, 1, mid, &gt, &sp));

    benchmark::Run(p.get<bool>("-j"), p.get<size_t>("-r"));

    if (verbose) {
        gt.pprint("gt");
        sp.pprint("sp");
    }

    return 0;
}