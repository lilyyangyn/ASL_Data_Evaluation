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
    auto data = load_exact_data(p.get<std::string>("-i"));
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