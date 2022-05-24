from concurrent.futures import thread
import matplotlib.pyplot as plt
from typing import Mapping, Set, NamedTuple, Tuple, List
from pathlib import Path
from math import log2
import os
import tempfile
import subprocess
import json
import threading
import shutil
class Matrix(NamedTuple):
    M: int
    N: int

    def count(self):
        return self.M * self.N

class Result(NamedTuple):
    x_train: Matrix
    x_test: Matrix
    y_train: Matrix
    y_test: Matrix
    cycles: float
    flops: float

REPEAT = 20

def search_and_execute(dir: Path, measures: Tuple[Path, Path]) -> Mapping[Path, Mapping[str, Result]]:
    ret = {}

    def _check_dir(d: Path):
        info = d / "info.txt"
        x_train = None
        x_test = None
        y_train = None
        y_test = None
        if info.exists():
            print(f"Measuring {d}...")
            with open(info, "r+") as f:
                for ln in f:
                    tks = ln.split(":")
                    if len(tks) != 2:
                        continue
                    mt = tks[0].strip()
                    sz = tks[1] # 1, 2
                    tks = list(map(lambda x: int(x.strip()), sz.split(", ")))

                    if "x_train" in mt:
                        x_train = Matrix(tks[0], tks[1])
                    elif "x_test" in mt:
                        x_test = Matrix(tks[0], tks[1])
                    elif "y_train" in mt:
                        y_train = Matrix(tks[0], tks[1])
                    elif "y_test" in mt:
                        y_test = Matrix(tks[0], tks[1])
                    
            if None not in (x_train, x_test, y_train, y_test):
                measure = measures[0]
                count =measures[1]
                r = {}
                try:
                    measure_output = json.loads(subprocess.check_output(["taskset", "-c", "1", str(measure), "-i", str(d), "-r", str(REPEAT), "-j"]))
                    count_output = json.loads(subprocess.check_output([str(count), "-i", str(d), "-r", "1", "-j"]))
                except subprocess.CalledProcessError:
                    print(f"Failed! Probably a SEGSEGV")
                    return

                for test, test_r in measure_output.items():
                    r[test] = Result(x_train, x_test, y_train, y_test, float(test_r['cycles']), float(count_output[test]['flops']))

                ret[d] = r


    def _search_impl(d: Path):
        _check_dir(d)
        for pp in os.listdir(d):
            p = d / Path(pp)
            if p.is_dir():
                _search_impl(p)
    
    _search_impl(dir)
    return ret
            
def cmake_build(source_str: Path, build_str: Path, flags: List[str]):
    subprocess.check_call(["cmake", "-S", str(source_str), "-B", str(build_str), *flags])
    subprocess.check_call(["cmake", "--build", str(build_str), "-j4"])

def prepare_builds(flags: List[str]):
    source_dir = Path(__file__).parent.parent.resolve()
    release_dir = Path(tempfile.mkdtemp())
    count_dir = Path(tempfile.mkdtemp())

    t = threading.Thread(target=cmake_build, args=[source_dir, release_dir, ["-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_CXX_FLAGS={' '.join(flags)}"]])
    t.start()
    cmake_build(source_dir, count_dir, ["-DCMAKE_BUILD_TYPE=Release", "-DCOUNT_FLOPS=yes", f"-DCMAKE_CXX_FLAGS={' '.join(flags)}"])
    t.join()
    
    return (release_dir / "measure", count_dir / "measure")

def clean_builds_dir(measures: Tuple[Path, Path]):
    tmp1 = measures[0].parent
    tmp2 = measures[1].parent
    shutil.rmtree(tmp1)
    shutil.rmtree(tmp2)

if __name__ == "__main__":
    progs = prepare_builds(["-march=native", "-O3", "-ffast-math"])
    tests_dir = Path(__file__).parent.parent / "tests"
    results = search_and_execute(tests_dir, progs)

    all_tests = []
    for p, result in results.items():
        for k in result.keys():
            if k not in all_tests:
                all_tests.append(k)
    
    print(f"Tests to plot: {all_tests}")

    # Dict is not ordered
    data: List[List[Result]] = []
    for test in all_tests:
        data.append([ result[test] for result in results.values() ])
    
    # xline -> Data in total
    # yline -> flops/cycle
    xs = []
    for d in data:
        xs.append([ log2( r.x_train.count() + r.x_test.count() + r.y_train.count() + r.y_test.count()) for r in d if r.x_train.M == 1000 and r.x_test.M == 2])
    
    ys = []
    for d in data:
        ys.append([ r.flops / r.cycles for r in d if r.x_train.M == 1000 and r.x_test.M == 2])

    for idx, test in enumerate(all_tests):
        x = xs[idx]
        y = ys[idx]
        new_x, new_y = zip(*sorted(zip(x, y), key=lambda t: t[0]))
        plt.plot(new_x, new_y, label=f"{test} x_train.M == 1000")
    
    #plt.show()
    plt.legend()
    plt.ylabel("Performance [flops/cycle]")
    plt.xlabel("log(N)")
    plt.savefig("plot1.png")

    clean_builds_dir(progs)
