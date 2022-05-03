import numpy as np
import csv
import sys
from pathlib import Path

def get_true_KNN(x_trn, x_tst):
    N = x_trn.shape[0]
    N_tst = x_tst.shape[0]
    # print(f"N={N} N_tst={N_tst}")
    x_tst_knn_gt = np.zeros((N_tst, N))
    for i_tst in range(N_tst):
        dist_gt = np.zeros(N)
        for i_trn in range(N):
            dist_gt[i_trn] = np.linalg.norm(x_trn[i_trn, :] - x_tst[i_tst, :], 2)
        # print(dist_gt)
        x_tst_knn_gt[i_tst, :] = np.argsort(dist_gt)
    return x_tst_knn_gt.astype(int)


def compute_single_unweighted_knn_class_shapley(x_trn, y_trn, x_tst_knn_gt, y_tst, K):
    N = x_trn.shape[0]
    N_tst = x_tst_knn_gt.shape[0]
    sp_gt = np.zeros((N_tst, N))
    # print(f"x_trn\n{x_trn}")
    # print(f"y_trn\n{y_trn}")
    # print(f"gt\n{x_tst_knn_gt}")
    # print(f"y_test\n{y_test}")
    for j in range(N_tst):
        sp_gt[j, x_tst_knn_gt[j, -1]] = (y_trn[x_tst_knn_gt[j, -1]] == y_tst[j]) / N
        #print(f"{x_tst_knn_gt[j, -1]} {j}")
        # print(sp_gt)
        for i in np.arange(N - 2, -1, -1):
            # print(f"{i}: {y_trn[x_tst_knn_gt[j, i + 1]]}=={y_tst[j]}")
            sp_gt[j, x_tst_knn_gt[j, i]] = sp_gt[j, x_tst_knn_gt[j, i + 1]] + \
                                           (int(y_trn[x_tst_knn_gt[j, i]] == y_tst[j]) -
                                            int(y_trn[x_tst_knn_gt[j, i + 1]] == y_tst[j])) / K * min([K, i + 1]) / (
                                                       i + 1)
    return sp_gt


def read_csv(p: Path):
    result = []
    with open(p, newline="") as f:
        r = csv.reader(f)
        for row in r:
            result.append(list(map(float, row)))
    return np.array(result)

def read_data(d: Path):

    x_test = read_csv( d / "x_test.csv" )
    x_train = read_csv( d / "x_train.csv" )
    y_test = read_csv( d / "y_test.csv" )
    y_train = read_csv( d / "y_train.csv" )

    return x_test, x_train, y_test, y_train

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"{sys.argv[0]} [path to data directory]")
        exit(-1)
    data_path = Path(sys.argv[1])
    x_test, x_train, y_test, y_train = read_data(data_path)
    y_test = y_test.flatten()
    y_train = y_train.flatten()
    gt = get_true_KNN(x_train, x_test)
    print(f"gt={gt}")
    np.savetxt(data_path / "py_gt.csv", gt, delimiter=",", fmt="%.08f")

    l = compute_single_unweighted_knn_class_shapley(x_train, y_train, gt, y_test, 1)
    print(f"l={l}")
    np.savetxt(data_path / "py_sp.csv", l, delimiter=",", fmt="%.08f")