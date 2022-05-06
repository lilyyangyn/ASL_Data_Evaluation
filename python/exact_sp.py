import numpy as np
import csv
from pathlib import Path
import argparse
import os


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
        sorted = np.argsort(dist_gt)
        x_tst_knn_gt[i_tst, :] = sorted

        # s = f"{i_tst}:"
        # for it in sorted.flat:
        #     s += f"{it} "
        # print(s, file=sys.stderr)
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
    parser = argparse.ArgumentParser(description='Generate data.')
    parser.add_argument('data_path', help='Path to data directory.')
    
    parser.add_argument('--sizes', metavar='N', type=int, nargs=3,
                    help='Size of the matrix: train_m, test_m, x_n')

    args = parser.parse_args()
    data_path = Path(args.data_path)
    # print(args.data_path)
    # print(args.sizes)

    if args.sizes != None:
        n1 = args.sizes[0]
        n2 = args.sizes[1]
        n = args.sizes[2]
        x_train = np.ndarray(shape=(n1, n), dtype=float, buffer=np.random.uniform(0,50,[n1*n]))
        # x_test = np.ndarray(shape=(n2, n), dtype=float, buffer=np.random.uniform(0,50,[n2*n]))
        x_test = np.zeros(shape=(n2, n), dtype=float)
        y_train = np.random.uniform(0,100,[n1])
        test_idx = np.random.randint(0, 2*n1, [n2])
        y_test = np.zeros(shape=(n2))
        for i in range(n2):
            if test_idx[i] > n1 - 1:
                # np.vstack([x_test, np.random.uniform(0,100,[n])])
                x_test[i] = np.random.uniform(0,50,[n])
                y_test[i] = np.random.uniform(0,100)
            else:
                # np.vstack([x_test, x_train[test_idx[i]]])
                x_test[i] = x_train[test_idx[i]]
                y_test[i] = y_train[test_idx[i]]
        # y_test = np.random.uniform(0,100,[n2])

        if not os.path.exists(data_path):
            os.makedirs(data_path)

        with open(data_path / "info.txt", "a") as f:
            f.write("x_train:    {}, {}\n".format(n1, n))
            f.write("x_test:     {}, {}\n".format(n2, n))
            f.write("y_train:    {}, {}\n".format(1, n1))
            f.write("y_test:     {}, {}\n".format(1, n2))

        np.savetxt(data_path / "x_train.csv", x_train, delimiter=", ", fmt="%.08f")
        np.savetxt(data_path / "x_test.csv", x_test, delimiter=", ", fmt="%.08f")
        np.savetxt(data_path / "y_train.csv", np.atleast_2d(y_train), delimiter=", ", fmt="%.08f")
        np.savetxt(data_path / "y_test.csv", np.atleast_2d(y_test), delimiter=", ", fmt="%.08f")
    
    x_test, x_train, y_test, y_train = read_data(data_path)
    y_test = y_test.flatten()
    y_train = y_train.flatten()
    
    gt = get_true_KNN(x_train, x_test)
    print(f"gt={gt}")
    np.savetxt(data_path / "knn_gt.csv", gt, delimiter=", ", fmt="%.08f")

    l = compute_single_unweighted_knn_class_shapley(x_train, y_train, gt, y_test, 1)
    print(f"l={l}")
    np.savetxt(data_path / "sp_gt.csv", l, delimiter=", ", fmt="%.08f")