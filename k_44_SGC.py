# https://graphsinspace.net
# https://tigraphs.pmf.uns.ac.rs

# SGC (Simplifying Graph Convolution)

# GRASP + kvantna regresija, deterministički, strukturno
# Varijanta: SGC – Simplified GCN (samo propagacija A_norm^K @ X, bez nelinearnosti)
 
"""
Graphs in Space: Graph Embeddings for Machine Learning on Complex Data.
SGC (Simplifying Graph Convolution): Z = A_norm^K @ X, bez ReLU između slojeva.
X = Laplacian prvih k; K = broj koraka propagacije.
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd

from itertools import combinations

from qiskit_machine_learning.utils import algorithm_globals
from qiskit.circuit.library import ZZFeatureMap, TwoLocal
from qiskit.quantum_info import Statevector, Pauli

CSV_PATH = "/Users/4c/Desktop/GHQ/data/loto7hh_4580_k21.csv"

df = pd.read_csv(CSV_PATH)
print()
print(df)
print()

SEED = 39
np.random.seed(SEED)
algorithm_globals.random_seed = SEED

EMBED_DIM = 3   # 3 dimenzije za embedding 
MAX_EPOCHS = 20 # 20 epoha za fitanje
LR = 0.2        # 0.2 learning rate
FD_EPS = 1e-3   # 1e-3 finite difference epsilon

SGC_K = 3 # 3 steps for SGC


def load_draws(csv_path=CSV_PATH):
    df = pd.read_csv(csv_path, encoding="utf-8")
    expected_cols = [f"Num{i}" for i in range(1, 8)]
    for c in expected_cols:
        if c not in df.columns:
            raise ValueError(f"Nedostaje kolona {c} u CSV fajlu.")
    draws = []
    for _, row in df.iterrows():
        nums = [int(row[f"Num{i}"]) for i in range(1, 8)]
        nums_sorted = sorted(nums)
        draws.append(nums_sorted)
    return draws


def compute_cooccurrence_matrix(draws):
    M = np.zeros((40, 40), dtype=np.int64)
    for draw in draws:
        for i_idx in range(len(draw)):
            for j_idx in range(i_idx + 1, len(draw)):
                a = draw[i_idx]
                b = draw[j_idx]
                M[a, b] += 1
                M[b, a] += 1
    return M


def compute_sgc_embeddings(M, k=EMBED_DIM):
    A = M[1:40, 1:40].astype(float)
    A_tilde = A + np.eye(39)
    d = A_tilde.sum(axis=1)
    d_inv_sqrt = np.zeros_like(d)
    d_inv_sqrt[d > 0] = 1.0 / np.sqrt(d[d > 0])
    A_norm = np.diag(d_inv_sqrt) @ A_tilde @ np.diag(d_inv_sqrt)
    L = np.diag(M.sum(axis=1)) - M
    L_sub = L[1:40, 1:40]
    _, vecs = np.linalg.eigh(L_sub)
    X = vecs[:, :k].astype(float)
    Z = X.copy()
    for _ in range(SGC_K):
        Z = A_norm @ Z
    emb = Z
    for d in range(k):
        col = emb[:, d]
        min_v, max_v = col.min(), col.max()
        if max_v - min_v > 0:
            emb[:, d] = (col - min_v) / (max_v - min_v) * np.pi
        else:
            emb[:, d] = 0.0
    return emb


def structural_target_from_graph(M):
    degrees = M.sum(axis=1)
    deg_sub = degrees[1:40].astype(float)
    min_v = deg_sub.min()
    max_v = deg_sub.max()
    if max_v - min_v > 0:
        deg_sub = (deg_sub - min_v) / (max_v - min_v)
    else:
        deg_sub = np.zeros_like(deg_sub)
    return deg_sub


class QuantumRegressor:
    def __init__(self, num_features: int):
        self.num_features = num_features
        self.feature_map = ZZFeatureMap(feature_dimension=num_features, reps=1)
        self.ansatz = TwoLocal(
            num_qubits=num_features,
            rotation_blocks="ry",
            entanglement_blocks="cz",
            reps=1,
            insert_barriers=False,
        )
        self.observable = Pauli("Z" * num_features)
        self.num_params = len(self.ansatz.parameters)
        self.theta = np.zeros(self.num_params, dtype=float)
        self.base_circuit = self.feature_map.compose(self.ansatz)

    def _predict_single(self, x_vec, theta_vec):
        param_bind = {}
        for p, val in zip(self.feature_map.parameters, x_vec):
            param_bind[p] = float(val)
        for p, val in zip(self.ansatz.parameters, theta_vec):
            param_bind[p] = float(val)
        bound = self.base_circuit.assign_parameters(param_bind, inplace=False)
        sv = Statevector.from_instruction(bound)
        exp = np.real(sv.expectation_value(self.observable))
        n = self.num_features
        norm_exp = (exp + n) / (2.0 * n)
        return float(norm_exp)

    def predict(self, X):
        preds = [self._predict_single(x, self.theta) for x in X]
        return np.array(preds, dtype=float)

    def _loss(self, theta_vec, X, y):
        preds = [self._predict_single(x, theta_vec) for x in X]
        preds = np.array(preds, dtype=float)
        diff = preds - y
        return float(np.mean(diff * diff))

    def fit(self, X, y, epochs=MAX_EPOCHS, lr=LR, fd_eps=FD_EPS):
        theta = self.theta.copy()
        for _ in range(epochs):
            grad = np.zeros_like(theta)
            for j in range(len(theta)):
                orig = theta[j]
                theta[j] = orig + fd_eps
                loss_plus = self._loss(theta, X, y)
                theta[j] = orig - fd_eps
                loss_minus = self._loss(theta, X, y)
                theta[j] = orig
                grad[j] = (loss_plus - loss_minus) / (2.0 * fd_eps)
            theta = theta - lr * grad
        self.theta = theta


def greedy_best_combo(pred_scores, M):
    order = sorted(range(1, 40), key=lambda i: pred_scores[i], reverse=True)
    chosen = [order[0]]
    while len(chosen) < 7:
        best_candidate = None
        best_value = None
        for cand in order:
            if cand in chosen:
                continue
            value = pred_scores[cand]
            for c in chosen:
                value += M[cand, c]
            if best_value is None or value > best_value:
                best_value = value
                best_candidate = cand
        chosen.append(best_candidate)
    chosen.sort()
    return tuple(chosen)


def main():
    draws = load_draws()
    M = compute_cooccurrence_matrix(draws)
    emb = compute_sgc_embeddings(M, k=EMBED_DIM)

    x_train = emb
    y_train = structural_target_from_graph(M)

    qreg = QuantumRegressor(num_features=EMBED_DIM)
    qreg.fit(x_train, y_train)

    y_pred = qreg.predict(x_train)
    pred_scores = {i: float(y_pred[i - 1]) for i in range(1, 40)}
    best_combo = greedy_best_combo(pred_scores, M)

    print()
    print("Predikcija (SGC + kvantna regresija, deterministički, strukturno):")
    print(best_combo)
    print()
    print("Score:", pred_scores[best_combo[0]])
    print()
    """
    Predikcija (SGC + kvantna regresija, deterministički, strukturno):
    (8, 11, 23, 26, 33, 34, 35)

    Score: 0.5076098596486648
    """


if __name__ == "__main__":
    main()

"""
SGC (Simplifying Graph Convolution):

A_norm: simetrično normalizovana matrica susedstva (A + I), D^{-1/2}(A+I)D^{-1/2}.
Početne feature: X = prvih k sopstvenih vektora Laplaciana (39×k).
Propagacija: Z = A_norm^SGC_K @ X, bez nelinearnosti (SGC_K = 3).

Normalizacija u [0, π] (39×k).
"""
