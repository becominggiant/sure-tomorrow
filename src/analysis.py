"""Linear-algebra techniques for insurance data: kNN + scaling, custom linear regression, obfuscation."""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.preprocessing import MaxAbsScaler
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, mean_squared_error, r2_score

FEATURES = ["gender", "age", "income", "family_members"]
DATA = Path(__file__).resolve().parents[1] / "data" / "insurance_synthetic.csv"


class MyLinearRegression:
    """OLS via the normal equation w = (X^T X)^-1 X^T y."""
    def fit(self, X, y):
        X1 = np.c_[np.ones(len(X)), X]
        w = np.linalg.inv(X1.T @ X1) @ X1.T @ y
        self.intercept_, self.coef_ = w[0], w[1:]
        return self

    def predict(self, X):
        return X @ self.coef_ + self.intercept_


def similar_customers(df, idx=0, k=5, scaled=True):
    X = df[FEATURES].to_numpy(float)
    if scaled:
        X = MaxAbsScaler().fit_transform(X)
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    d, i = nn.kneighbors(X[[idx]])
    return df.iloc[i[0]].assign(distance=d[0])


def knn_f1(df, seed=12345):
    X = df[FEATURES].to_numpy(float)
    y = (df.insurance_benefits > 0).astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    rows = []
    dummy = DummyClassifier(strategy="stratified", random_state=seed).fit(Xtr, ytr)
    rows.append(("dummy", "-", f1_score(yte, dummy.predict(Xte))))
    sc = MaxAbsScaler().fit(Xtr)
    for k in range(1, 11):
        for name, (a, b) in {"unscaled": (Xtr, Xte), "scaled": (sc.transform(Xtr), sc.transform(Xte))}.items():
            m = KNeighborsClassifier(n_neighbors=k).fit(a, ytr)
            rows.append((name, k, f1_score(yte, m.predict(b))))
    return pd.DataFrame(rows, columns=["model", "k", "f1"])


def regression_and_obfuscation(df, seed=12345):
    X = df[FEATURES].to_numpy(float); y = df.insurance_benefits.to_numpy(float)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed)
    mine = MyLinearRegression().fit(Xtr, ytr); sk = LinearRegression().fit(Xtr, ytr)
    p_mine, p_sk = mine.predict(Xte), sk.predict(Xte)

    rng = np.random.default_rng(seed)
    P = rng.normal(size=(X.shape[1], X.shape[1]))
    assert abs(np.linalg.det(P)) > 1e-6, "P must be invertible"
    obf = MyLinearRegression().fit(Xtr @ P, ytr)
    p_obf = obf.predict(Xte @ P)
    recovered = (Xte @ P) @ np.linalg.inv(P)
    return {
        "custom_r2": r2_score(yte, p_mine), "sklearn_r2": r2_score(yte, p_sk),
        "custom_vs_sklearn_max_diff": np.abs(p_mine - p_sk).max(),
        "obfuscated_r2": r2_score(yte, p_obf),
        "obfuscated_rmse": mean_squared_error(yte, p_obf) ** 0.5,
        "original_rmse": mean_squared_error(yte, p_mine) ** 0.5,
        "obfuscation_pred_max_diff": np.abs(p_mine - p_obf).max(),
        "recovery_max_err": np.abs(recovered - Xte).max(),
    }


if __name__ == "__main__":
    df = pd.read_csv(DATA)
    print("Nearest neighbours of customer 0 (scaled):\n", similar_customers(df), "\n")
    f1 = knn_f1(df)
    print(f1.groupby("model").f1.max().round(3), "\n")
    for k, v in regression_and_obfuscation(df).items():
        print(f"{k:30s} {v:.3e}" if v < 1e-3 else f"{k:30s} {v:.4f}")
