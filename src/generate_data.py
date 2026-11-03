"""Generate a synthetic insurance-like dataset (gender, age, income, family_members, insurance_benefits)."""
import numpy as np, pandas as pd
from pathlib import Path

def generate(n=5000, seed=42):
    rng = np.random.default_rng(seed)
    gender = rng.integers(0, 2, n)
    age = np.clip(rng.normal(31, 8.5, n), 18, 65).round()
    income = np.clip(rng.normal(40000, 10000, n) + (age - 31) * 300, 5000, 80000).round(-2)
    family = np.clip(rng.poisson(1.2, n), 0, 6)
    score = 0.18 * (age - 40) + 0.00002 * (income - 40000) + rng.normal(0, 1.0, n)
    benefits = np.where(score > 0, np.clip(np.round(score), 1, 5), 0).astype(int)
    return pd.DataFrame({"gender": gender, "age": age.astype(int), "income": income,
                         "family_members": family, "insurance_benefits": benefits})

if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "data" / "insurance_synthetic.csv"
    generate().to_csv(out, index=False)
    print(f"wrote {out}")
