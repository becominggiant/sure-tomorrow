# Sure Tomorrow — Linear Algebra for Insurance Data

Portfolio project inspired by TripleTen Sprint 11 (Linear Algebra) coursework. This is an original reimplementation on a **synthetic** dataset; it is not affiliated with or endorsed by TripleTen.

## Problem
A fictional insurer, Sure Tomorrow, wants to:
1. Find customers similar to a given customer (marketing).
2. Predict whether a customer will receive an insurance benefit.
3. Predict the number of benefits a customer will receive.
4. Protect personal data by obfuscation without hurting model quality.

Schema (`data/insurance_synthetic.csv`, 5,000 rows): `gender`, `age`, `income`, `family_members`, `insurance_benefits`.

## Methods
- **MaxAbsScaler** + **NearestNeighbors** for similar-customer lookup.
- **KNeighborsClassifier** (k = 1–10), scaled vs unscaled, compared to a **DummyClassifier** baseline (F1).
- **MyLinearRegression**: OLS via the normal equation `w = (XᵀX)⁻¹Xᵀy`, checked against scikit-learn.
- **Obfuscation**: `X' = XP` with a random invertible matrix `P`. Since `w' = P⁻¹w`, predictions stay the same, and `X = X'P⁻¹` recovers the original data.

## Results (seed 12345)
| Check | Result |
|---|---|
| Best F1, dummy baseline | 0.172 |
| Best F1, kNN unscaled | 0.472 |
| Best F1, kNN scaled (MaxAbs) | **0.661** |
| R², custom LR vs sklearn | 0.4041 vs 0.4041 (max pred diff ~3e-13) |
| R² / RMSE before vs after obfuscation | 0.4041 / 0.4888, same for both |
| Max prediction diff after obfuscation | ~5e-6 (floating-point noise) |

**Takeaways:** scaling matters for distance-based models. Without it, `income` dominates. The normal-equation implementation matches scikit-learn. Multiplying features by an invertible matrix hides the raw values but doesn't change linear-regression predictions.

## How to run
```bash
pip install -r requirements.txt
python src/generate_data.py   # regenerate synthetic data (optional; CSV included)
python src/analysis.py        # print all results
jupyter notebook notebooks/sure_tomorrow.ipynb
```

## Structure
```
data/insurance_synthetic.csv
src/generate_data.py   synthetic data generator
src/analysis.py        kNN, MyLinearRegression, obfuscation
notebooks/sure_tomorrow.ipynb
```
