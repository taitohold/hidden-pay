"""Step 2: who hides pay? Classifier, target = posting has no salary field.

data/processed/jobs_clean.csv -> reports/02_who_hides_metrics.json
                              -> reports/02_who_hides_top_signals.csv
                              -> data/processed/hide_proba.csv (p_hide for every posting)

Trained on all of California; test metrics reported overall and for the Bay Area.
Features: TF-IDF of job_title and desc_clean, job_type, city, company_rating,
log(company_reviews_count). No salary fields, no raw description.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
SEED = 42
C_GRID = [0.03, 0.1, 0.3, 1.0]

USECOLS = ["jobid", "title_clean", "desc_clean", "job_type", "city", "company_rating",
           "company_reviews_count", "has_salary", "bay_area", "split"]


def metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    return {
        "n": int(len(y)),
        "share_hiding": round(float(y.mean()), 4),
        "auc": round(float(roc_auc_score(y, p)), 4) if 0 < y.mean() < 1 else None,
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "macro_f1": round(float(f1_score(y, pred, average="macro", zero_division=0)), 4),
    }


def main():
    df = pd.read_csv(PROCESSED / "jobs_clean.csv", usecols=USECOLS)
    df["title_clean"] = df["title_clean"].fillna("")
    df["desc_clean"] = df["desc_clean"].fillna("")
    df["job_type"] = df["job_type"].fillna("missing")
    df["y"] = (~df["has_salary"]).astype(int)
    df["rating_missing"] = df["company_rating"].isna().astype(float)
    df["log_reviews"] = np.log1p(df["company_reviews_count"].fillna(0))

    tr, va, te = (df[df["split"] == s] for s in ("train", "val", "test"))
    rating_fill = tr["company_rating"].median()  # train only

    # every transformer is fit on train only
    tf_title = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    tf_desc = TfidfVectorizer(ngram_range=(1, 2), min_df=10, max_features=100_000,
                              sublinear_tf=True, stop_words="english")
    ohe = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20)
    scaler = StandardScaler()

    def numeric(d):
        return np.c_[d["company_rating"].fillna(rating_fill), d["rating_missing"], d["log_reviews"]]

    def build(d, fit=False):
        f = (lambda t, x: t.fit_transform(x)) if fit else (lambda t, x: t.transform(x))
        return [
            f(tf_title, d["title_clean"]),
            f(tf_desc, d["desc_clean"]),
            f(ohe, d[["job_type", "city"]]),
            csr_matrix(f(scaler, numeric(d))),
        ]

    B_tr, B_va, B_te = build(tr, fit=True), build(va), build(te)
    X_tr, X_va, X_te = (hstack(b).tocsr() for b in (B_tr, B_va, B_te))
    # same model without the description block (title, job type, city, company only)
    N_tr, N_va, N_te = (hstack(b[:1] + b[2:]).tocsr() for b in (B_tr, B_va, B_te))
    y_tr, y_va, y_te = tr["y"].to_numpy(), va["y"].to_numpy(), te["y"].to_numpy()
    print(f"train {X_tr.shape}, val {X_va.shape}, test {X_te.shape}")

    def fit(Xtr, Xva, label):
        """Tune C on validation AUC, then the threshold on validation macro-F1."""
        best, tuning = None, []
        for C in C_GRID:
            m = LogisticRegression(C=C, class_weight="balanced", solver="liblinear",
                                   random_state=SEED, max_iter=1000).fit(Xtr, y_tr)
            auc = roc_auc_score(y_va, m.predict_proba(Xva)[:, 1])
            tuning.append({"C": C, "val_auc": round(float(auc), 4)})
            print(f"{label} C={C}: val AUC {auc:.4f}")
            if best is None or auc > best[0]:
                best = (auc, C, m)
        _, C, m = best
        p = m.predict_proba(Xva)[:, 1]
        grid = np.linspace(0.05, 0.95, 91)
        thr = float(grid[np.argmax([f1_score(y_va, p >= t, average="macro") for t in grid])])
        return m, C, thr, tuning, p

    model, C, thr, tuning, p_va = fit(X_tr, X_va, "full")
    model_nd, C_nd, thr_nd, tuning_nd, _ = fit(N_tr, N_va, "no-description")

    # test set: touched once, here
    p_te = model.predict_proba(X_te)[:, 1]
    p_te_nd = model_nd.predict_proba(N_te)[:, 1]
    bay = te["bay_area"].to_numpy()
    zeros = np.zeros(len(y_te))  # majority-class baseline: "nobody hides pay"
    report = {
        "target": "posting has no salary field",
        "train_rows": int(len(tr)), "val_rows": int(len(va)), "test_rows": int(len(te)),
        "n_features": int(X_tr.shape[1]),
        "tuning": tuning, "chosen_C": C, "threshold": round(thr, 2),
        "no_description_model": {"tuning": tuning_nd, "chosen_C": C_nd, "threshold": round(thr_nd, 2)},
        "validation": metrics(y_va, p_va, thr),
        "test": {
            "baseline_majority": metrics(y_te, zeros, 0.5) | {"auc": 0.5},
            "logistic_regression_no_description": metrics(y_te, p_te_nd, thr_nd),
            "logistic_regression": metrics(y_te, p_te, thr),
        },
        "test_bay_area": {
            "baseline_majority": metrics(y_te[bay], zeros[bay], 0.5) | {"auc": 0.5},
            "logistic_regression_no_description": metrics(y_te[bay], p_te_nd[bay], thr_nd),
            "logistic_regression": metrics(y_te[bay], p_te[bay], thr),
        },
    }
    (REPORTS / "02_who_hides_metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    # top signals: largest coefficients in each direction
    names = np.r_[
        ["title: " + t for t in tf_title.get_feature_names_out()],
        ["desc: " + t for t in tf_desc.get_feature_names_out()],
        list(ohe.get_feature_names_out(["job_type", "city"])),
        ["company_rating", "rating_missing", "log_reviews"],
    ]
    coef = model.coef_[0]
    order = np.argsort(coef)
    top = pd.DataFrame({
        "direction": ["hides pay"] * 40 + ["shows pay"] * 40,
        "feature": np.r_[names[order[::-1][:40]], names[order[:40]]],
        "coefficient": np.r_[coef[order[::-1][:40]], coef[order[:40]]].round(3),
    })
    top.to_csv(REPORTS / "02_who_hides_top_signals.csv", index=False)

    # propensity for every posting (used by subgroup and selection-bias steps)
    out = df[["jobid", "split"]].copy()
    out["p_hide"] = model.predict_proba(hstack(build(df)).tocsr())[:, 1].round(5)
    out["pred_hide"] = (out["p_hide"] >= thr).astype(int)
    out.to_csv(PROCESSED / "hide_proba.csv", index=False)
    print("wrote reports/02_who_hides_metrics.json, 02_who_hides_top_signals.csv, data/processed/hide_proba.csv")


if __name__ == "__main__":
    main()
