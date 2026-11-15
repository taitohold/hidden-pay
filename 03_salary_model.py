"""Step 3: what are they hiding? Regression on log(annual salary midpoint).

Trained only on postings with a usable salary (all of California).
data/processed/jobs_clean.csv -> reports/03_salary_metrics.json
                              -> reports/03_salary_top_words.csv
                              -> data/processed/salary_pred.csv (prediction for every posting)

Baseline: median salary by normalized job title (fallback: global median).
Models:   TF-IDF + Ridge, then LightGBM on TF-IDF SVD + categoricals.
Features are the same as step 2: no salary fields, no raw description.
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
SEED = 42
ALPHAS = [0.3, 1.0, 3.0, 10.0]

USECOLS = ["jobid", "title_clean", "desc_clean", "job_type", "city", "company_rating",
           "company_reviews_count", "salary_mid", "salary_in_range", "pay_in_text",
           "text_salary_mid", "bay_area", "split"]


def norm_title(s):
    return s.str.lower().str.replace(r"[^a-z ]+", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()


def metrics(y_log, pred_log):
    """RMSE and MAE in dollars, R2 on the log scale."""
    y, p = np.exp(y_log), np.exp(pred_log)
    return {
        "n": int(len(y)),
        "rmse_dollars": round(float(np.sqrt(np.mean((y - p) ** 2)))),
        "mae_dollars": round(float(np.mean(np.abs(y - p)))),
        "median_abs_err_dollars": round(float(np.median(np.abs(y - p)))),
        "r2_log": round(float(r2_score(y_log, pred_log)), 4),
    }


def main():
    df = pd.read_csv(PROCESSED / "jobs_clean.csv", usecols=USECOLS)
    df["title_clean"] = df["title_clean"].fillna("")
    df["desc_clean"] = df["desc_clean"].fillna("")
    df["job_type"] = df["job_type"].fillna("missing")
    df["title_norm"] = norm_title(df["title_clean"])
    df["rating_missing"] = df["company_rating"].isna().astype(float)
    df["log_reviews"] = np.log1p(df["company_reviews_count"].fillna(0))
    df["y"] = np.log(df["salary_mid"])

    lab = df[df["salary_in_range"]]
    tr, va, te = (lab[lab["split"] == s] for s in ("train", "val", "test"))
    y_tr, y_va, y_te = tr["y"].to_numpy(), va["y"].to_numpy(), te["y"].to_numpy()
    rating_fill = tr["company_rating"].median()

    # ---- baseline: median by normalized title, fallback global median (train only)
    global_med = float(np.median(y_tr))
    title_med = tr.groupby("title_norm")["y"].median()

    def baseline(d):
        return d["title_norm"].map(title_med).fillna(global_med).to_numpy()

    # ---- features, fit on train only
    tf_title = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    tf_desc = TfidfVectorizer(ngram_range=(1, 2), min_df=10, max_features=100_000,
                              sublinear_tf=True, stop_words="english")
    ohe = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20)
    scaler = StandardScaler()

    def numeric(d):
        return np.c_[d["company_rating"].fillna(rating_fill), d["rating_missing"], d["log_reviews"]]

    def build(d, fit=False):
        f = (lambda t, x: t.fit_transform(x)) if fit else (lambda t, x: t.transform(x))
        return hstack([
            f(tf_title, d["title_clean"]),
            f(tf_desc, d["desc_clean"]),
            f(ohe, d[["job_type", "city"]]),
            csr_matrix(f(scaler, numeric(d))),
        ]).tocsr()

    X_tr, X_va, X_te = build(tr, fit=True), build(va), build(te)
    print(f"train {X_tr.shape}, val {X_va.shape}, test {X_te.shape}")

    # ---- Ridge, alpha tuned on validation
    best, ridge_tuning = None, []
    for a in ALPHAS:
        m = Ridge(alpha=a, random_state=SEED).fit(X_tr, y_tr)
        rmse = float(np.sqrt(np.mean((m.predict(X_va) - y_va) ** 2)))
        ridge_tuning.append({"alpha": a, "val_rmse_log": round(rmse, 4)})
        print(f"ridge alpha={a}: val RMSE(log) {rmse:.4f}")
        if best is None or rmse < best[0]:
            best = (rmse, a, m)
    _, alpha, ridge = best

    # ---- LightGBM on SVD of the text + categoricals + numerics
    n_text = len(tf_title.vocabulary_) + len(tf_desc.vocabulary_)
    svd = TruncatedSVD(n_components=150, random_state=SEED).fit(X_tr[:, :n_text])
    cats = {c: pd.CategoricalDtype(tr[c].value_counts()[lambda s: s >= 20].index) for c in ("job_type", "city")}

    def dense(d, X):
        out = pd.DataFrame(svd.transform(X[:, :n_text]), columns=[f"svd{i}" for i in range(150)], index=d.index)
        out[["rating", "rating_missing", "log_reviews"]] = np.c_[d["company_rating"], d["rating_missing"], d["log_reviews"]]
        for c, dt in cats.items():
            out[c] = d[c].astype(dt)
        return out

    D_tr, D_va, D_te = dense(tr, X_tr), dense(va, X_va), dense(te, X_te)
    gbm = lgb.LGBMRegressor(n_estimators=3000, learning_rate=0.05, num_leaves=63,
                            subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                            min_child_samples=30, random_state=SEED, verbose=-1)
    gbm.fit(D_tr, y_tr, eval_set=[(D_va, y_va)], callbacks=[lgb.early_stopping(100, verbose=False)])
    print(f"lightgbm best iteration {gbm.best_iteration_}")

    val = {
        "baseline_title_median": metrics(y_va, baseline(va)),
        "ridge": metrics(y_va, ridge.predict(X_va)),
        "lightgbm": metrics(y_va, gbm.predict(D_va)),
    }
    final = min(("ridge", "lightgbm"), key=lambda k: -val[k]["r2_log"])  # chosen on validation

    # ---- test set: touched once, here
    preds_te = {"baseline_title_median": baseline(te), "ridge": ridge.predict(X_te), "lightgbm": gbm.predict(D_te)}
    bay = te["bay_area"].to_numpy()

    # ---- closest thing to ground truth for the hidden group: postings with no
    # salary field whose pay was parsed from the description (never trained on).
    pit = df[df["pay_in_text"] & df["text_salary_mid"].notna()]
    X_pit = build(pit)
    y_pit = np.log(pit["text_salary_mid"].to_numpy())
    preds_pit = {"baseline_title_median": baseline(pit), "ridge": ridge.predict(X_pit),
                 "lightgbm": gbm.predict(dense(pit, X_pit))}
    pit_test = (pit["split"] == "test").to_numpy()

    report = {
        "target": "log(annual salary midpoint)",
        "train_rows": int(len(tr)), "val_rows": int(len(va)), "test_rows": int(len(te)),
        "title_baseline_coverage_test": round(float(te["title_norm"].isin(title_med.index).mean()), 4),
        "ridge_tuning": ridge_tuning, "ridge_alpha": alpha,
        "lightgbm_best_iteration": int(gbm.best_iteration_),
        "final_model_chosen_on_validation": final,
        "validation": val,
        "test": {k: metrics(y_te, p) for k, p in preds_te.items()},
        "test_bay_area": {k: metrics(y_te[bay], p[bay]) for k, p in preds_te.items()},
        "hidden_pay_in_text_test_companies": {k: metrics(y_pit[pit_test], p[pit_test]) for k, p in preds_pit.items()},
        "hidden_pay_in_text_all": {k: metrics(y_pit, p) for k, p in preds_pit.items()},
    }
    (REPORTS / "03_salary_metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("final_model_chosen_on_validation", "test", "test_bay_area",
                                             "hidden_pay_in_text_test_companies", "hidden_pay_in_text_all")}, indent=2))

    # ---- top Ridge words pushing pay up / down
    names = np.r_[
        ["title: " + t for t in tf_title.get_feature_names_out()],
        ["desc: " + t for t in tf_desc.get_feature_names_out()],
        list(ohe.get_feature_names_out(["job_type", "city"])),
        ["company_rating", "rating_missing", "log_reviews"],
    ]
    order = np.argsort(ridge.coef_)
    pd.DataFrame({
        "direction": ["pay up"] * 40 + ["pay down"] * 40,
        "feature": np.r_[names[order[::-1][:40]], names[order[:40]]],
        "coefficient": np.r_[ridge.coef_[order[::-1][:40]], ridge.coef_[order[:40]]].round(3),
    }).to_csv(REPORTS / "03_salary_top_words.csv", index=False)

    # ---- predictions for every posting (dollars)
    X_all = build(df)
    out = df[["jobid", "split"]].copy()
    out["pred_baseline"] = np.exp(baseline(df)).round()
    out["pred_ridge"] = np.exp(ridge.predict(X_all)).round()
    out["pred_lightgbm"] = np.exp(gbm.predict(dense(df, X_all))).round()
    out["pred_final"] = out[f"pred_{final}"]
    out.to_csv(PROCESSED / "salary_pred.csv", index=False)
    print("wrote reports/03_salary_metrics.json, 03_salary_top_words.csv, data/processed/salary_pred.csv")


if __name__ == "__main__":
    main()
