"""Step 4: both models by subgroup (test set only), plus failure examples.

Reads jobs_clean.csv, hide_proba.csv, salary_pred.csv. Writes:
  data/processed/job_family.csv            job family for every posting
  reports/job_family_handcheck.csv         50 random title -> family rows to eyeball
  reports/04_subgroups_classifier.csv      hide-pay classifier by city / job_type / job family
  reports/04_subgroups_salary.csv          salary model by city / job_type / job family
  reports/04_salary_error_by_propensity.csv  salary error by quintile of P(hides pay)
  reports/failures.csv                     20 worst salary-model misses with text snippets
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, r2_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
SEED = 42
MIN_N = 100  # smallest subgroup we report

# Keyword buckets on the job title. First match wins, so order matters.
FAMILIES = [
    ("tech", r"software|developer|programmer|devops|data scien|data engineer|data analyst|machine learning"
             r"|\bml\b|\bai\b|cyber|cloud|full[- ]?stack|front[- ]?end|back[- ]?end|\bsre\b|\bqa\b"
             r"|network engineer|systems? (?:engineer|admin)|\bit (?:support|specialist|technician|manager)"
             r"|product manager|firmware|hardware engineer|web designer|ux\b|solutions architect"),
    ("healthcare", r"nurse|nursing|\brn\b|\blvn\b|\blpn\b|\bcna\b|medical|physician|dental|dentist|therap"
                   r"|clinic|health|pharm|caregiver|patient|surg|hospital|veterin|phlebot|radiolog|psych"
                   r"|hospice|doctor|\bmd\b|optomet|sonograph|paramedic|\bemt\b|behavior"),
    ("education", r"teacher|tutor|instructor|professor|school|education|paraeducator|teaching|lecturer"
                  r"|substitute|preschool|child ?care|faculty|coach|academic|librarian"),
    ("food service", r"cook|chef|barista|server|bartender|dishwasher|restaurant|food|kitchen|crew member"
                     r"|baker|host(?:ess)?\b|busser|cater|^team member|dining"),
    ("retail", r"retail|cashier|sales associate|store|merchandis|stock(?:er| associate)|beauty advisor"
               r"|sales lead|key ?holder|grocery|personal shopper"),
    ("warehouse", r"warehouse|forklift|picker|packer|loader|material handler|shipping|receiving|driver"
                  r"|delivery|logistics|order selector|package handler|fulfillment|dock"),
]


def job_family(titles):
    t = titles.fillna("").str.lower()
    out = pd.Series("other", index=t.index)
    for name, pat in reversed(FAMILIES):  # apply in reverse so the first in the list wins
        out[t.str.contains(pat, regex=True)] = name
    return out


def clf_row(d):
    y, p, pred = d["hides"].to_numpy(), d["p_hide"].to_numpy(), d["pred_hide"].to_numpy()
    both = 0 < y.mean() < 1
    return {
        "n": len(d),
        "share_hiding": round(float(y.mean()), 4),
        "auc": round(float(roc_auc_score(y, p)), 4) if both else np.nan,
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "macro_f1": round(float(f1_score(y, pred, average="macro", zero_division=0)), 4),
    }


def sal_row(d):
    y, p, b = d["salary_mid"].to_numpy(), d["pred_final"].to_numpy(), d["pred_baseline"].to_numpy()
    return {
        "n": len(d),
        "median_salary": round(float(np.median(y))),
        "mae_dollars": round(float(np.mean(np.abs(y - p)))),
        "rmse_dollars": round(float(np.sqrt(np.mean((y - p) ** 2)))),
        "r2_log": round(float(r2_score(np.log(y), np.log(p))), 4),
        # relative error and direction: is the model worse, or biased, for low-wage groups?
        "median_abs_pct_err": round(float(np.median(np.abs(p - y) / y)), 4),
        "mean_pct_bias": round(float(np.mean((p - y) / y)), 4),  # > 0 = model over-predicts
        "baseline_mae_dollars": round(float(np.mean(np.abs(y - b)))),
    }


def by_group(d, col, fn, keep=None):
    rows = []
    for g, part in d.groupby(col):
        if len(part) >= MIN_N and (keep is None or g in keep):
            rows.append({"dimension": col, "group": g, **fn(part)})
    return rows


def main():
    df = pd.read_csv(PROCESSED / "jobs_clean.csv",
                     usecols=["jobid", "job_title", "job_type", "city", "has_salary", "salary_mid",
                              "salary_in_range", "bay_area", "split", "desc_clean"])
    df = df.merge(pd.read_csv(PROCESSED / "hide_proba.csv").drop(columns="split"), on="jobid")
    df = df.merge(pd.read_csv(PROCESSED / "salary_pred.csv").drop(columns="split"), on="jobid")
    df["hides"] = (~df["has_salary"]).astype(int)
    df["job_family"] = job_family(df["job_title"])
    # "Part-time, Full-time" -> first listed type
    df["job_type_main"] = df["job_type"].fillna("missing").str.split(",").str[0].str.strip()

    df[["jobid", "job_family"]].to_csv(PROCESSED / "job_family.csv", index=False)
    hc = df.sample(50, random_state=SEED)[["job_title", "job_family"]].copy()
    hc["looks_right"] = ""
    hc.to_csv(REPORTS / "job_family_handcheck.csv", index=False)
    print("job family counts:", df["job_family"].value_counts().to_dict())

    te = df[df["split"] == "test"]            # classifier test set
    ts = te[te["salary_in_range"]]            # salary-model test set
    # cities: the largest Bay Area cities in the test set, plus LA / San Diego for contrast
    bay_cities = set(te[te["bay_area"]]["city"].value_counts().head(12).index)
    cities = bay_cities | {"Los Angeles", "San Diego"}

    def table(d, fn):
        rows = [{"dimension": "all", "group": "all test", **fn(d)},
                {"dimension": "region", "group": "Bay Area", **fn(d[d["bay_area"]])},
                {"dimension": "region", "group": "rest of CA", **fn(d[~d["bay_area"]])}]
        rows += by_group(d, "job_family", fn)
        rows += by_group(d, "job_type_main", fn)
        rows += by_group(d, "city", fn, keep=cities)
        return pd.DataFrame(rows)

    clf = table(te, clf_row)
    sal = table(ts, sal_row)
    clf.to_csv(REPORTS / "04_subgroups_classifier.csv", index=False)
    sal.to_csv(REPORTS / "04_subgroups_salary.csv", index=False)
    pd.set_option("display.width", 220)
    print("\nCLASSIFIER\n", clf.to_string(index=False))
    print("\nSALARY MODEL\n", sal.to_string(index=False))

    # selection-bias check: does salary error grow with the propensity to hide pay?
    q = ts.copy()
    q["propensity_quintile"] = pd.qcut(q["p_hide"], 5, labels=[1, 2, 3, 4, 5])
    prop = pd.DataFrame([
        {"propensity_quintile": k, "mean_p_hide": round(float(part["p_hide"].mean()), 4), **sal_row(part)}
        for k, part in q.groupby("propensity_quintile", observed=True)
    ])
    prop.to_csv(REPORTS / "04_salary_error_by_propensity.csv", index=False)
    print("\nSALARY ERROR BY PROPENSITY TO HIDE\n", prop.to_string(index=False))

    # 20 worst salary misses (largest ratio between predicted and actual)
    f = ts.copy()
    f["abs_log_err"] = np.abs(np.log(f["pred_final"]) - np.log(f["salary_mid"]))
    f = f.nlargest(20, "abs_log_err")
    f["snippet"] = f["desc_clean"].fillna("").str[:300]
    f[["job_title", "city", "job_type", "job_family", "salary_mid", "pred_final", "snippet"]].rename(
        columns={"salary_mid": "actual_salary", "pred_final": "predicted_salary"}
    ).to_csv(REPORTS / "failures.csv", index=False)
    print("\nWORST MISSES\n", f[["job_title", "city", "salary_mid", "pred_final"]].to_string(index=False))


if __name__ == "__main__":
    main()
