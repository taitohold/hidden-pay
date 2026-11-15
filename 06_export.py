"""Step 6: write small JSON files for the website to site/data/.

Everything comes from files in reports/ (the same numbers the slides use), plus two
aggregates computed here from data/processed/: a pay histogram and the demo table
(job family x city). The demo table is also saved to reports/06_demo_table.csv.
Only aggregates are exported: no postings, no addresses.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
SITE = ROOT / "site" / "data"
BURDEN = 0.30
MIN_CELL = 10  # smallest family x city cell shown in the demo

LIMITATIONS = [
    "Selection bias: employers that disclose pay may pay differently from those that do not, so "
    "pay for hidden-pay postings is an estimate, not a fact. Model error rises with a posting's "
    "predicted probability of hiding pay, and is higher on postings whose pay we could read from the description.",
    "A missing salary field is not proof of a legal violation: small employers are exempt from SB 1162, "
    "and the scrape may have missed pay shown elsewhere on the page.",
    "Zillow asking rents are for units listed now, not what current tenants pay.",
    "Rent burden uses gross pay, not take-home, and assumes full-time hours (hourly x 2080).",
    "The models are trained on postings from all of California; results are reported for the Bay Area.",
    "Job families come from title keywords; more than half of postings fall in 'other'.",
]


def read_json(name):
    return json.loads((REPORTS / name).read_text())


def records(name, **kw):
    df = pd.read_csv(REPORTS / name, **kw)
    return json.loads(df.to_json(orient="records"))  # NaN -> null


def write(name, obj):
    (SITE / name).write_text(json.dumps(obj, indent=1))
    print(f"site/data/{name}: {(SITE / name).stat().st_size / 1024:.1f} KB")


def main():
    SITE.mkdir(parents=True, exist_ok=True)
    clean = read_json("01_clean_summary.json")
    hide = read_json("02_who_hides_metrics.json")
    sal = read_json("03_salary_metrics.json")
    rent = read_json("05_rent_burden_summary.json")

    # ---- headline
    write("headline.json", {
        "postings_california": clean["rows_after_dedupe"],
        "postings_bay_area": clean["bay_area_rows"],
        "bay_area_no_salary": clean["bay_area_no_salary"],
        "bay_area_share_no_salary": clean["bay_area_share_no_salary"],
        "california_share_no_salary": clean["share_no_salary"],
        "pay_in_text": clean["pay_in_text"],
        "bay_area_pay_in_text": clean["bay_area_pay_in_text"],
        "hidden_burdened_city_1br": rent["hidden_pay_bay_area"]["burdened_city_1br"],
        "hidden_burdened_county_1br": rent["hidden_pay_bay_area"]["burdened_county_1br"],
        "hidden_burdened_city_2br": rent["hidden_pay_bay_area"]["burdened_city_2br"],
        "hidden_median_predicted_pay": rent["hidden_pay_bay_area"]["median_predicted_pay"],
        "shown_burdened_city_1br": rent["shown_pay_bay_area_actual_salary"]["burdened_city_1br"],
        "shown_median_actual_pay": rent["shown_pay_bay_area_actual_salary"]["median_actual_pay"],
        "model_check": rent["model_check_on_test_postings_with_salary"],
        "definition": rent["definition"],
    })

    # ---- who hides pay
    write("who_hides.json", {
        "test": hide["test"], "test_bay_area": hide["test_bay_area"],
        "subgroups": records("04_subgroups_classifier.csv"),
        "top_signals": records("02_who_hides_top_signals.csv"),
    })

    # ---- estimated hidden pay: metrics, top words, and a histogram of pay
    jobs = pd.read_csv(PROCESSED / "jobs_clean.csv",
                       usecols=["jobid", "city", "has_salary", "salary_mid", "salary_in_range", "bay_area"])
    jobs = jobs.merge(pd.read_csv(PROCESSED / "salary_pred.csv")[["jobid", "pred_final"]], on="jobid")
    jobs = jobs.merge(pd.read_csv(PROCESSED / "job_family.csv"), on="jobid")
    bay = jobs[jobs["bay_area"]]
    hidden = bay[~bay["has_salary"]]
    shown = bay[bay["salary_in_range"]]
    edges = np.arange(20_000, 320_001, 10_000)

    def hist(x):
        counts, _ = np.histogram(np.clip(x, edges[0], edges[-1] - 1), bins=edges)
        return (counts / counts.sum()).round(4).tolist()

    write("hidden_pay.json", {
        "final_model": sal["final_model_chosen_on_validation"],
        "test": sal["test"], "test_bay_area": sal["test_bay_area"],
        "hidden_pay_in_text_test_companies": sal["hidden_pay_in_text_test_companies"],
        "top_words": records("03_salary_top_words.csv"),
        "histogram": {
            "bin_start": edges[:-1].tolist(), "bin_width": 10_000,
            "note": "Bay Area. Top bin includes everything above it.",
            "hidden_predicted": hist(hidden["pred_final"]),
            "shown_actual": hist(shown["salary_mid"]),
        },
        "by_family": records("05_rent_burden_by_family.csv"),
    })

    # ---- rent gap
    rent_city = pd.read_csv(REPORTS / "rent_by_city.csv")
    write("rent.json", {
        "summary": rent,
        "by_city": records("05_rent_burden_by_city.csv"),
        "by_family": records("05_rent_burden_by_family.csv"),
        "rent_table": json.loads(rent_city[rent_city["postings"] >= 30].to_json(orient="records")),
    })

    # ---- model card
    fail = pd.read_csv(REPORTS / "failures.csv")
    fail["snippet"] = fail["snippet"].str[:160]
    write("model_card.json", {
        "classifier": hide, "salary_model": sal,
        "salary_subgroups": records("04_subgroups_salary.csv"),
        "salary_error_by_propensity": records("04_salary_error_by_propensity.csv"),
        "failures": json.loads(fail.to_json(orient="records")),
        "data": clean,
        "limitations": LIMITATIONS,
    })

    # ---- demo: job family x city -> spread of predicted pay, rent, share of income
    rents = rent_city.set_index("city")[["rent", "rent_source"]]
    rows = []
    for (fam, city), part in hidden.groupby(["job_family", "city"]):
        if len(part) < MIN_CELL or city not in rents.index:
            continue
        p10, p50, p90 = np.percentile(part["pred_final"], [10, 50, 90])
        r = float(rents.loc[city, "rent"])
        rows.append({
            "job_family": fam, "city": city, "n_postings": len(part),
            "pay_p10": round(p10), "pay_p50": round(p50), "pay_p90": round(p90),
            "monthly_rent": r, "rent_source": rents.loc[city, "rent_source"],
            "rent_share_at_p10": round(r * 12 / p10, 3),
            "rent_share_at_p50": round(r * 12 / p50, 3),
            "rent_share_at_p90": round(r * 12 / p90, 3),
            "share_burdened": round(float((r * 12 > BURDEN * part["pred_final"]).mean()), 3),
        })
    demo = pd.DataFrame(rows).sort_values(["job_family", "n_postings"], ascending=[True, False])
    demo.to_csv(REPORTS / "06_demo_table.csv", index=False)
    write("demo.json", {
        "note": "Hidden-pay Bay Area postings. p10/p50/p90 are the spread of model-predicted pay across "
                "postings in the cell, not a confidence interval for one job.",
        "min_postings_per_cell": MIN_CELL,
        "cells": json.loads(demo.to_json(orient="records")),
    })
    print(f"demo cells: {len(demo)} ({demo['job_family'].nunique()} families, {demo['city'].nunique()} cities)")


if __name__ == "__main__":
    main()
