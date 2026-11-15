"""Step 5: does hidden pay cover local rent? Joins Indeed postings to Zillow rents by city.

Rent measure: median monthly asking rent (`price` on FOR_RENT rows) for studio/1BR
units by city. A single worker is the unit, so whole houses are not averaged in.
Fallbacks when a city has < 30 such listings: median rentZestimate for <= 1BR
properties in that city, then the county median asking rent.

"Rent-burdened" = annual rent > 30% of predicted GROSS annual pay (not take-home).
Sensitivity: county median rent (people commute) and 2BR rent.

Ethics rule from the data provider: only city / county aggregates are written.
No addresses, no single properties.

Writes:
  reports/rent_by_city.csv              rent table, source per city, pay needed
  reports/rent_join_unmatched.csv       Indeed cities with no Zillow match
  reports/05_rent_burden_summary.json   headline shares, all three rent scenarios
  reports/05_rent_burden_by_city.csv
  reports/05_rent_burden_by_family.csv
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
MIN_LISTINGS = 30
RENT_LO, RENT_HI = 500, 15_000
BURDEN = 0.30
MIN_POSTINGS = 30  # smallest city / family reported

ZCOLS = ["zpid", "city", "county", "zipcode", "homeStatus", "homeType", "price", "rentZestimate", "bedrooms"]


def load_zillow():
    z = pd.concat(pd.read_csv(f, usecols=ZCOLS, low_memory=False)
                  for f in sorted(RAW.glob("[Zz]illow*.csv")))
    for c in ("price", "rentZestimate", "bedrooms"):
        z[c] = pd.to_numeric(z[c], errors="coerce")
    z["city_key"] = z["city"].str.lower().str.strip()
    return z.dropna(subset=["city_key", "county"])


def rent_table(z):
    rent = z[z["homeStatus"] == "FOR_RENT"]
    print("FOR_RENT `price` distribution (should look like monthly rent):")
    print(rent["price"].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).round(0).to_string())
    kept = rent[rent["price"].between(RENT_LO, RENT_HI)]
    print(f"kept {len(kept):,} of {len(rent):,} FOR_RENT rows with rent in ${RENT_LO}-${RENT_HI:,}")

    small = kept[kept["bedrooms"] <= 1]
    two = kept[kept["bedrooms"] == 2]
    zest = z[(z["bedrooms"] <= 1) & z["rentZestimate"].between(RENT_LO, RENT_HI)]

    cities = z.groupby("city_key").agg(city=("city", lambda s: s.mode().iloc[0]),
                                       county=("county", lambda s: s.mode().iloc[0]))
    ask = small.groupby("city_key")["price"].agg(n_asking="size", rent_asking="median")
    est = zest.groupby("city_key")["rentZestimate"].agg(n_zest="size", rent_zest="median")
    ask2 = two.groupby("city_key")["price"].agg(n_2br="size", rent_2br_city="median")
    t = cities.join([ask, est, ask2])
    t[["n_asking", "n_zest", "n_2br"]] = t[["n_asking", "n_zest", "n_2br"]].fillna(0).astype(int)

    county_1br = small.groupby("county")["price"].median().rename("rent_county")
    county_2br = two.groupby("county")["price"].median().rename("rent_2br_county")
    t = t.join(county_1br, on="county").join(county_2br, on="county")

    use_ask = t["n_asking"] >= MIN_LISTINGS
    use_zest = ~use_ask & (t["n_zest"] >= MIN_LISTINGS)
    t["rent"] = np.where(use_ask, t["rent_asking"], np.where(use_zest, t["rent_zest"], t["rent_county"]))
    t["rent_source"] = np.where(use_ask, "city asking rent", np.where(use_zest, "city rentZestimate", "county median"))
    t["n_listings_used"] = np.where(use_ask, t["n_asking"], np.where(use_zest, t["n_zest"], 0))
    t["rent_2br"] = np.where(t["n_2br"] >= MIN_LISTINGS, t["rent_2br_city"], t["rent_2br_county"])
    t["pay_needed"] = (t["rent"] * 12 / BURDEN).round()  # gross pay at which rent = 30%
    return t[["city", "county", "rent", "rent_source", "n_listings_used", "rent_county", "rent_2br", "pay_needed"]]


def shares(d):
    """Share rent-burdened under each rent scenario, using predicted gross pay."""
    pay = d["pred_final"]
    return {
        "n": int(len(d)),
        "median_predicted_pay": round(float(pay.median())),
        "burdened_city_1br": round(float((d["rent"] * 12 > BURDEN * pay).mean()), 4),
        "burdened_county_1br": round(float((d["rent_county"] * 12 > BURDEN * pay).mean()), 4),
        "burdened_city_2br": round(float((d["rent_2br"] * 12 > BURDEN * pay).mean()), 4),
    }


def main():
    REPORTS.mkdir(exist_ok=True)
    rents = rent_table(load_zillow())

    jobs = pd.read_csv(PROCESSED / "jobs_clean.csv",
                       usecols=["jobid", "city", "has_salary", "salary_mid", "salary_in_range", "bay_area", "split"])
    jobs = jobs.merge(pd.read_csv(PROCESSED / "salary_pred.csv")[["jobid", "pred_final"]], on="jobid")
    jobs = jobs.merge(pd.read_csv(PROCESSED / "job_family.csv"), on="jobid")
    jobs["city_key"] = jobs["city"].str.lower().str.strip()

    # join report: which Indeed cities found no Zillow city
    matched = jobs["city_key"].isin(rents.index)
    unmatched = jobs[~matched]["city"].value_counts().rename_axis("city").reset_index(name="postings")
    unmatched.to_csv(REPORTS / "rent_join_unmatched.csv", index=False)
    print(f"\njoin: {matched.sum():,} of {len(jobs):,} postings matched a Zillow city "
          f"({jobs.loc[matched, 'city_key'].nunique()} cities); {len(unmatched)} Indeed cities unmatched")
    print("largest unmatched:", unmatched.head(15).set_index("city")["postings"].to_dict())

    j = jobs[matched].join(rents[["county", "rent", "rent_county", "rent_2br"]], on="city_key")

    posts = j.groupby("city_key").agg(postings=("jobid", "size"), hidden_pay_postings=("has_salary", lambda s: int((~s).sum())))
    city_tab = rents.join(posts).fillna({"postings": 0, "hidden_pay_postings": 0})
    city_tab.sort_values("postings", ascending=False).to_csv(REPORTS / "rent_by_city.csv", index=False)
    print("\nrent source counts:", city_tab["rent_source"].value_counts().to_dict())
    print("largest 10 cities (hand-check these against Zillow):")
    print(city_tab.sort_values("postings", ascending=False).head(10).to_string(index=False))

    hidden = j[~j["has_salary"]]
    # reference: postings that DO show pay, using their actual salary, and the model's
    # prediction for the same rows (does the prediction reproduce the true share?)
    shown = j[j["salary_in_range"]]
    shown_test = shown[shown["split"] == "test"]

    def actual_share(d):
        return round(float((d["rent"] * 12 > BURDEN * d["salary_mid"]).mean()), 4)

    summary = {
        "definition": "rent-burdened = 12 x monthly rent > 30% of gross annual pay (gross, not take-home)",
        "rent_measure": "median asking rent, studio/1BR, by city; fallbacks: city rentZestimate, county median",
        "cities_in_rent_table": int(len(city_tab)),
        "rent_source_counts": city_tab["rent_source"].value_counts().to_dict(),
        "hidden_pay_bay_area": shares(hidden),
        "shown_pay_bay_area_actual_salary": {
            "n": int(len(shown)), "median_actual_pay": round(float(shown["salary_mid"].median())),
            "burdened_city_1br": actual_share(shown),
        },
        "model_check_on_test_postings_with_salary": {
            "n": int(len(shown_test)),
            "burdened_using_actual_pay": actual_share(shown_test),
            "burdened_using_predicted_pay": shares(shown_test)["burdened_city_1br"],
            "agreement_per_posting": round(float(((shown_test["rent"] * 12 > BURDEN * shown_test["salary_mid"])
                                                  == (shown_test["rent"] * 12 > BURDEN * shown_test["pred_final"])).mean()), 4),
        },
    }
    (REPORTS / "05_rent_burden_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n" + json.dumps(summary, indent=2))

    def table(col):
        rows = [{col: g, **shares(part)} for g, part in hidden.groupby(col) if len(part) >= MIN_POSTINGS]
        return pd.DataFrame(rows).sort_values("n", ascending=False)

    by_city = table("city").merge(rents[["city", "rent", "rent_source", "pay_needed"]], on="city")
    by_city.to_csv(REPORTS / "05_rent_burden_by_city.csv", index=False)
    by_family = table("job_family")
    by_family.to_csv(REPORTS / "05_rent_burden_by_family.csv", index=False)
    pd.set_option("display.width", 220)
    print("\nBY CITY (hidden-pay postings)\n", by_city.to_string(index=False))
    print("\nBY JOB FAMILY (hidden-pay postings)\n", by_family.to_string(index=False))


if __name__ == "__main__":
    main()
