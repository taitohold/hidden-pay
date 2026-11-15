"""Step 1: clean the raw Indeed postings.

data/raw/indeed.csv -> data/processed/jobs_clean.csv
                    -> reports/salary_handcheck.csv   (50 rows to eyeball)
                    -> reports/01_clean_summary.json  (every count quoted in slides)

Raw description text is NOT written to the processed file, only `desc_clean`,
so later steps cannot use it as a feature by accident.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "indeed.csv"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
SEED = 42
SAL_LO, SAL_HI = 20_000, 600_000

# our name -> names accepted in the raw file (first match wins)
COLS = {
    "jobid": ["jobid", "job_id", "id"],
    "url": ["url", "job_url"],
    "job_title": ["job_title", "title"],
    "company_name": ["company_name", "company"],
    "location": ["location", "job_location"],
    "date_posted": ["date_posted_parsed", "date_posted", "posted_date"],
    "salary_formatted": ["salary_formatted", "salary", "salary_text"],
    "job_type": ["job_type", "employment_type"],
    "description_text": ["description_text", "description"],
    "benefits": ["benefits"],
    "qualifications": ["qualifications"],
    "company_rating": ["company_rating", "rating"],
    "company_reviews_count": ["company_reviews_count", "reviews_count"],
}
REQUIRED = ["job_title", "company_name", "location", "salary_formatted", "description_text"]

# ---------------------------------------------------------------- salary parse

NUM_RE = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*([kK]\b)?")
# period -> (multiplier to annual, pattern). Hourly x 2080, monthly x 12 per spec;
# weekly x 52 and daily x 260 (5 days x 52 weeks) are our additions.
PERIODS = [
    ("hour", 2080, re.compile(r"\bhour|\bhr\b|/\s*hr|hourly|\bhora\b", re.I)),
    ("day", 260, re.compile(r"\bdays?" + r"(?-i:(?![a-z]))" + r"|\bdaily\b|\bd[ií]a\b", re.I)),
    ("week", 52, re.compile(r"\bweek|\bwk\b|\bsemana\b", re.I)),
    ("month", 12, re.compile(r"\bmonth|\bmo\b|\bmes" + r"(?-i:(?![a-z]))", re.I)),
    ("year", 1, re.compile(r"\byear|\byr\b|/\s*yr|annual|annum|\baño\b", re.I)),
]
# "$50 per session", "$0.60 per mile": a unit we cannot annualize.
OTHER_UNIT_RE = re.compile(r"\b(?:per|por)\s+\w+", re.I)


def parse_salary(text):
    """'$20 - $25 an hour' -> (41600.0, 52000.0, 'hour'). Unparseable -> (nan, nan, None).

    No range filter here; that is applied by the caller so it can be counted.
    """
    if not isinstance(text, str) or not text.strip():
        return np.nan, np.nan, None
    found = NUM_RE.findall(text)[:2]
    if not found:
        return np.nan, np.nan, None
    vals = [float(n.replace(",", "")) * (1000 if k else 1) for n, k in found]
    # "$80 - $100K": the K belongs to both ends
    if len(vals) == 2 and found[1][1] and not found[0][1] and vals[0] < 1000:
        vals[0] *= 1000
    period, mult = None, None
    for name, m, pat in PERIODS:
        if pat.search(text):
            period, mult = name, m
            break
    if period is None:  # no unit given: only trust obvious magnitudes
        if OTHER_UNIT_RE.search(text):
            return np.nan, np.nan, None
        if max(vals) < 500:
            period, mult = "hour", 2080
        elif min(vals) >= 10_000:
            period, mult = "year", 1
        else:
            return np.nan, np.nan, None
    lo, hi = min(vals) * mult, max(vals) * mult
    return lo, hi, period


def in_range(lo, hi):
    return (lo >= SAL_LO) & (hi <= SAL_HI)


# ------------------------------------------------------- description cleaning

# End of word that also accepts glued text like "per hourExpected hours:"
# (Indeed drops line breaks). Blocks only a following lowercase letter.
_E = r"(?-i:(?![a-z]))"
_AMT = r"\$\s*\d[\d,]*(?:\.\d+)?\s*(?:[kKmM]\b)?"
_BARE = r"\d[\d,]*(?:\.\d+)?\s*(?:[kK]\b)?"
_SEP = r"\s*(?:-|–|—|to|and)\s*"
_UNIT = (
    r"(?:(?:per|an|a|each)\s+|/\s*)(?:hour|hr|year|yr|annum|month|mo|week|wk|day)s?" + _E +
    r"|\bpor\s+(?:hora|año|mes|semana|d[ií]a)" + _E +
    r"|\bhourly" + _E + r"|\bannually" + _E + r"|\bannual" + _E
    + r"|\byearly" + _E + r"|\bbi-?weekly" + _E
)
# Stripped even with no amount next to it. Narrower than _UNIT so that
# "40 hours a week" and "annual bonus" survive.
_UNIT_ALONE = (
    r"(?:(?:per|an|a)\s+|/\s*)(?:hour|hr|year|yr|annum)s?" + _E
    + r"|\bhourly" + _E + r"|\bannually" + _E + r"|\bpor\s+(?:hora|año)" + _E
)

# "$25 - $30 per hour", "$80,000 to 95,000 a year", "$25/hr"
UNIT_PAY_RE = re.compile(
    rf"{_AMT}(?:{_SEP}\$?\s*{_BARE})?[^$\n]{{0,25}}?(?:{_UNIT})", re.I
)
# "salary range: $80,000 - $95,000" (no unit; parse_salary infers from magnitude)
KEYWORD_PAY_RE = re.compile(
    rf"(?:salary|salaries|pay|compensation|wage|wages|rate)\b[^$\n]{{0,60}}?"
    rf"({_AMT}(?:{_SEP}\$?\s*{_BARE})?)",
    re.I,
)

# Things removed from the description, in order. Edit this list to change
# what counts as a "pay phrase".
STRIP_RES = [
    UNIT_PAY_RE,  # "$1,200 a week" with its unit
    re.compile(rf"(?:USD\s*)?{_AMT}\+?(?:\s*USD)?", re.I),  # any dollar amount
    re.compile(rf"\b{_BARE}(?:{_SEP}{_BARE})?\s*(?:{_UNIT}|\bUSD\b)", re.I),  # "25 per hour"
    re.compile(r"(?<![\d(])(?!401)\d{2,3}\s?[kK]\b"),  # "80k" (but not 401k)
    re.compile(_UNIT_ALONE, re.I),  # "per hour", "hourly", "annually"
    re.compile(
        r"(?:salary|pay|compensation|wage)\s+(?:range|scale|rate|band|transparency)s?" + _E
        + r"|\b(?:base|starting|annual|annualized|hourly|target)\s+(?:salary|pay|compensation|wage|rate)s?" + _E
        + r"|\brate\s+of\s+pay" + _E + r"|pay\s*:|\bon[- ]target\s+earnings" + _E + r"|\bOTE\b"
        + r"|salar(?:y|ies)" + _E + r"|\bwages?" + _E,
        re.I,
    ),
]
_WS = re.compile(r"\s+")
_GLUE = re.compile(r"(?<=[a-z])(?=[A-Z])")


def strip_pay(text):
    """Remove dollar amounts and pay phrases from a description."""
    if not isinstance(text, str):
        return ""
    for pat in STRIP_RES:
        text = pat.sub(" ", text)
    # Un-glue "Full-timeBenefits" -> "Full-time Benefits". Otherwise a removed
    # "Pay: ..." line leaves a space that an absent one does not, and the
    # classifier reads the label straight off the tokenization.
    text = _GLUE.sub(" ", text)
    return _WS.sub(" ", text).strip()


def find_text_pay(text):
    """Pay stated in the description -> (annual_min, annual_max), else (nan, nan)."""
    if not isinstance(text, str):
        return np.nan, np.nan
    for m in UNIT_PAY_RE.finditer(text):
        lo, hi, _ = parse_salary(m.group(0))
        if lo == lo and in_range(lo, hi):
            return lo, hi
    for m in KEYWORD_PAY_RE.finditer(text):
        lo, hi, _ = parse_salary(m.group(1))
        if lo == lo and in_range(lo, hi):
            return lo, hi
    return np.nan, np.nan


# ----------------------------------------------------------------- small utils


def norm_key(s):
    return s.fillna("").astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()


_CITY_RE = re.compile(r"([^,]+),\s*CA\b")


def city_from_location(loc):
    """'920 University Avenue, Berkeley, CA 94710' -> 'Berkeley'.

    The city is the segment right before ', CA'. A bare 'California'
    (statewide / unspecified) becomes 'Unknown'.
    """
    if not isinstance(loc, str):
        return "Unknown"
    m = _CITY_RE.search(loc)
    return m.group(1).strip().title() if m else "Unknown"


def load_raw(path):
    """Read only the columns we need (the file has three copies of the description)."""
    header = pd.read_csv(path, nrows=0).columns
    lower = {c.lower().strip(): c for c in header}
    rename, missing = {}, []
    for ours, candidates in COLS.items():
        hit = next((lower[c] for c in candidates if c in lower), None)
        if hit is not None:
            rename[hit] = ours
        else:
            missing.append(ours)
    hard = [c for c in missing if c in REQUIRED]
    if hard:
        sys.exit(f"Missing required columns {hard}. Raw columns are: {list(header)}")
    if missing:
        print(f"note: optional columns not found, filled with NaN: {missing}")
    df = pd.read_csv(path, usecols=list(rename), dtype=str).rename(columns=rename)
    for c in missing:
        df[c] = np.nan
    return df


# ------------------------------------------------------------------------ main


def main():
    if not RAW.exists():
        sys.exit(f"Raw file not found: {RAW}")
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    df = load_raw(RAW)
    n_raw = len(df)

    # 1. dedupe on title + company + location, keep earliest posting
    df["_date"] = pd.to_datetime(df["date_posted"], errors="coerce", utc=True)
    df["company_norm"] = norm_key(df["company_name"])
    df["_k_title"] = norm_key(df["job_title"])
    df["_k_loc"] = norm_key(df["location"])
    df = (
        df.sort_values("_date", kind="stable", na_position="last")
        .drop_duplicates(subset=["_k_title", "company_norm", "_k_loc"], keep="first")
        .reset_index(drop=True)
    )
    n_dedup = len(df)

    # 2. salary field -> annual min / max / midpoint
    sal_text = df["salary_formatted"].fillna("").astype(str).str.strip()
    parsed = [parse_salary(t) for t in sal_text]
    df["salary_min"] = [p[0] for p in parsed]
    df["salary_max"] = [p[1] for p in parsed]
    df["salary_period"] = [p[2] for p in parsed]

    field_nonempty = sal_text.ne("")
    parseable = df["salary_min"].notna()
    # has_salary = the structured salary field is filled in. That includes pay we
    # cannot annualize ("$50 per session") or later reject as out of range:
    # the employer did disclose, we just can't use it to train the pay model.
    df["has_salary"] = field_nonempty
    n_unparseable = int((field_nonempty & ~parseable).sum())

    ok = in_range(df["salary_min"], df["salary_max"])
    n_out_of_range = int((parseable & ~ok).sum())
    df["salary_in_range"] = parseable & ok
    df.loc[~df["salary_in_range"], ["salary_min", "salary_max"]] = np.nan
    df["salary_mid"] = (df["salary_min"] + df["salary_max"]) / 2

    # 3. description: strip pay, and detect pay stated only in the text
    text_pay = [find_text_pay(t) for t in df["description_text"]]
    df["text_salary_min"] = [p[0] for p in text_pay]
    df["text_salary_max"] = [p[1] for p in text_pay]
    df["text_salary_mid"] = (df["text_salary_min"] + df["text_salary_max"]) / 2
    df["desc_has_pay"] = df["text_salary_min"].notna()
    df["pay_in_text"] = ~df["has_salary"] & df["desc_has_pay"]
    df["desc_clean"] = df["description_text"].map(strip_pay)
    # ~1% of titles carry pay too ("School CNA | Up to $24/hr"); models use title_clean
    df["title_clean"] = df["job_title"].map(strip_pay)

    # 4. light typing for later steps
    df["city"] = df["location"].map(city_from_location)
    df["company_rating"] = pd.to_numeric(df["company_rating"], errors="coerce")
    df["company_reviews_count"] = pd.to_numeric(df["company_reviews_count"], errors="coerce")

    # 4b. Bay Area flag. The Indeed file is all of California; Zillow covers the
    # nine Bay Area counties, so "Bay Area" = city that appears in Zillow.
    z = pd.concat(
        pd.read_csv(f, usecols=["city", "county"], dtype=str)
        for f in sorted((ROOT / "data" / "raw").glob("[Zz]illow*.csv"))
    ).dropna()
    z["city_key"] = z["city"].str.lower().str.strip()
    city_county = z.groupby("city_key")["county"].agg(lambda s: s.mode().iloc[0])
    city_county.rename("county").to_csv(PROCESSED / "bay_area_cities.csv")
    df["county"] = df["city"].str.lower().str.strip().map(city_county)
    df["bay_area"] = df["county"].notna()

    # 4c. One company-level 70/15/15 split shared by every later step, so the
    # same employer never appears in two splits.
    from sklearn.model_selection import GroupShuffleSplit

    g = df["company_norm"].to_numpy()
    idx = np.arange(len(df))
    tr, rest = next(GroupShuffleSplit(1, test_size=0.30, random_state=SEED).split(idx, groups=g))
    va, te = next(GroupShuffleSplit(1, test_size=0.50, random_state=SEED).split(rest, groups=g[rest]))
    df["split"] = "train"
    df.loc[rest[va], "split"] = "val"
    df.loc[rest[te], "split"] = "test"

    # 5. hand-check sample: raw salary text vs parsed value
    pool = df[field_nonempty]
    sample = pool.sample(n=min(50, len(pool)), random_state=SEED)[
        ["jobid", "job_title", "salary_formatted", "salary_period",
         "salary_min", "salary_max", "salary_mid", "salary_in_range"]
    ].copy()
    sample["looks_right"] = ""  # fill in by hand: y / n
    sample.to_csv(REPORTS / "salary_handcheck.csv", index=False)

    # 6. save (no raw description, no helper columns)
    out = df.drop(columns=["description_text", "_date", "_k_title", "_k_loc"])
    out.to_csv(PROCESSED / "jobs_clean.csv", index=False)

    n_hidden = int((~df["has_salary"]).sum())
    summary = {
        "rows_raw": n_raw,
        "rows_after_dedupe": n_dedup,
        "duplicates_removed": n_raw - n_dedup,
        "has_salary": int(df["has_salary"].sum()),
        "no_salary": n_hidden,
        "share_no_salary": round(n_hidden / n_dedup, 4),
        "salary_field_unparseable": n_unparseable,
        "salary_out_of_range_dropped": n_out_of_range,
        "salary_range_bounds": [SAL_LO, SAL_HI],
        "salary_usable_for_model": int(df["salary_in_range"].sum()),
        "salary_period_counts": df["salary_period"].value_counts().to_dict(),
        "pay_in_text": int(df["pay_in_text"].sum()),
        "share_of_no_salary_with_pay_in_text": round(int(df["pay_in_text"].sum()) / max(n_hidden, 1), 4),
        "has_salary_and_pay_in_description": int((df["has_salary"] & df["desc_has_pay"]).sum()),
        "titles_containing_pay": int((df["title_clean"].str.len() < df["job_title"].fillna("").str.strip().str.len() - 1).sum()),
        "n_cities": int(df["city"].nunique()),
        "bay_area_rows": int(df["bay_area"].sum()),
        "bay_area_no_salary": int((df["bay_area"] & ~df["has_salary"]).sum()),
        "bay_area_share_no_salary": round(float((~df.loc[df["bay_area"], "has_salary"]).mean()), 4),
        "rest_of_ca_share_no_salary": round(float((~df.loc[~df["bay_area"], "has_salary"]).mean()), 4),
        "bay_area_salary_usable": int((df["bay_area"] & df["salary_in_range"]).sum()),
        "bay_area_pay_in_text": int((df["bay_area"] & df["pay_in_text"]).sum()),
        "split_rows": df["split"].value_counts().to_dict(),
        "split_companies": df.groupby("split")["company_norm"].nunique().to_dict(),
    }
    (REPORTS / "01_clean_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {PROCESSED / 'jobs_clean.csv'} ({len(out):,} rows)")


if __name__ == "__main__":
    main()
