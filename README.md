# Hidden Pay

**Datathon for Social Good — Advanced Track**

California's SB 1162 has required pay ranges in job postings since 2023. We ask two questions of Bay Area job postings:

1. **Who hides pay?** Which postings omit salary, and what predicts it?
2. **What are they hiding?** What would those jobs likely pay, and does that cover local rent?

## Headline results

Every number below comes from a file in `reports/`.

| Finding | Value | Source |
|---|---|---|
| Bay Area postings with no salary field | 6,986 of 45,979 (15.2%) | `01_clean_summary.json` |
| Hide-pay classifier, test AUC (baseline 0.50) | 0.84 with description text, 0.70 without | `02_who_hides_metrics.json` |
| Salary model, test MAE (title-median baseline $44,802) | $20,692 | `03_salary_metrics.json` |
| Salary model, test R² on log pay (baseline 0.17) | 0.84 | `03_salary_metrics.json` |
| Hidden-pay postings that are rent-burdened on a local studio/1BR | 54.7% | `05_rent_burden_summary.json` |
| Same, for postings that show pay (actual salaries) | 50.0% | `05_rent_burden_summary.json` |
| Hidden-pay food service / warehouse / retail postings that are rent-burdened | 98.7% / 95.8% / 94.6% | `05_rent_burden_by_family.csv` |

"Rent-burdened" means annual rent is more than 30% of **gross** annual pay.

## How to run

Requirements: Python 3.11, about 8 GB of free RAM, and the raw data in `data/raw/`:

- `data/raw/indeed.csv` — Indeed job postings (140,816 rows, all of California)
- `data/raw/Zillow_part1.csv`, `zillow_part2.csv`, `zillow_part3.csv` — Zillow listings for the nine Bay Area counties (148,005 rows in total)

Set up the environment (with [uv](https://docs.astral.sh/uv/)):

```bash
uv venv --python 3.11 .venv
```

```bash
uv pip install --python .venv/bin/python -r requirements.txt
```

Or with plain pip: `python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt`.

Run everything, raw data to reported results (about 20 minutes on a laptop):

```bash
.venv/bin/python run_all.py
```

Or run one step at a time, in order:

```bash
.venv/bin/python 01_clean.py
.venv/bin/python 02_who_hides.py
.venv/bin/python 03_salary_model.py
.venv/bin/python 04_subgroups.py
.venv/bin/python 05_rent.py
.venv/bin/python 06_export.py
```

Files in `data/raw/` are never modified. Outputs go to `data/processed/` (intermediate), `reports/` (every reported number), and `site/data/` (JSON for the website).

## Pipeline

| Step | What it does | Main outputs |
|---|---|---|
| `01_clean.py` | Dedupe, parse salaries to annual, strip pay from text, flag Bay Area, make the train/val/test split | `data/processed/jobs_clean.csv`, `reports/01_clean_summary.json`, `reports/salary_handcheck.csv` |
| `02_who_hides.py` | Classifier: does a posting have no salary field? | `reports/02_who_hides_metrics.json`, `reports/02_who_hides_top_signals.csv` |
| `03_salary_model.py` | Regression on log annual pay, trained on postings that show pay | `reports/03_salary_metrics.json`, `reports/03_salary_top_words.csv` |
| `04_subgroups.py` | Both models by job family, job type, city and region; failure examples; selection-bias check | `reports/04_subgroups_*.csv`, `reports/04_salary_error_by_propensity.csv`, `reports/failures.csv`, `reports/job_family_handcheck.csv` |
| `05_rent.py` | Zillow rent by city, joined to postings; rent burden under three rent scenarios | `reports/rent_by_city.csv`, `reports/05_rent_burden_*.{json,csv}`, `reports/rent_join_unmatched.csv` |
| `06_export.py` | Small JSON files for the website | `site/data/*.json`, `reports/06_demo_table.csv` |

## Data decisions

**Scope.** The Indeed file covers all of California. Both models are trained on all of it (more data), and results are reported for the Bay Area, defined as a posting whose city appears in the Zillow data (nine counties). Test metrics are given for all of California and for the Bay Area subset.

**Dedupe.** Postings with the same job title, company and location are collapsed to the earliest one: 4,758 removed, 136,058 kept.

**Salary parsing.** `salary_formatted` is converted to annual pay: hourly × 2080, daily × 260, weekly × 52, monthly × 12. Ranges give a min, max and midpoint. About 13% of values are in Spanish ("por hora", "por año") and are handled. Units that cannot be annualized ("per session", "per mile") are left unparsed (183 postings). Parsed salaries below $20,000 or above $600,000 are treated as errors and dropped from model training (834 postings). 50 random rows are in `reports/salary_handcheck.csv` for manual review.

**`has_salary`.** True when the salary field is filled in at all. A posting whose salary we could not parse or dropped as out of range still counts as disclosing pay.

**Leakage control.** About 70% of postings with a salary field also state pay in the description, so the text would give the answer away. Before modelling we strip dollar amounts, pay units ("per hour", "annually") and pay labels ("pay range", "base salary", "salary", "wage") from the description (`desc_clean`) and the job title (`title_clean`). Models never see the salary fields, the raw description or the raw title. After stripping, 0.5% of descriptions still contain a "$".

**Pay stated only in the text.** 1,837 postings have no salary field but state pay in the description (`pay_in_text`), 514 of them in the Bay Area. We use these as the closest thing to ground truth for hidden-pay postings. Where a posting has both, the pay we parse from text agrees with the salary field within 10% for 93% of postings.

**Split.** 70/15/15 train/validation/test by company (`GroupShuffleSplit` on normalized company name, `random_state=42`), made once in step 1 and reused by every step, so one employer's postings never straddle splits. Text vectorizers and scalers are fit on train only; hyperparameters and thresholds are tuned on validation only.

**Job families.** Keyword buckets on the job title: tech, healthcare, education, food service, retail, warehouse, other. 50 random assignments are in `reports/job_family_handcheck.csv`. 58% of postings fall in "other".

**Rent.** Median monthly asking rent for studio and one-bedroom units (`price` on `FOR_RENT` listings, kept between $500 and $15,000) by city. Cities with fewer than 30 such listings use the county median (140 of 201 cities; 61 use their own). Sensitivity checks use county rent (people commute) and two-bedroom rent. Indeed locations are matched to Zillow cities by lower-cased city name; 45,979 postings matched 145 cities. The only unmatched Bay Area labels are regions, not cities ("San Francisco Bay Area", "Silicon Valley", county names); these are listed in `reports/rent_join_unmatched.csv` and left out.

**Privacy.** Following the data provider's rule, only city- and county-level aggregates are written. No addresses or single properties appear in any output.

## Models

**Who hides pay** (target: no salary field). Features: TF-IDF of cleaned title and description, job type, city, company rating, log company review count.

| Test set | Model | AUC | Precision | Recall | Macro-F1 |
|---|---|---|---|---|---|
| All CA (18,569) | Majority-class baseline | 0.50 | 0.00 | 0.00 | 0.46 |
| | Logistic regression, no description | 0.70 | 0.33 | 0.27 | 0.60 |
| | Logistic regression | 0.84 | 0.50 | 0.54 | 0.71 |
| Bay Area (6,403) | Logistic regression | 0.84 | 0.49 | 0.56 | 0.72 |

Some of the gap between 0.70 and 0.84 comes from descriptions that talk about pay in words we did not strip ("range", "compensation", "depending on experience"). Treat 0.70 as the conservative figure.

**What they are hiding** (target: log annual pay midpoint; same features).

| Test set | Model | MAE | RMSE | R² (log) |
|---|---|---|---|---|
| All CA (15,682) | Median by job title (fallback: global median) | $44,802 | $70,661 | 0.17 |
| | TF-IDF + Ridge (final, chosen on validation) | $20,692 | $36,204 | 0.84 |
| | LightGBM on TF-IDF SVD | $20,975 | $36,644 | 0.83 |
| Bay Area (5,419) | TF-IDF + Ridge | $27,629 | $45,776 | 0.83 |

Subgroup performance for both models (job family, job type, city, region) is in `reports/04_subgroups_classifier.csv` and `reports/04_subgroups_salary.csv`. The 20 worst salary misses are in `reports/failures.csv`; many are errors in the posted salary (a housekeeper at $364,000), and the genuine model failures are specialist physicians.

## Limitations

- **Selection bias.** Employers that disclose pay may pay differently from those that do not, so pay for hidden-pay postings is an estimate, not a fact. We tested this two ways. On hidden-pay postings whose pay we could read from the description, R² on log pay falls from 0.84 to 0.68 (212 postings from test companies). And typical salary error rises from 12.7% to 16.5% between the lowest and highest fifth of predicted probability of hiding pay (`reports/04_salary_error_by_propensity.csv`).
- **Rent burden is slightly overstated by the model.** On Bay Area test postings with a real salary, 52.4% are rent-burdened using actual pay and 54.2% using predicted pay. That two-point overstatement is about half of the gap between hidden-pay (54.7%) and shown-pay (50.0%) postings.
- **A missing salary field is not proof of a legal violation.** Small employers are exempt from SB 1162, and the scrape may have missed pay shown elsewhere.
- **Zillow asking rents are for units listed now,** not what current tenants pay.
- **Gross pay, full-time hours.** Rent burden uses gross pay, not take-home, and annualizes hourly pay at 2080 hours, which overstates income for part-time jobs.
- **Low-wage job families.** Salary errors are not larger for them (food service, retail and warehouse have the smallest errors in dollars and in percent), but the model over-predicts retail pay by about 5%.
- **Weak spots.** The salary model under-predicts pay in Palo Alto by 18% and in Fremont by 11%.
- **Pay estimates are point estimates.** There are no prediction intervals yet.

## Stack

Python 3.11, pandas, numpy, scipy, scikit-learn, lightgbm.
