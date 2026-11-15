# Hidden Pay — Datathon for Social Good (Advanced Track)

## Deadline

Submit by Sunday 1:30 PM (hard deadline 2:00 PM). Needs: project link, slide deck, video (max 5 min). Prefer the simple thing that works. No refactors, no new dependencies unless they save time.

## Question

California's SB 1162 has required pay ranges in job postings since 2023.

1. Who hides pay? Which Bay Area postings omit salary, and what predicts it?
2. What are they hiding? Estimate pay for postings that omit it, and check whether that pay covers local rent.

## Data

* `data/raw/indeed.csv` — ~100k Bay Area postings. Expected columns (verify names first): jobid, url, job_title, company_name, location, date_posted, salary_formatted (+ min/max), job_type, description_text, benefits, qualifications, company_rating, company_reviews_count
* `data/raw/zillow.csv` — ~148k Bay Area listings with rent estimates and city.
* Never edit files in `data/raw/`. Write outputs to `data/processed/`.

Verified on disk (2026-11-14):

* `indeed.csv` has 140,816 rows and 28 columns, and covers all of California (LA, San Diego, etc.), not only the Bay Area. Column names match the list above; the usable date column is `date_posted_parsed` (`date_posted` is text like "30+ days ago"); there are no salary min/max columns; `qualifications` is empty. About 13% of `salary_formatted` values are in Spanish ("por hora", "por año").
* Zillow is split into three files, 140 columns each, same header: `Zillow_part1.csv`, `zillow_part2.csv`, `zillow_part3.csv`. Read all three and concatenate. All nine columns named in step 5 are present.

## Pipeline (one script per step, run in order by `make all` or `run_all.py`)

1. `01_clean.py`
   * Dedupe on job_title + company_name + location (keep earliest).
   * Parse salary to annual: hourly x 2080, monthly x 12, "K" handled, ranges -> min, max, midpoint.
   * Drop parsed salaries outside $20k–$600k as parse errors; log how many.
   * `has_salary` = structured salary present.
   * Strip all dollar amounts and pay phrases from description_text into `desc_clean`. Also record `pay_in_text` (salary appears in the description but not the field). Report this count: it is a finding.
   * Save a random sample of 50 rows (raw salary text vs parsed value) to `reports/salary_handcheck.csv` for manual review.
2. `02_who_hides.py` — classifier, target `has_salary == False`
   * Features: TF-IDF of job_title and desc_clean, job_type, city, company_rating, log(company_reviews_count).
   * Baseline: majority class. Model: logistic regression, then LightGBM if time.
   * Metrics on held-out test: AUC, precision, recall, macro-F1.
3. `03_salary_model.py` — regression, target log(annual midpoint), trained only on rows with salary
   * Baseline: median salary by normalized job title (fallback: global median).
   * Model: TF-IDF + Ridge first. Then LightGBM on TF-IDF SVD or sentence embeddings if time.
   * Metrics on held-out test: RMSE and MAE in dollars, R2 on log scale.
4. `04_subgroups.py`
   * Both models' metrics by: city, job_type, and job family (keyword buckets from title: healthcare, tech, food service, retail, warehouse, education, other).
   * Export the 20 worst salary-model misses with text snippets to `reports/failures.csv`.
5. `05_rent.py`
   * Load Zillow with `usecols=[zpid, city, county, zipcode, homeStatus, homeType, price, rentZestimate, bedrooms]` (file is large, ~139 columns).
   * First verify that `price` on FOR_RENT rows is monthly rent: print its distribution. Keep rents between $500 and $15,000.
   * Rent measure: median asking rent (`price`, FOR_RENT rows) for studio/1BR (`bedrooms <= 1`) by city. A single worker is the unit, so do not average over whole houses.
   * If a city has fewer than 30 such rows, use median `rentZestimate` for `bedrooms <= 1` properties in that city; if still under 30, use the county median. Record which source each city used.
   * Join: Indeed `location` ("City, CA") -> strip ", CA", lowercase, trim; match to Zillow `city` lowercased. Print unmatched cities and fix by hand. Save the city table to `reports/rent_by_city.csv` and hand-check 10 cities.
   * Predict pay for no-salary postings. Flag "rent-burdened" where annual rent > 30% of predicted gross pay (state that it is gross, not take-home).
   * Sensitivity: repeat with county median rent (people commute) and with 2BR rent. Report all three.
   * Output shares by city and job family.
   * Ethics rule from the data provider: city or zip-level aggregates only. Never output addresses or single properties.
6. `06_export.py` — write small JSON files for the website to `site/data/`.

## Rules that protect the result

* Split by company (GroupShuffleSplit on normalized company_name), 70/15/15 train/val/test. Reposts and same-employer postings must not straddle splits.
* Tune on validation only. Touch the test set once, at the end.
* Fix `random_state=42` everywhere.
* Fit TF-IDF and any scalers on train only.
* Never use salary fields or raw description text as features. Only `desc_clean`.
* Every number in the slides must come from a file in `reports/`.

## Advanced Track checklist (judges check this first)

* [ ] Model trained (not an off-the-shelf API)
* [ ] Baseline comparison for both models
* [ ] Task-appropriate metrics on held-out test set
* [ ] Performance across >= 3 subgroups + failure examples
* [ ] Data decisions explained + hand-checked sample (salary parse, Zillow join)
* [ ] Code runs raw data -> reported results (`README.md` with exact commands)
* [ ] Two datasets joined (Indeed + Zillow)

## Depth upgrades (only after every checklist box is green; do in this order)

1. Uncertainty on every pay estimate. Train LightGBM quantile models (10th, 50th, 90th percentile). Report interval coverage on the test set. Rent-burden claims use the range: "even at the optimistic 90th percentile, X% of hidden-pay postings can't cover rent."
2. Test the selection-bias problem instead of only admitting it.
   * Use the hide-pay classifier's predicted probability as a propensity score. Report salary-model error by propensity quintile on the test set. If error is flat, extrapolating to hidden-pay postings is more credible; if it climbs, say so.
   * Evaluate the salary model on `pay_in_text` postings (no salary field, but pay parsed from the description). These are the closest thing to ground truth for the hidden group.
3. Ablation table. Title-median baseline -> TF-IDF + Ridge -> LightGBM -> sentence embeddings. One table, same test set.
4. Explainability. Top words pushing pay up/down (Ridge coefficients or SHAP), and top signals of a posting hiding pay.
5. Site demo. Pick a job family + city -> estimated pay range and share of income going to rent.

Also hand-check 50 job-family keyword assignments and the Zillow city join (cities with no match, rent based on studio/1BR listings only).

## Limitations to state plainly

* Selection bias: employers that disclose pay may pay differently from those that don't, so predictions for hidden-pay postings are estimates, not facts.
* A missing salary field is not proof of a legal violation (employer size exemption, scraping gaps).
* Zillow rent estimates are for listed properties, not what current tenants pay.
* Check whether errors are larger for low-wage job families; if so, say so.

## Website (`site/`)

Static single page, no backend, deployed on GitHub Pages. Reads precomputed JSON from `site/data/`. Sections: headline number -> who hides pay -> estimated hidden pay -> rent gap by city -> model card (metrics, subgroups, limits). Charts: Plotly.js. Use the frontend-design skill for layout and typography.

## Stack

Python 3.11, pandas, numpy, scikit-learn, lightgbm, (optional) sentence-transformers, shap, matplotlib/plotly.
