# Partisanship and Clean-Energy Manufacturing Investment

## Empirical handoff

Narrow Republican House victories produced an estimated **+0.070 log-point** change in documented manufacturing CAPEX announced in the district during the following term (robust 95% interval **−1.188 to +1.329**, p = 0.913). The effective comparison includes 188 Democratic and 191 Republican victories across 40 states. The interval accommodates substantial decreases and increases. These data resolve neither the sign of the effect nor whether it is economically small.

| Outcome | Estimated Republican-minus-Democratic discontinuity | Robust 95% interval |
|---|---:|---:|
| Log of one plus documented CAPEX in millions of 2023 dollars | +0.070 | −1.188 to +1.329 |
| Documented CAPEX per district-term, millions of 2023 dollars | −159 | −475 to +157 |
| Manufacturing announcements per district-term | +0.000 | −0.474 to +0.474 |
| Probability of at least one announcement, percentage points | −8.5 | −28.0 to +11.0 |

Each outcome has its own data-driven bandwidth. The dollar-level specification gives large projects greater influence. The log specification includes zeros and applies to one plus millions of dollars. Converting its coefficient to a percent change in aggregate investment or dollars per facility would change the estimand. The outcomes measure publicly documented announcements through December 2024. Cross-district displacement and later construction or production are outside this design.

## How the evidence changes the original interpretation

The archived forest used **483 manufacturing and 6,889 energy-and-industry observations** after trimming. Its 0.29154 estimate describes a mixed-sector, facility-level comparison coded with the release's later House representative. The script forms separate state, year and quarter dummy blocks. The forest referenced by `cf_base` for the saved average used these controls. The later `CF_MAIN` fit used for heterogeneity and bootstrap refits omitted them. Its cluster bootstrap selected rows by membership in the sampled districts, losing multiplicities, and refit a shared forest object in concurrent workers. The saved within-district permutation reproduced the original point estimate (p = 1). The positive bootstrap interval and subgroup rankings cannot establish a causal manufacturing premium.

Two comparisons isolate the changes. Holding the legacy party label, regression specification and 2018–24 screen fixed, the adjusted project-size coefficient is 0.388 for the mixed sample (7,596 projects) and 0.314 for manufacturing (490). On the **same 488 manufacturing projects** with both party measures, recoding the representative on the announcement date changes 61 labels and reduces the coefficient from **0.316** (95% interval −0.144 to 0.776) to **0.252** (−0.237 to 0.742). These two regressions use state, calendar-quarter and technology effects with identical map-specific district clusters. They describe project size conditional on announcement. The close-election analysis instead includes districts that received zero observed projects.

## Identification and measurement checks

- The dated sample contains 577 manufacturing announcements, including 534 with valid coordinates and 491 with both valid coordinates and CAPEX. All 534 geolocated to a state-consistent district; ten are within 0.001 geographic degrees of a district-polygon edge. Such records are flagged for geographic sensitivity. The election panel includes 473 announcements, of which 435 have a dollar value. Of 1,191 eligible district-terms, 893 have no observed announcement and 22 have only announcements of unknown value.
- Comparing predetermined manufacturing announcements from January–September of the election year on the same electoral map gives a pre-election CAPEX discontinuity of **+0.004** (−0.411 to +0.419). The vote-margin density difference is −0.284 (−0.803 to +0.235, p = 0.284). These checks reveal no clear break at a tie. The pre-period has only 124 observed announcements across eligible races and covers a shorter period than the two-year outcome.
- For comparable prior footprints, the Republican incumbent-party discontinuity is −11.8 percentage points (−31.1 to +7.5). Income, population, unemployment count and Gini intervals include zero. Restricting to prior ACS polygons with at least 99% intersection-over-union leaves 757 races. The checks have limited power against sorting, especially after redistricting. Among districts with projects, the disclosure-rate difference is +3.9 points (−21.5 to +29.3). Differential disclosure remains plausible.
- At fixed ±5 and ±10 percentage-point margins, the main estimate is +1.156 and +0.364 log points, respectively. Omitting one cycle at a time produces estimates between −0.441 and +0.595. Every interval includes zero. Removing the largest announcement, a $7.97 billion North Carolina battery project, gives +0.095 (−1.164 to +1.354). The sign varies by election cohort; this single plant has little influence on the pooled log estimate.
- Every winner and vote margin in the 379-race primary estimation window has a published-return audit. Dated terms independently agree for 378; the NY-22 seat had no member on January 3, 2021. Its certified Republican victory remains in the assigned group. The NC legislature's 2019 boundaries were used for the 2020 election, retaining ten matched post-election announcements in eligible NC races. Alaska's 2022 final ranked-choice count supersedes first preferences.

## Section 45X during the 118th Congress

Inside a fixed ±10-point margin, **8 of 42 Democratic-won districts** (19.0%) and **6 of 31 Republican-won districts** (19.4%) announced at least one clearly exposed project. Those districts account for 16 and 7 projects, respectively. These are unadjusted window averages. An RD fits separate slopes to estimate the difference *at a tied election*, so its point estimate can differ from the raw gap.

The bias-corrected probability estimate using the data-driven bandwidth is **−17.0 percentage points** (95% interval **−52.9 to +18.8**), based on 15 Democratic and 14 Republican districts with events. The interval includes both a ten-point loss and a ten-point gain. Adding ambiguous products gives −18.4 points (−49.7 to +12.9). The estimates describe siting under a common federal policy. Project descriptions identify product exposure; tax claims, qualifying production and sales, and the incremental effect of the federal credit are separate questions. [IRS final regulations](https://www.irs.gov/irb/2024-51_IRB)

## Replication

From the repository root:

```sh
conda run -n clean-investment-causal python revision_analysis/analyze.py --out revision_analysis/reproduced
```

Choose a new output path on each run. The program stages outputs, verifies the district totals, vote margins, figure bins and policy counts, and publishes the finished directory together. Compare two completed directories with `conda run -n clean-investment-causal python revision_analysis/verify.py revision_analysis/results --replica revision_analysis/reproduced`. Use the package versions in `results/environment.csv` and Poppler's `pdftotext` on PATH (the script also detects the existing local conda installation). The input SHA-256 manifest is in `results/inputs.csv`.

The published results use local-linear triangular-kernel RD with cycle indicators, MSE bandwidth selection, quadratic bias correction and state-clustered CR1 inference. Fixed windows use twice the estimation bandwidth for bias correction. The balance tests share both primary bandwidths. Each side's raw outcome mean is in the results tables. The cutoff plot shows raw binned means and separate unadjusted triangular-weighted lines; the coefficient plot shows the cycle-adjusted bias-corrected estimates.

Start with **main_rd.csv**, **sensitivity.csv**, **diagnostics.csv**, **project_size_associations.csv** and **policy_counts.csv** in `results/`. The corresponding election, geographic, product-classification and sample ledgers are in the same directory. The **election_roster.csv** audit uses the district candidate tables, sums fusion ballot lines, documents exclusions, and uses Clerk totals for discrepancies in New York. The product crosswalk was specified before estimating party contrasts. The incumbent and ACS checks compare simplified polygon footprints; the project assignments use the original polygons with offline CRS transformation. The 118th district layer is cartographically generalized, so precise boundary locations are a remaining source of uncertainty.

Sources: [House Clerk election statistics](https://history.house.gov/Institution/Election-Statistics/Election-Statistics/), [Alaska official final-round report](https://www.elections.alaska.gov/results/22GENR/US%20REP.pdf), [North Carolina enacted shapefile](https://ncleg.gov/Files/GIS/Plans_Main/Congress_2019/HB1029%203rd%20Edition%20-%20Shapefile.zip), [Clean Investment Monitor](https://www.cleaninvestmentmonitor.org/us/methodology), and the repository's frozen MIT Election Lab, Census, and congress-legislators inputs.
