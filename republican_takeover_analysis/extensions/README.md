# Republican takeover: balance, precision, and the 2011 replication audit

## Findings

Completed the requested equivalence/power calculations, improved baseline balance, audited the 2011 Republican takeover, retained honest causal forests, and examined technology categories and states. The manuscript and previous analyses are unchanged.

**The improved 2023 analysis finds no statistically detectable average effect. It does not establish that the effect is zero or economically negligible. Neither the category comparisons nor the held-out forests establish reliable heterogeneous effects.** The 2011 data provide more geographic comparisons but show substantial specification sensitivity.

The question remains relative: did Republican-represented areas change relative to Democratic-represented areas when Republicans gained the House majority? Republican incumbents gain majority alignment while Democratic incumbents lose it. There is no unaffected party group. Neither the forest nor the national takeover by itself makes party assignment exogenous. This is not an estimate of the causal effect of the IRA or Section 45X.

The numerical tables below use `results_current/`. `replication_current/` is the fresh rebuild. The extension protocol was recorded before fitting these extensions, but earlier results were already known, so this is exploratory work, not a preregistered study.

## 2023 results after improving balance

Effects are Republican-minus-Democratic changes in annual outcomes, comparing 2023–24 with 2019–22. Intervals use state-clustered inference.

| Outcome per area-year | Original matching, 95% CI | Balance-constrained matching, 95% CI | New matching p-value | Overlap-weighted, 95% CI |
|---|---|---|---:|---|
| Clean-power announcements | +0.464 [−0.773, +1.702] | −0.030 [−0.773, +0.712] | .933 | −0.428 [−1.304, +0.448] |
| Manufacturing announcements | +0.036 [−0.151, +0.222] | +0.106 [−0.070, +0.282] | .221 | −0.034 [−0.331, +0.262] |
| Net wind/solar capacity, MW | +25.28 [−0.22, +50.77] | −0.02 [−23.70, +23.66] | .999 | +19.12 [−14.14, +52.38] |

The new matching retains **33 pairs, 66 areas, across 19 states**, versus 49 pairs across 21 states originally. Its 2023–24 sample contains 176 clean-power announcements and 39 manufacturing announcements. Zero-project areas remain included. Announcement counts include projects with undisclosed expenditure and subsequent cancellations. These are not investment-dollar estimates.

The largest absolute standardized mean difference across nine measured baseline characteristics falls from **0.743 to 0.098**. Characteristics are the original five matching variables plus manufacturing announcement levels/slopes and net wind/solar capacity-change levels/slopes during 2019–21. All use information before 2022. No post-takeover outcome determines matching.

Matching first maximizes the number of non-reused, within-state pairs satisfying all nine absolute mean-balance constraints of 0.10 pooled eligible-sample SDs and the original five-variable distance caliper of 1.5. It then minimizes total distance holding that maximal count fixed. Both optimization stages certify optimality. Pair selection is not based on significance.

Overlap weights retain all 196 eligible areas in 25 states but emphasize comparable areas. An unpenalized logistic model includes state indicators and the same nine covariates. Weighted mean imbalance is below 0.000001 SD. Kish effective sample sizes are approximately **34 Democratic and 51 Republican areas**, and the concentration-adjusted state count is about 16.5. Weight estimation is repeated in a 999-draw state bootstrap; 998 draws converge to the prespecified score tolerance. Repeated state draws retain their multiplicity. Main intervals use the bootstrap SE and a t reference with 24 degrees of freedom; percentile intervals and every successful draw are also saved.

These estimators target different comparison populations. The movement from +25 MW to approximately zero is therefore not an estimate of how much bias was removed. Exact balance of included means also does not establish equality of distributions: the new matching has variance ratios up to 2.24 and a maximum weighted KS distance of 0.273. Overlap weighting has variance ratios up to 2.00 and KS distance 0.205.

No income, population, renewable-resource or grid-access variable has been fabricated for the geographic cores. Existing whole-district socioeconomic data are not geography-matched to the partial district intersections. Those controls remain missing. Measured balance does not remove omitted confounding, heterogeneous IRA exposure, regression-to-the-mean concerns from conditioning on noisy pre-outcomes, or correlated national party-specific shocks.

## Significance, equivalence, and power

The user clarified that “no effect” meant statistical nonsignificance, not a chosen substantive equivalence threshold. Accordingly, no equivalence bound is labeled agreed or substantively negligible.

The calculations distinguish:

- A conventional two-sided test of zero, with 95% intervals.
- Two one-sided equivalence tests at 5%, equivalent to requiring a **90% interval** to lie entirely within the specified bounds.
- Power to detect a specified nonzero effect, rather than retrospective “observed power” at the point estimate.
- Illustrative precision needed to demonstrate practical equivalence if the true average effect were zero.

| New 2023 matching | Pre-period annual baseline | Approximate 80%-power minimum detectable effect | Smallest symmetric bound containing the 90% interval |
|---|---:|---:|---:|
| Clean-power announcements | 0.924 | 1.047 projects | ±0.643 projects |
| Manufacturing announcements | 0.121 | 0.248 projects | ±0.251 projects |
| Net wind/solar capacity | 24.50 MW | 33.39 MW | ±19.56 MW |

Thus the current design is powered mainly for changes comparable to or larger than its baseline outcomes. It cannot confidently distinguish modest effects from zero. The last column describes present precision; it is **not** a proposed economic definition of negligible.

The complete grid includes relative bounds of ±10%, ±25%, and ±50% of the pooled pre-period baseline. **None of these relative bounds establishes equivalence for a main average effect in the reported designs.** Absolute bounds are also reported: clean announcements ±0.1/0.2/0.5/1; manufacturing ±0.025/0.05/0.1/0.2; capacity ±5/10/25/50 MW. Some broad absolute bounds pass—for example, ±50 MW in the new 2023 matching—but that bound is approximately twice the baseline annual capacity increase. It cannot reasonably be described as establishing zero. The ±25 MW bound passes the unadjusted test (p=.020), but not its Holm-adjusted family (p=.060).

Holm corrections are saved for the three primary outcomes, six category outcomes, and 15 pairwise category-effect differences. Equivalence corrections group the three outcomes within each position of the prespecified bound grid; absolute bounds have different units across outcomes. A conservative additional correction covers the primary-outcome average estimates and their trend/pre-period sensitivities in `estimates.csv`. The separate unmatched 2011 feasibility diagnostics are exploratory and are not included in that latter family.

Minimum detectable effects use a noncentral-t planning approximation with state-cluster degrees of freedom and the estimated SE as a design input. Numerical gamma quadrature evaluates the normal/chi-square mixture because the installed SciPy noncentral-t function produced nonfinite values. Null rejection probability is checked against .05. This does not make small-cluster inference exact.

Equivalence-power and information-multiplier columns use an explicitly labeled **fixed-SE normal approximation**, not unconditional repeated-sampling power. A reported zero means that, at that fixed SE, even an estimate exactly at zero cannot pass the equivalence criterion. Scaling scenarios hold the design structure fixed and imagine independent replicated information; they are not literal recommendations for more states, more project rows, or a universal forest sample-size requirement. For example, the ±25%-baseline benchmark requires approximately 21, 69 and 31 times the information in the new matched design for clean announcements, manufacturing and capacity, respectively, under this approximation.

## Categories, states, and causal forests

Six mutually exclusive technology outcomes are analyzed using the same samples and weights: solar, wind and storage clean-power announcements; battery, solar and other manufacturing announcements. Their counts reconcile exactly to the two parent announcement outcomes.

No category has a statistically significant average effect after correction. All six category p-values in the balanced 2023 design become 1 under Holm adjustment. Pairwise differences between category effects are also supplied; a significant result in one category and not another would not itself demonstrate a difference.

Support is thin. In the new matched sample there are **zero post-period wind announcements**, 10 battery manufacturing announcements, 15 solar manufacturing announcements, and 14 other manufacturing announcements. A wind coefficient can still arise from differences in the pre-period, but these zero post-events do not support a category-specific causal claim. The support table flags these cases explicitly.

`state_descriptive.csv` gives every state's matched-pair count, pre/post means and relative changes. State-specific contrasts are labeled descriptive: there is only one state cluster within each state, so the national state-clustered inference cannot simply be reused for state-specific p-values. Individual projects are not independent political treatment assignments.

The forests remain actual `grf::causal_forest` models with 4,000 trees, honest state-clustered sampling, fixed seeds/hyperparameters and predetermined pair-average features. They learn heterogeneity in paired DiD contrasts. Pair residualization and `W.hat = 0.5` are a weighting identity, not random assignment. They do not identify causal effects of the features or precise individual-district treatment effects.

| Held-out forest high-minus-low contrast | Estimate | 95% interval | p |
|---|---:|---|---:|
| 2023 clean-power announcements | −0.717 projects | [−3.034, +1.601] | .478 |
| 2011 net capacity, transition-stable | +46.85 MW | [−29.04, +122.74] | .196 |
| 2011 net capacity, continuous-member sensitivity | +14.34 MW | [−5.86, +34.54] | .143 |

The 2023 forest trains on 19 pairs in 12 states and evaluates 14 pairs in seven other states. Manufacturing and capacity predictions put all held-out observations on the same side of the training-defined threshold, so high/low validation is not estimable; the code flags this rather than changing the threshold to obtain a result. The larger 2011 primary forest trains on 38 pairs in 17 states and evaluates 43 pairs in ten other states.

No main held-out heterogeneity contrast is statistically significant. The 2011 transition-stable forest's apparent +47 MW ranking gap becomes approximately −4 MW under linear-pretrend adjustment. Subgroup covariate imbalance also reappears: the largest within-group SMD is approximately 0.51 for 2023 clean announcements and 0.31 for the 2011 forests. Overall balance does not establish conditional balance inside learned subgroups. Thus the forests are retained as a transparent, validated ML component, not claimed as confirmed causal heterogeneity.

## 2011 Republican takeover feasibility

Downloaded the official Census 112th-Congress boundary file. Census documentation confirms that these district boundaries also apply to the 110th and 111th Congresses, covering **2007–2012**. Georgia and Texas changed before this window, between the 109th and 110th Congresses. No 2013 redistricting is included.

The audit starts with all 435 seats. It compares dated member/party records on December 15, 2010, and January 15, 2011, and verifies party against the local MIT 2010 general-election winner after aggregating fusion lines by candidate. All 435 serving-party labels agree with the reconstructed winner party; only the 340 seats with the same member and party at the transition are eligible. One vacancy/other-party transition and 94 member/party changes are excluded.

Of those 340 seats, **301 in 30 states** have within-state opposite-party comparisons. Balance-constrained matching retains **81 pairs in 27 states**. The stricter same-member-throughout sensitivity has 247 mixed-state seats and retains **69 pairs in 26 states**. Source terms occasionally begin at a January 4/5 swearing-in rather than January 3. Only short gaps during odd-year January Congress openings are tolerated for this sensitivity and counted in the roster. Other gaps or member/party changes fail continuity. Conditioning on survival through the post-period is itself a selection concern.

RHG operating capacity is available throughout this window. The outcome is annual net wind/solar operating-capacity change, pre 2007–10 versus post 2011–12. It is not manufacturing investment, gross commissioning, generation or announcement counts. Only one of the three identical scenarios is used; within-scenario identical rows are retained because missing generator identifiers make deduplication unsafe. All years are spatially assigned to the fixed historical map.

The spatial audit matches 14,026 distinct plant-coordinate-state locations across all available source years; 24 are outside polygons, 23 disagree with the source state, and one has invalid coordinates. The map audit flags 204 locations within approximately 100 meters of a boundary; this is not a guarantee of coordinate precision. Within the 301 eligible mixed-state districts, the post-period has 547 operating wind/solar plant IDs in Republican areas and 325 in Democratic areas; 230 and 158, respectively, have a positive plant-level net capacity change in at least one post year. These are support counts, not new announcements or independent political assignments. Coverage by year/fuel and plant-level support are saved. Historical plant records in the supplied retrospective release have not independently been reconciled to original EIA generator files.

| 2011 comparison, net MW per area-year | Estimate | 95% interval | p |
|---|---:|---|---:|
| All 301 mixed-state eligible districts, state-adjusted | +9.98 | [−11.74, +31.70] | .355 |
| 81 balanced pairs | −15.65 | [−31.77, +0.46] | .056 |
| Balanced pairs, linear-pretrend sensitivity | −4.25 | [−17.50, +8.99] | .515 |
| 69 continuous-member balanced pairs | −0.28 | [−10.20, +9.64] | .955 |

The unmatched linear-trend-adjusted estimates are nominally positive: +23.25 MW (p=.038) and +29.65 MW (p=.029) in the continuous-member sensitivity. They are preserved in the audit outputs. They do not persist in the balanced designs and are not selected as the result. The differences in population and assumptions across these estimates preclude treating the nominal significance as a robust causal discovery.

The 2011 joint pre-trend test is p=.395 in the all-eligible comparison and p=.268 in the matched comparison. The continuous-member matched test is p=.076. The 2023 matched tests are p=.880 for clean announcements, .305 for manufacturing, and .375 for the longer 2017–22 capacity history. These are nonrejections, not evidence of equivalent trends. Matching uses early pre-outcomes, so the post-matching pretests are partly design-induced; the separate last-pre-year placebo and trend-extrapolation sensitivities are also reported.

**Feasibility verdict:** 2011 supplies enough geographic units to run the forest and a useful independent political-event audit, but not a stable causal finding. It is not pooled with 2023: the periods differ, only capacity is observed historically, and both samples are sensitive to design assumptions. The larger number of districts does not create many independent national majority transitions.

## Replication and deliverables

Run from the repository root, with a new output directory:

```sh
conda run -n clean-investment-causal python republican_takeover_analysis/extensions/run.py --out republican_takeover_analysis/extensions/rebuilt
```

Python uses the existing `clean-investment-causal` environment; R uses GRF and sandwich. The R script checks `TAKEOVER_R_LIB`, defaulting to `/private/tmp/clean-grf-r46`, then standard libraries. The temporary library must remain available or be replaced with a durable installation. No new package installation was required.

The build reads frozen verified audit inputs and existing matched inputs, plus supplied raw CIM/RHG/MIT/dated-term files and the newly downloaded Census map. It refuses to overwrite an output directory. Default bootstrap count is 999. The smaller `--bootstrap-reps` option is for development, not the reported results.

Key deliverables:

- `estimates.csv`, `equivalence.csv`, `power_summary.csv`: all effects, intervals, significance, equivalence and planning calculations.
- `balance.csv`, matching ledgers, `overlap_weights.csv`, `overlap_support.csv`: design and overlap audits, including distributional diagnostics.
- `matched_support.csv`, `state_descriptive.csv`, `category_effect_differences.csv`: category/state support and comparisons.
- `forest_pairs.csv`, `forest_predictions.csv`, `forest_validation.csv`, `forest_subgroup_balance.csv`: actual forest inputs, honest predictions, validation and subgroup-balance warnings.
- `audit2011/`: historical roster, dated terms, geographic exclusions, zero-inclusive annual panel, plant support, pre-trends and sensitivity estimates.
- Four PNG figures: estimates across designs, baseline balance, 2011 pre-trends and held-out forest validation.
- `verification.json`, input/code hashes and runtime settings: reproducibility checks.

The independent verification checks political direction and same-state pairing, no unit reuse, balance constraints, state-disjoint forest evaluation, all paired SEs against an independent statsmodels calculation, TOST/90%-interval equivalence, and the 435-seat historical roster. There are 25 passing checks. A separate full rebuild reproduces all 51 generated files byte for byte. All four scientific figures were visually reviewed. Scientific-data handling follows a flat-table audit workflow: raw observations are preserved and derived records, exclusions, calculations and results are separate. No manuscript file is edited. Superseded development outputs are retained under `development_archive/`, not used for the findings.

## Method and geography sources

- [Census historical district-map documentation](https://www.census.gov/geographies/reference-maps/2005/geo/109th-112th-congressional-district-maps.html) and [112th-Congress TIGER download](https://www2.census.gov/geo/tiger/TIGER2011/CD/tl_2011_us_cd112.zip).
- [Cardinality matching documentation](https://search.r-project.org/CRAN/refmans/designmatch/html/cardmatch.html).
- [Li, Morgan and Zaslavsky: balancing and overlap weights](https://www2.stat.duke.edu/~fl35/papers/psweight_final.pdf).
- [TOST hypothesis-testing documentation](https://aaroncaldwell.us/TOSTERpkg/articles/hypothesis_testing.html).
- [GRF reference, including honesty and cluster sampling](https://grf-labs.github.io/grf/REFERENCE.html).

The appropriate current statement is: **“We find no statistically detectable relative effect in the balanced 2023 comparison, and no reliably validated heterogeneous effects. The intervals remain too wide to rule out economically meaningful effects, and identification remains conditional on unverified assumptions.”**
