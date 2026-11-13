# Republican House takeover and clean investment

> Historical matched-design documentation. The current 21-control,
> overlap-weighted forest analysis and replication command are in
> [pretreatment_data/README.md](pretreatment_data/README.md). The estimates below
> are retained for provenance and are not the current headline specification.

## Result

Implemented a 2023 matched difference-in-differences analysis with honest causal forests for exploratory heterogeneity. The results do **not** establish a causal investment advantage or reliable heterogeneous effects. Remaining baseline imbalance, wide intervals, and concurrent party-specific policy exposure prevent confident causal conclusions.

The question is whether areas represented by Republicans gained investment or deployment relative to Democratic-represented areas when Republicans took the House majority in January 2023. Both groups change institutional status. This is a gainers-versus-losers comparison, not an estimate of gaining majority status relative to an unaffected control group and not an effect of the IRA or Section 45X.

The manuscript and all previous analyses remain unchanged.

## Main estimates

The primary sample begins with 196 eligible fixed geographic cores across 25 states, from the verified majority-alignment audit. Representatives and parties are unchanged across the transition. Matching retains **49 Republican–Democratic pairs, 98 areas, in 21 states**. Each pair receives equal weight.

Outcomes compare the annual average in 2023–2024 with the annual average in 2019–2022. Positive estimates indicate a relative increase in Republican areas.

| Outcome, per area-year | Matched relative change | 95% state-clustered interval | Linear-pretrend sensitivity |
|---|---:|---|---|
| Clean-power announcements | +0.464 | −0.773 to +1.702 | −0.448 [−2.306, +1.410] |
| Manufacturing announcements | +0.036 | −0.151 to +0.222 | −0.056 [−0.404, +0.292] |
| Net wind/solar operating capacity, MW | +25.28 | −0.22 to +50.77 | +24.36 [−17.08, +65.79] |

These intervals describe the matched DiD average. They are **not** confidence intervals for the unaugmented forest prediction mean. The mean full-forest out-of-bag predictions are +0.514 announcements, +0.024 manufacturing announcements, and +21.69 MW, respectively. Paired residual augmentation returns the observed paired contrast exactly, so the reported average and its inference remain transparent.

There are 346 post-period clean-power announcements across the matched areas, of which 225 are in Republican areas and 121 in Democratic areas. Manufacturing has 62 announcements, split 35 and 27. Zero-project areas remain in the sample and projects with undisclosed CAPEX remain in the counts. No dollar effect is estimated here.

## What the forest contributes

The forest estimates how the **paired relative change** varies across types of matched areas. It does not identify a separate causal effect for every district or a causal effect of the characteristics used as modifiers.

Each matched pair contributes two area-level changes. Both rows receive the same pair-average baseline features, their shared outcome midpoint is removed, and treatment is Republican/gainer versus Democratic/loser. With balanced within-pair residual treatment values of ±0.5, the regression slope equals the pair's DiD. The forest therefore learns an adaptive conditional mean of paired differences. The value 0.5 is a weighting identity, not a claim that parties were randomly assigned.

Four predetermined modifiers are used:

- Log of one plus wind/solar operating capacity in 2021.
- Log of one plus coal/gas/oil operating capacity in 2021.
- Log of one plus annual clean-power announcements averaged over 2019–2021.
- Log of one plus fixed-core land-and-water polygon area in square kilometers.

No post-2021 investment information enters matching or the modifiers. Geographic footprint is the fixed intersection of the 116th, 117th, and 118th district polygons. Area is computed from equal-area projected geometries simplified at 50 meters, while facility membership uses the original audited polygon assignments. This area is a size control, not population or economic output. We do not claim to have controlled for core-specific income, workforce, or complete industrial composition.

All honest forest sampling is clustered by state. There are 4,000 trees, minimum target leaf size four, no hyperparameter search, and fixed seeds. Full-sample out-of-bag predictions are exploratory outputs. A separate forest trains on **30 pairs in 13 states** and predicts **19 pairs in eight different states**. The training out-of-bag median defines high/low groups before evaluating held-out outcomes. The same split is used for every primary outcome. Earlier feasibility analyses inspected aggregate outcomes, so this is internal validation, not a preregistered external experiment.

## Heterogeneity validation

| Held-out forest high-minus-low contrast | Estimate | 95% interval | Unadjusted p |
|---|---:|---|---:|
| Clean-power announcements/area-year | +0.892 | −3.693 to +5.478 | 0.659 |
| Manufacturing announcements/area-year | +0.153 | −0.515 to +0.822 | 0.604 |
| Wind/solar net MW/area-year | +41.45 | −83.63 to +166.53 | 0.459 |

All three Holm-adjusted p-values are 1. No reliable effect ranking was demonstrated. The clean-power forest produces the same held-out high/low partition as a simple baseline-capacity rule in this split, so there is no demonstrated added ranking value from the more flexible model. The manufacturing forest's held-out prior-year placebo gap is +0.591 [−0.014, +1.195], which is a substantive warning rather than evidence that pre-trends are equivalent.

Forest groups are also evaluated on linear-trend-adjusted outcomes and a 2022-minus-2021 placebo. The placebos were not used in matching. Variable importance is not presented as evidence of causal drivers. Individual prediction intervals are not reported as precise CATE inference in this small clustered sample.

## Identification checks and limitations

Matching minimizes standardized distances in baseline features within each state, without replacement. Specifically, the assignment minimizes the sum of RMS Euclidean distances in five standardized variables, then drops pairs exceeding a 1.5-distance caliper. The fifth matching variable is the inverse-hyperbolic-sine 2019–2021 announcement slope. No post-period outcomes determine matches or calipers. This algorithm does not optimize the number of matches satisfying the caliper.

Matching improves several differences but **does not produce close baseline balance**. Remaining Republican-minus-Democratic standardized mean differences include +0.743 for log core area, +0.415 for log clean capacity, +0.287 for prior announcements, and +0.264 for fossil capacity. Prior announcement slope imbalance increases from +0.086 to +0.190. Pair-average forest features do not remove residual within-pair imbalance. Conditional parallel trends within these comparisons remain an assumption, not a demonstrated property.

The primary matched joint pre-year tests give p=0.135 for clean-power announcements, p=0.235 for manufacturing announcements, and p=0.526 for net wind/solar capacity over 2019–2022. The longer 2017–2022 capacity test gives p=0.749. These tests have limited power and do not establish equivalence of trends. Annual event-study plots show intervals and effects, not only p-values.

Other important limitations are:

- The national turnover supplies one common 2023 event. State clustering does not remove correlated party-specific national shocks or isolate political influence from heterogeneous IRA implementation.
- Both party groups change majority status. Common state shocks cancel within a pair only if they affect both areas additively in the same way. Different exposure to those shocks remains a confounder.
- Fixed cores avoid mechanical redistricting changes in measured outcomes, but do not remove redistricting's effects on constituencies or representatives' priorities.
- Commissioning in 2023–2024 may reflect investment decisions made before the takeover. RHG net MW changes include retirements, uprates, and revisions and are not gross investment dollars or new-factory announcements.
- The 2022 pre-period contains IRA implementation and potential anticipation of the election. A sensitivity excludes 2022 from the pre-period. Announcement effects between the election and January 3 are not separately identified.
- RHG has no coordinate-precision flag or generator identifiers. The audit verifies scenario duplication and spatial consistency but final causal work would still benefit from validation against EIA source records.
- The stricter continuous-member sample conditions on later survival in office and may introduce selection. It is a sensitivity sample, not automatically a better causal design.

## Sensitivity and sample changes

The full eligible-sample state-adjusted estimates and matched estimates are saved side by side. They target different populations. Main matching retains 49 pairs. Requiring continuous membership through 2019–2024 retains 33 pairs in 16 states. Requiring 80% geographic overlap retains only 14 pairs in 11 states. The latter supports average diagnostics but not a forced held-out forest validation. In the continuous-member sample, high/low subgroup support is inadequate for announcement validation and this is explicitly flagged.

Calipers of 1.0 and 0.75 retain 33 and 22 primary pairs. All calipers, pre-period choices, trend adjustments, and leave-one-state-out results are saved. For wind/solar capacity, one narrower caliper produces a positive interval excluding zero, but adjacent calipers do not. This isolated result is not treated as confirmation. No specification was selected for significance.

## Replication

Python uses `clean-investment-causal`. R uses GRF 2.6.1 and `sandwich`, with its session recorded. The current installation is in `/private/tmp/clean-grf-r46`; that directory is temporary. A durable installation in a normal R library is also supported, or set `TAKEOVER_R_LIB` to a custom library path. The code checks the custom/temporary library and then standard R libraries.

From the repository root, use a new output directory:

```sh
conda run -n clean-investment-causal python republican_takeover_analysis/run.py --out republican_takeover_analysis/rebuilt
```

The entry point refuses to overwrite an existing output directory. It reads the frozen verified feasibility inputs in `majority_alignment_audit/verified_run` and the supplied RHG source file. Rebuild that audit first if its source inputs change. No remote services or new data downloads are needed.

Outputs include the unit and pair panels, rejected and retained matching ledger, baseline balance, all-sample and matched estimates, annual pre-trends, joint tests, sensitivity and influence tables, honest forest predictions, held-out state assignments and validation, two figures, input/code/geography hashes, and numerical checks.

Verification reconciles pair identities and treatment direction, verifies state-disjoint evaluation, and independently reproduces every R paired estimate and interval in Python. A fresh end-to-end run reproduced all 22 generated artifacts byte for byte. Both figures were visually checked.

## Method references

The implementation uses the [GRF honest causal-forest API](https://grf-labs.github.io/grf/reference/causal_forest.html). The held-out effect-ranking exercise follows the separation of learning and evaluation discussed in [GRF's heterogeneity-validation guidance](https://grf-labs.github.io/grf/articles/rate.html), but it is a simple clustered high/low paired-DiD contrast rather than a native RATE estimate. Neither reference establishes identification for this political application.
