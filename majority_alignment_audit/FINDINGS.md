# House-majority alignment feasibility audit

## Assessment

The new data improve coverage and make a political majority-alignment forest technically feasible as an exploratory exercise. They do not establish causal identification or adequate precision for confirmatory heterogeneous effects. The strongest candidate is the January 2023 transition, using geographically fixed district cores. The 2019 comparison exhibits important pre-trend differences in the longer power-capacity history and should not be pooled into a clean causal estimate.

The question is whether investment or deployment changes differently in areas represented by a member whose party gains the House majority than in areas whose member's party loses it. Both groups change institutional status. The contrast is a **gainers-versus-losers relative effect**, not the effect of gaining majority status relative to remaining unchanged. National majority control itself has no district-level untreated comparison.

No manuscript, legacy result, or raw input was changed. No new majority-alignment forest was fitted during this feasibility audit.

## The manuscript's switch count is incorrect

The attached manuscript's appendix says **five**, rather than six, switches were removed by trimming. The legacy code defines treatment from the facility metadata's `US Representative Party` and clusters projects by `CD119_2024_Name`. That supplies a static contemporary label, not a dated history. Its apparent absence of turnover cannot establish that historical House seats did not change party.

Using dated member terms, December 15 before each transition and January 15 after it, the audit finds the following. Geographic comparability requires at least 99% of both old and new district area in their largest common intersection, measured in an equal-area projection.

| Term starts | Comparable major-party seats | Geographic party changes | R to D | D to R |
|---|---:|---:|---:|---:|
| 2019 | 414 | 41 | 39 | 2 |
| 2021 | 413 | 14 | 1 | 13 |

These are changes within the comparable subset, not national electoral flip totals or unique districts over the entire period. Vacancies, other parties, changed boundaries, and the snapshot dates affect the denominator. Comparing unadjusted seat numbers instead gives 50, 16, and 47 changes in 2019, 2021, and 2023. The last figure is particularly unsuitable as a geographic flip count because of redistricting. Only nine 2023 seats meet the strict geographic and major-party comparability rule.

The official House history confirms national majority changes into Democratic control in 2019 and Republican control in 2023. There was no majority change in 2021. These institutional transitions and individual seat flips are different quantities. Sources are the [House party divisions](https://history.house.gov/Institution/Party-Divisions/Party-Divisions/) and [majority changes](https://history.house.gov/Institution/Majority-Changes/Majority-Changes/). The local `seat_transition_ledger.csv` records the identities and exclusions underlying our counts.

## Geography and sample construction

All 435 districts enter the roster before joining projects. Maps cover the 115th, 116th, corrected North Carolina 117th, and 118th Congresses. The 118th map is generalized at 1:500,000, so overlap thresholds are approximate and coastal generalization can exclude otherwise stable seats. Thresholds of 99%, 95%, 80%, and 50% are shown together.

For 2023, an unchanged-boundary rule leaves only seven transition-stable members and five members continuous throughout the analysis window. A larger usable geographic unit is the **fixed common core** of the member's old and new districts. Each included point must fall inside every linked old polygon and the new polygon, irrespective of its announcement year. At the 50% rule, each old/new pair must share at least half of both areas. Outcomes cover the common core only, not the entire district. The larger-core thresholds were added after seeing the strict rule's attrition and are exploratory choices.

Two samples are retained. `transition_stable` requires the same member and party immediately before and after turnover. Later replacements and switches are retained as consequences. `continuous_member` additionally requires the same member and party throughout 2017–2020 for 2019, 2021–2024 for the short 2023 window, or 2019–2024 for the extended 2023 window. This stricter sample conditions on future survival in office and can introduce selection. It is a sensitivity sample rather than an automatic identification improvement.

State-adjusted comparisons use only states with both gainers and losers. Thus the raw 2023 extended-window core samples contain 254 transition-stable or 188 continuous-member districts, but the effective samples contain 196 districts in 25 states or 135 districts in 21 states. A district, rather than each project or quarter, is the potential forest observation. Changing core sizes require baseline exposure/population controls before a substantive conditional analysis.

## The new CIM release helps, but mostly for clean power

| Release | Facilities | Manufacturing | Precisely geolocated | Last announcement |
|---|---:|---:|---:|---|
| Frozen 2024Q4 | 8,532 | 581 | 8,484 | 2024-12-31 |
| New 2026Q2 | 10,721 | 711 | 10,635 | 2026-06-30 |

The new release contains 2,843 newly appearing IDs and omits 654 old IDs. These are not all new investments. The release bridge preserves ID additions, removals, and date changes. Old amounts are in 2023 dollars and new amounts in 2024 dollars, so this audit never pools their dollar levels. Both files retain announced projects subsequently canceled.

For comparability, political outcomes end in December 2024. Merely extending them into 2025–2026 would introduce another presidency and policy regime, 119th district maps, and representative changes. It would not add another House-majority transition. Both releases provide very little manufacturing announcement history before 2018. CIM's documented comprehensive investment baseline begins in 2018, so a 2017-to-2018 pre-trend comparison is especially weak. [CIM methodology](https://www.cleaninvestmentmonitor.org/us/methodology)

The following counts use the **new** release, common cores with at least 50% bidirectional overlap, pre-period 2019–2022, and post-period 2023–2024. Counts include undisclosed CAPEX.

| Effective sample | Districts | States | Manufacturing projects post | Districts with manufacturing post | Clean-power projects post | Districts with clean power post |
|---|---:|---:|---:|---:|---:|---:|
| Transition-stable | 196 | 25 | 113 | 66 | 922 | 133 |
| Continuous member | 135 | 21 | 88 | 47 | 520 | 88 |

Clean power means energy/industry solar, wind, and storage. Manufacturing remains separate. Before the within-state restriction, the continuous-member sample's post projects increase from 103 to 109 manufacturing and from 649 to 783 clean power between releases. The extra clean-power observations help precision, but manufacturing remains sparse.

The new district actual-spending file is not a facility-quarter panel. Its README describes 118th boundaries, while its field names use CD119 and metadata districts disagree with our 118th spatial assignments in 315 cases. We therefore do not treat it as a historically mapped spending panel or allocate whole-district spending into smaller cores. Facility-level actual expenditures or verified geography-specific aggregates would be needed for that extension. The included federal spending totals are modeled estimates of credit utilization, not observed district-level 45X claims.

## Announcement pre-trends and descriptive changes

Diagnostics use equal district weights, annual outcomes, state fixed effects, state-cluster CR1 standard errors, and t critical values based on the effective number of states. Quarterly pre-period slopes provide a separate check. Dollar outcomes preserve documented expenditure and do not treat unknown CAPEX as disclosed zero. None of these diagnostics creates random political assignment.

For the new-release 2023 continuous-member sample, the joint pre-year tests give p=0.189 for manufacturing counts and p=0.980 for clean-power counts. The corresponding annualized pre-quarterly slope differences are +0.011 manufacturing projects (95% interval −0.011 to +0.033) and +0.021 clean-power projects (−0.091 to +0.133). The pre-year level-contrast intervals are wider. Non-rejection does not demonstrate parallel trends, especially for sparse manufacturing events.

The descriptive post-minus-pre annual announcement differences are:

| Sample | Manufacturing projects/district/year | 95% interval | Clean-power projects/district/year | 95% interval |
|---|---:|---|---:|---|
| Transition-stable | +0.056 | −0.080 to +0.192 | +0.500 | −0.989 to +1.989 |
| Continuous member | +0.083 | −0.082 to +0.247 | +0.423 | −1.296 to +2.141 |

For 2019 clean-power announcements, the short pre-period's dollar outcomes already give warnings. In the continuous-member sample, pre-year p=0.029 for log CAPEX and p=0.0027 for dollar CAPEX. Counts alone would have missed this.

## RHG power-plant data provide a useful longer history

`plants_in_operation.csv` contains 1,178,883 rows across three scenario labels for 2005–2025. Their complete non-scenario row multisets, including multiplicities, are identical. The audit keeps only `ts_mid`, yielding 392,961 rows covering 14,074 plant IDs. Within-scenario identical rows are retained because generator IDs are absent and such rows may represent separate generators. Treating three scenarios as independent observations would triple the apparent data.

The retirement file includes historic, planned, and economic retirements, with future values extending to 2040. No retirement forecasts enter any outcome. The operating-capacity file's historical-looking records agree across scenarios, but their EIA provenance and the preliminary/final status of the last years still warrant validation before final estimation. Rhodium describes Taking Stock as an energy-system projection exercise. [Taking Stock documentation](https://rhg.com/research/taking-stock-2026/)

Plant capacity is aggregated by fixed common core and year. The outcome is **net annual operating-capacity change in MW**, including retirements, uprates, and revisions. It is not gross new-build MW, announcement counts, investment dollars, or generation. Earliest Operating Year is not used to date individual generators. Solar/wind and storage are separate outcomes.

The RHG `Storage` fuel category is not verified as battery-only and must not be equated to CIM battery-storage investment. No solar/wind/storage MW sum is used as the primary deployment outcome.

Unlike CIM, the RHG files have no coordinate-quality flag. The spatial checks verify valid coordinate ranges, unique polygon matches, and agreement with the reported state, but they do not establish facility-level location precision. Matching coordinates and generator identities back to EIA would be required before treating common-core assignment as fully validated.

For 2019, capacity pre-trends use 2013–2018, with 2019–2020 post. For 2023 they use 2017–2022, with 2023–2024 post. The extra history uses fixed footprints defined by the audited maps. It does not establish that the same representative served throughout the additional early pre-years.

**2019 raises a clear identification warning.** In the 273-district, 28-state continuous-member comparison, the joint wind/solar net-capacity pre-trend p-value is 0.0052. The gainer-minus-loser pre-slope is −6.30 MW of annual net additions per year (95% interval −10.05 to −2.55). A post-minus-pre estimate of −29.27 MW cannot credibly be read as a clean majority effect given that pattern.

**2023 is more promising but remains fragile.** In the continuous-member sample, the joint capacity pre-trend p-value is 0.107, and the pre-slope difference is +1.77 MW/year (−6.68 to +10.22). Only 64 of 135 districts show a positive wind/solar net addition in either post year. The estimated relative post-minus-pre change is +28.64 MW/district/year (0.18 to 57.10). Extrapolating each district's pre-period linear trend instead gives +21.57 MW (−26.95 to +70.09).

In the less-selected transition-stable sample, the unadjusted pretrend slope is more concerning at +9.27 MW/year (−0.99 to +19.54). The relative change of +32.53 MW (4.61 to 60.46) becomes −4.56 MW (−45.36 to +36.24) under pretrend extrapolation. These are exploratory model sensitivities, not alternative guaranteed counterfactuals. Tightening overlap to 80% leaves only 44 continuous-member districts in 11 mixed-party states and yields −8.08 MW (−98.20 to +82.03). Leave-one-state estimates at 50% remain positive but statistical significance often disappears. The modest p-values from selected specifications should not be treated as confirmation.

## Can a causal forest estimate heterogeneity reliably?

There are enough observations to fit a small exploratory forest for 2023. There is limited independent information for validating its heterogeneity.

In a fixed state-disjoint holdout of the 135-district continuous-member sample, the test set contains 65 districts in ten states. It has 37 manufacturing projects across 24 districts and 219 clean-power projects across 38 districts. Hundreds of projects do not become hundreds of independent treatment assignments.

An illustrative precision exercise assigns a known balanced subgroup without looking at outcomes, fits a state-adjusted treatment-by-subgroup contrast, and repeats 300 fixed-seed subgroup allocations. Using median cluster SE and a t/normal approximation to 80% power, the held-out minimum detectable subgroup gaps are about 0.86 manufacturing announcements per district-year and 4.72 clean-power announcements per district-year. The observed held-out outcome averages are only 0.285 and 1.685. For net wind/solar capacity, the illustrative held-out gap is roughly 111 MW/district/year. These calculations omit adaptive subgroup discovery and do not guarantee forest power or form a universal upper bound. The broader transition-stable sample improves counts but remains imprecise.

A defensible exploratory forest would use district-level changes or DiD scores, a small set of predetermined modifiers, state-disjoint nuisance fitting and evaluation, baseline levels and pre-trends, and conservative complexity. It would report an average relative effect before claiming a successful ranking. Baseline operating capacity from RHG is useful even if announcements remain the primary outcome. The forest cannot repair differential policy exposure or produce independent national majority shocks. [GRF cluster and honesty documentation](https://grf-labs.github.io/grf/reference/causal_forest.html)

## Recommended decision

Proceed only with a clearly exploratory **2023 majority-alignment** prototype, if retaining political heterogeneous-effects estimation is the priority. Keep the transition-stable sample as the less-selected comparison and the continuous-member sample as sensitivity. Use CIM announcements for investment siting, RHG capacity for baseline history and a distinct deployment outcome, and show all geography and trend sensitivities together.

Do not present a pooled 2019/2023 causal majority effect. Do not claim the IRA or 45X caused the estimated differential. IRA implementation and other party-specific shocks may affect Republican and Democratic districts differently exactly when the House majority changes. State clustering does not eliminate national party-specific confounding. More historical data improve the diagnosis, but neither new dataset supplies an exogenous instrument or an unchanged political control group.

## Reproduction and verification

The completed end-to-end build is saved under `verified_run/`. Its `VERIFICATION.json`, `latest/`, `frozen/`, `power_plants/`, and `supplement/` contain the final verified audit outputs. The other root-level result folders preserve earlier audit stages.

The scripts preserve raw inputs and fail before overwriting the main output directories. The verified analysis ran under `pypsa-earth` and was replicated in `clean-investment-causal`. The single entry point rebuilds both releases, a fresh replication, the power-capacity audit, support/precision checks, and verification in a new destination:

```sh
conda run -n clean-investment-causal python majority_alignment_audit/run.py --out majority_alignment_audit/rebuilt
```

Individual announcement-audit entry points are also available:

```sh
conda run -n clean-investment-causal python majority_alignment_audit/audit.py --out majority_alignment_audit/rebuilt_frozen
conda run -n clean-investment-causal python majority_alignment_audit/audit.py --out majority_alignment_audit/rebuilt_latest --facilities data/CleanInvestmentMonitor_Download/manufacturing_energy_and_industry_facility_metadata.csv
```

The optional `--crosswalk-cache` reuses a saved, unchanged geographic crosswalk and avoids repeating polygon intersections. The single entry point manages those caches and routes every derived output into the fresh destination.

The audit saves input hashes, district and member ledgers, core eligibility, spatial facility assignments, complete district-quarter/year panels, sample support, joint and individual pre-trend diagnostics, state-disjoint holdout counts, influence checks, and figures. `verify.py` reconciles annual and quarterly sums, project disclosure identities, all support rows, independently checks the headline capacity coefficient, and verifies eight dated political assignments. A fresh latest-release run reproduced all **17** artifacts byte for byte. Figures were visually checked. See `VERIFICATION.json`.
