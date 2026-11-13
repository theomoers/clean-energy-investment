# Can closure-boundary variation support a causal forest?

**Decision: no-go for a confirmatory heterogeneity claim about solar, wind and storage announcements with CIM through 2024Q4.** There is usable geographic variation. Too few closure regions contributed post-policy announcements for credible out-of-region validation of moderately different responses. This is a precision assessment, not an estimate that the bonus had zero effect.

## Where the observations went

The primary sample has 3,563 same-state boundary links connecting 1,191 inside and 2,540 outside tracts in 39 states. Each side contributes unique tracts within each region. Contiguous designated zones and zones sharing outside controls are grouped before applying the sample restrictions, including connections across state lines. This gives 123 non-overlapping geographic clusters. Additional common electricity-market shocks could correlate them. These clusters are a proxy for independent closure regions, not 123 randomized assignments.

| Boundary screen | Regions | Post projects inside / outside | Regions with a post project on both sides | Regions with projects on both sides in both periods |
|---|---:|---:|---:|---:|
| Initial outside boundary, before alternative-route or date screens | 163 | 103 / 149 | 20 | 17 |
| Remove counties eligible through the statistical-area route in either vintage | 133 | 39 / 65 | 10 | 6 |
| Also require a verified pre-IRA closure next to the inside tract (primary) | 123 | 36 / 60 | 8 | 5 |
| Primary, same-county boundary links only | 123 | 29 / 35 | 6 | 3 |
| Primary, core closure preceding 2018 | 97 | 28 / 43 | 5 | 4 |
| Primary, excluding whole tracts containing a known EPA brownfield on either side | 113 | 18 / 27 | 3 | 0 |

There were 97 pre-period projects inside and 180 outside in the primary sample. Of the 96 post announcements, 68 were solar, 16 storage and 12 wind. Only 39 regions had any post announcement. The largest accounted for 22 of 96. The inverse concentration of event shares, 13.4, measures concentration, not a regression sample size. Zero-event regions stay in the data and in the power calculations.

## Can a forest ranking be validated?

For each region, compute the inside-minus-outside change in annual announcements per 100 tracts. Compare 2018–21 to July 2023–December 2024; omit 2022 and the first half of 2023 as a transition period. The comparator rate after the transition was 1.575 projects per 100 tract-years (pooled over outside tracts). The equal-region average outside rate was 2.001.

In 5,000 seeded splits, randomly allocate distinct regions to balanced high/low groups, keeping either half or all regions for a validation contrast. Use the empirical region contrasts as noise; add a known uniform effect difference to the high group. Evaluate two-sided 5% Welch tests. The high/low labels are supplied without estimating them: an actual forest must additionally learn and generalize the ranking. This is an illustrative oracle calculation for additive, balanced region-level heterogeneity. It is neither universal power for every possible moderator nor a simulation of exogenous assignment. Size, predictability of residual noise, subgroups, dependence, and effect scale matter.

| Validation scenario, primary geography | Power for a gap of 0.79 | Power for a gap of 1.57 | Gap detectable with 80% power |
|---|---:|---:|---:|
| Hold out 61 regions, equal region weights | 5.3% | 10.5% | 7.42 |
| Hold out 61 regions, weight by effective tract counts | 7.8% | 15.8% | 5.17 |
| All 123 regions available, equal weights, groups supplied | 7.5% | 15.4% | 5.08 |
| All 123 available, effective tract-count weights, groups supplied | 11.6% | 24.3% | 3.68 |

Gaps are extra announcements per 100 tract-years in one subgroup relative to the other. The half-sample equal-region design rejected in 3.3% of null simulations. The all-region tract-weighted case rejected in 6.6%; its higher power partly reflects a more liberal test. Weights are proportional to the harmonic combination of inside and outside tract counts. The relaxed geographic screen still gives only 25% power for a gap equal to its comparator rate with half the regions held out and equal weights.

## Identification constraints

- The official **deployment-bonus** coal-closure list in IRS Notice 2023-29 supplies directly adjoining treated tracts. A comparison tract must remain outside the cumulative list through Notice 2024-48. The sample estimates support for a stable cohort rather than assigning retroactive project-specific credit eligibility. The statutory treatment is access to an incentive; announcements are a siting outcome. Beginning-of-construction and placed-in-service rules matter for an actual credit claim. [Treasury eligibility files](https://home.treasury.gov/policy-issues/tax-policy/data-transparency/all-treasury-generated-energy-communities-data-sets) · [IRS rules and timing](https://www.irs.gov/credits-deductions/frequently-asked-questions-for-energy-communities)
- We verified a qualifying-type historical event in 970 official core tracts using MSHA abandoned surface/underground coal mines and EIA coal-fueled generator retirement dates. A primary inside tract must share a positive-length border with one of these cores. Retrospective MSHA records, plant-level coordinates, a generalized Census boundary, and incomplete source matches can exclude true events. Comparing strictly within counties is a sensitivity test. Electric-grid connectivity, queue access and suitable land are not balanced by construction. [MSHA](https://arlweb.msha.gov/OpenGovernmentData/OGIMSHA.asp) · [EIA data distributed by PUDL](https://docs.catalyst.coop/pudl/en/v2026.1.0/data_access.html) · [2020 Census polygons](https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_tract_500k.zip)
- The empirical pre-policy difference in changes from 2018–19 to 2020–21 averaged −2.06 announcements per 100 tract-years across regions (approximate region-level 95% interval −4.36 to +0.24). The magnitude is substantial relative to the post comparator rate. The audit does not establish parallel trends.
- Brownfield eligibility is parcel-specific. The sensitivity drops a whole tract when a known brownfield point appears in EPA's current inventory. This over-excludes many parcels, can include later entries, and misses undocumented brownfields. The screen is not a historical eligibility classification. Of the primary tracts, 1,188 of 1,191 inside and none outside were also in the first-round **manufacturing** 48C set-aside geography. That is a separate program: the overlap raises a potential concurrent local-investment channel, not evidence that deployment projects received 48C. [EPA cleanup sites](https://services.arcgis.com/cJ9YHowT8TU7DUyn/arcgis/rest/services/Cleanups_in_my_Community_Sites/FeatureServer) · [48C Round 1 Appendix C](https://www.irs.gov/irb/2023-25_IRB)

## Reproduce and use the results

From the repo root run `conda run -n pypsa-earth python energy_community_audit/audit.py`. That existing environment has GeoPandas, PyArrow, openpyxl, SciPy and NetworkX together. No package installation or change to the main research environment was needed. The downloaded inputs and SHA-256 manifest are included. The MSHA inventory snapshot was dated October 30, 2026. EIA plant and generator Parquet inputs use PUDL release v2024.11.0 on its public S3 bucket. The bundled EPA snapshot contains 64,215 points; `fetch_brownfields.py` can refresh it, which can change results.

Read `results/support.csv` and `results/oracle_power.csv` first. The boundary links, flags, closure verification, project IDs, tract memberships and region-level outcomes are exported for independent inspection. To revisit a forest design, extend the project cohort or add a clearly defined, independently geolocated outcome, then repeat the same region-level power gate. Select the smallest policy-relevant subgroup gap before fitting a forest.
