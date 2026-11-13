# Current analysis: 2023 House takeover

This is the current empirical specification. The parent directory's matched
design and earlier overlap/forest exercises are historical development steps,
not interchangeable estimates of the same target.

## Research question and interpretation

Do Republican-represented areas experience greater manufacturing-announcement
growth relative to Democratic-represented areas after the Republican House
takeover? Both parties change majority alignment; there is no unaffected group.
The design uses 196 fixed geographic cores in 25 states, unchanged members and
parties, and annual-average changes from 2019–2022 to 2023–2024. Announcement
counts retain undisclosed CAPEX and zero-project areas.

Baseline adjustment uses nine predetermined characteristics. Expanded adjustment
adds twelve socioeconomic, renewable-resource, grid-proximity and IRA-related
exposure measures. The honest GRF learns on 95 areas in 15 states and is
evaluated on 101 areas in ten different states. The state split is fixed.

No statistically significant average manufacturing response is detected:
expanded estimate **0.080 announcements per area-year**, 95% interval
**[−0.074, 0.235]**. This is not evidence of a zero effect. The expanded
manufacturing QINI is **0.054 [0.007, 0.101]**, nominal p=0.030 and six-test
Holm p=0.177. It is exploratory ranking evidence, not a percentage investment
effect or a validated subgroup policy recommendation. See `FINDINGS.txt`,
`PROTOCOL.txt`, and `DATA_SOURCES.txt` for qualifications and source definitions.

## Setup

From the repository root:

```sh
conda env create -f environment.takeover.yml
conda activate clean-investment-causal
```

Install R separately if it is not already available. Install GRF in a persistent
user library (the original analysis used GRF 2.6.1):

```sh
Rscript -e 'dir.create(Sys.getenv("R_LIBS_USER"), recursive=TRUE, showWarnings=FALSE); install.packages("grf", lib=Sys.getenv("R_LIBS_USER"), repos="https://cloud.r-project.org")'
```

Set `RSCRIPT` if the executable is not on PATH. Set `TAKEOVER_R_LIB` if GRF is in
a custom library. For exact numerical replication use GRF 2.6.1; newer versions
may change fitted forests. Python numerical-library versions and BLAS may also
affect numerical tolerances. Saved `R_session.txt` files document the original fit.

## Rebuild from included derived inputs

```sh
python republican_takeover_analysis/pretreatment_data/replicate.py \
  --from-derived \
  --out republican_takeover_analysis/pretreatment_data/rebuilt
```

The output directory must not exist. This runs both specifications, 999 state
bootstrap draws, subgroup-support diagnostics, prediction checks and pre-trend
checks, plus the six full-sample OLS association models. It needs no raw-source
downloads or manuscript files. It refits the
expanded forest and reuses the frozen earlier baseline ranking; it does not
retrospectively refit that baseline with a different nuisance specification.

Required derived inputs are committed at their historical paths:

- `covariates.csv` and `features.json` in this directory;
- `../overlap_forest/results_current/areas.csv`, with outcomes and fixed split;
- `../heterogeneity_upgrade/results_current/priorities.csv`, the frozen ranking;
- `../extensions/results_current/annual2023.csv`, used for pre-trend diagnostics.

These small derived inputs make the statistical stage reproducible, not the
entire raw-data collection and geographic construction. No claim is made that
`--from-derived` re-audits facility locations, election assignments or source
downloads. `download_manifest.json` records the original source URLs and hashes;
large downloads are not committed. `build.py` and `download.py` preserve the raw
construction code, which additionally requires the earlier boundary and dated
closure inputs described in `DATA_SOURCES.txt`.

## Verify saved results without changing them

```sh
python republican_takeover_analysis/pretreatment_data/verify.py \
  republican_takeover_analysis/pretreatment_data/results_current \
  --skip-raw --no-write
```

Compare a fresh rebuild against the frozen numerical results:

```sh
python republican_takeover_analysis/pretreatment_data/verify.py \
  republican_takeover_analysis/pretreatment_data/rebuilt \
  --skip-raw --no-write \
  --compare republican_takeover_analysis/pretreatment_data/results_current
```

The comparison checks every generated CSV except the code/input hash ledger,
which changes when portability code is updated. Original hash ledgers and
verification reports are preserved. Skipping raw checks is explicit and is
recorded in newly generated verification reports.

## Identification and inference

Conditional parallel trends, adequate overlap and absence of differential
concurrent shocks remain assumptions. The national takeover is one common
event; state clustering does not create independent political shocks. Expanded
controls improve overall measured balance but reduce effective sample sizes,
and no subgroup design meets every support criterion. Evaluation states were
inspected in earlier work. Intervals hold learned forest/outcome models fixed.
This is not an estimate of the IRA or Section 45X credit's causal effect.

## Publication validation (November 13, 2026)

The staged Git files were exported into a fresh temporary directory, without
local manuscript files or uncommitted raw/audit inputs. The complete
`--from-derived` command succeeded using GRF 2.6.1 and the existing
`clean-investment-causal` Python runtime. All 45 generated numerical CSVs
(excluding the provenance hash ledger) matched the frozen references within
1e-9 relative/absolute tolerance; verification passed 134 checks. The six OLS
models also matched the earlier independently verified regression outputs.
Raw-source collection and geographic reconstruction were not rerun in this
publication check. The supplied environment file documents dependencies; a
separate fresh-environment installation was not tested.
