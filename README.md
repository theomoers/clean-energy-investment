# Partisanship and Clean-Energy Manufacturing Investment

Analysis code for the 2023 Republican House takeover, overlap-weighted
investment changes, and honest causal forests. This repository contains code,
derived replication inputs and numerical reference outputs, not the manuscript.

## Current analysis (November 2026)

Start with the [current analysis and replication instructions](republican_takeover_analysis/pretreatment_data/README.md).
The question is whether Republican-represented areas experienced greater
manufacturing-announcement growth relative to Democratic-represented areas
following the Republican House takeover. The sample contains 196 fixed
geographic areas in 25 states; the forest learns in 15 states and is evaluated
in ten different states.

- No statistically significant average manufacturing response is detected:
  0.080 announcements per area-year, 95% interval [−0.074, 0.235]. This does not
  establish a zero effect.
- The expanded manufacturing forest shows suggestive ranking heterogeneity:
  QINI 0.054, unadjusted p=0.030; six-test Holm p=0.177. This is exploratory,
  specification-dependent evidence, not a validated subgroup causal finding.
- Local economic characteristics are adjustment variables and potential
  modifiers, not experimentally identified investment determinants. Conditional
  parallel trends, comparison support and concurrent IRA exposure remain
  important limitations. This design does not estimate the IRA's causal effect.

```sh
conda env create -f environment.takeover.yml
conda activate clean-investment-causal
# Install R and GRF as described in the replication instructions, then:
python republican_takeover_analysis/pretreatment_data/replicate.py \
  --from-derived --out republican_takeover_analysis/pretreatment_data/rebuilt
```

Large source downloads are excluded. The included frozen area panel, annual
outcomes, covariates and baseline ranking support rebuilding the statistical
analysis without manuscript files. Source definitions and download provenance
are documented in `republican_takeover_analysis/pretreatment_data/DATA_SOURCES.txt`
and `download_manifest.json`. Raw-to-panel reconstruction additionally needs
the source files and earlier geographic audit inputs; it is distinct from this
derived-data replication command.

## Legacy project-level analysis

The notebooks, original `scripts/`, and earlier data/results below are preserved
for provenance. They are not the current takeover specification. In particular,
the original 33.8% project-CAPEX estimate should not be interpreted as a causal
effect: the earlier code has assignment, sample-selection and inference issues.
See the current design's documentation for the revised research question and
the distinction between project size and announcement-count growth.

### Original setup and notebook guide

> April 2024 <br>
> Columbia University, SIPA <br>

<br>
Data is taken from the Clean Investment Monitor (Rhodium Group): https://www.cleaninvestmentmonitor.org/database/

## Setup
```bash
git clone https://github.com/theomoers/clean-energy-investment.git
cd clean-energy-investment
conda create -n clean-energy-CF python=3.9 pip -y
conda activate clean-energy-CF
pip install -r requirements.txt
```

## Running the analysis
The code is developed for jupyter notebooks (`.ipynb`). It can also be run using the `.py` files but there are some data downloads and bootstrapping processes that take quite long. It is easier to circumvent those by using the notebooks. <br>
<br>
- `data_exploration.ipynb/.py` includes the code for the introduction of the paper. <br>
- `political_analysis.ipynb/.py` includes the code for a preliminary political analysis of partisan dominance in each congressional district from 2018-2024. <br>
- `data_cleaning.ipynb/.py` includes the code for the assembly of the data set (downloading NASA POWER, ACS district data, and Clean Investment Monitor data). <br>
- `causal_forest_implementation.ipynb/.py` includes the code for the causal forest, the trimming of the data set, the weighting, calibration, optimization, and validation of the causal forest, and the plots and statistics for the result section. It also includes the DiD regression. <br>

### Key dependencies
```numpy, pandas, matplotlib, seaborn, geopandas, cartopy, us, scikit-learn, econml, statsmodels, linearmodels, scipy, tqdm, requests, censusdata, textwrap```


