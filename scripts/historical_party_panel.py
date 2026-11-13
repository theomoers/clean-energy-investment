"""Rebuild a historically dated House-party / project-announcement panel.

This is deliberately separate from the legacy causal-forest script.  It treats
the Clean Investment Monitor facility file as an *observed-project* outcome:
zeroes mean no announcement among projects observed by CIM, not necessarily no
private investment.  It therefore produces a fixed-effects association and
diagnostics, not a final causal estimate.

Required raw inputs (already staged under data/redesign_raw/):
  * 115th, 116th, and 118th congressional district boundaries;
  * congress-legislators JSON histories (CSV summaries lack term dates); and
  * the CIM facility metadata file already included in this repository.
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "redesign_raw"
OUT = ROOT / "data" / "redesign_results"

# 117th Congress retained the pre-2020-redistricting (116th) map.  The 118th
# map took effect in 2023.  The analysis window matches CIM's actual-investment
# coverage and avoids the 2013--17 announcement records until their maps are
# also staged.
MAPS = (
    ("cd115", "2018Q1", "2018Q4", RAW / "boundaries" / "tl_2016_us_cd115.zip", "CD115FP"),
    ("cd116", "2019Q1", "2022Q4", RAW / "boundaries" / "tl_2018_us_cd116.zip", "CD116FP"),
    ("cd118", "2023Q1", "2024Q4", RAW / "boundaries" / "cb_2022_us_cd118_500k.zip", "CD118FP"),
)

ELECTION_TO_MAP = {2018: "cd116", 2020: "cd116", 2022: "cd118"}
ELECTION_TO_POST_YEARS = {2018: (2019, 2020), 2020: (2021, 2022), 2022: (2023, 2024)}

FIPS_TO_STATE = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO",
    "09": "CT", "10": "DE", "11": "DC", "12": "FL", "13": "GA", "15": "HI",
    "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY",
    "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN",
    "28": "MS", "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH",
    "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD",
    "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA",
    "54": "WV", "55": "WI", "56": "WY",
}


def read_terms() -> pd.DataFrame:
    """Return dated House terms, including special-election replacements."""
    records = []
    for name in ("legislators-historical.json", "legislators-current.json"):
        with open(RAW / "legislators" / name, encoding="utf-8") as handle:
            records.extend(json.load(handle))

    terms = []
    for person in records:
        for term in person.get("terms", []):
            if term.get("type") != "rep" or "district" not in term:
                continue
            if not term.get("start") or not term.get("state"):
                continue
            terms.append(
                {
                    "state": term["state"],
                    "district": int(term["district"]),
                    "term_start": pd.Timestamp(term["start"]),
                    "term_end": pd.Timestamp(term.get("end", "2100-01-01")),
                    "party": term.get("party", "Unknown"),
                    "bioguide_id": person.get("id", {}).get("bioguide"),
                }
            )
    return pd.DataFrame(terms).drop_duplicates()


def read_map(map_id: str, path: Path, district_col: str, states: set[str]) -> gpd.GeoDataFrame:
    districts = gpd.read_file(path)[["STATEFP", district_col, "geometry"]].copy()
    districts["state"] = districts["STATEFP"].map(FIPS_TO_STATE)
    # TIGER includes pseudo-records such as ``ZZ`` for non-district areas.
    districts["district"] = pd.to_numeric(districts[district_col], errors="coerce")
    districts = districts.loc[districts["district"].notna()].copy()
    districts["district"] = districts["district"].astype(int)
    districts["map_id"] = map_id
    return districts.loc[districts["state"].isin(states), ["map_id", "state", "district", "geometry"]]


def attach_historical_geography(facilities: pd.DataFrame, states: set[str]) -> pd.DataFrame:
    pieces = []
    for map_id, first_q, last_q, path, district_col in MAPS:
        start = pd.Period(first_q, freq="Q").start_time
        end = pd.Period(last_q, freq="Q").end_time
        sub = facilities.loc[facilities["Announcement_Date"].between(start, end)].copy()
        if sub.empty:
            continue
        district_map = read_map(map_id, path, district_col, states)
        points = gpd.GeoDataFrame(
            sub,
            geometry=gpd.points_from_xy(sub["Longitude"], sub["Latitude"]),
            crs="EPSG:4326",
        )
        # TIGER's geographic CRS is NAD83 (EPSG:4269).  For point-in-polygon
        # assignment at congressional-district scale, treat the WGS84 source
        # coordinates as the same geographic datum.  This avoids a GeoPandas /
        # PROJ transform bug in the fresh environment that turns the polygon
        # coordinates into infinities; it is not a substantive reprojection.
        points = points.set_crs(district_map.crs, allow_override=True)
        joined = gpd.sjoin(points, district_map, how="left", predicate="within")
        pieces.append(pd.DataFrame(joined.drop(columns="geometry")))
    return pd.concat(pieces, ignore_index=True)


def attach_party(facilities: pd.DataFrame, terms: pd.DataFrame) -> pd.DataFrame:
    merged = facilities.merge(terms, on=["state", "district"], how="left", validate="m:m")
    matched = merged.loc[
        merged["Announcement_Date"].between(merged["term_start"], merged["term_end"])
    ].copy()
    # A point must have exactly one representative on its announcement date.
    duplicate_ids = matched.loc[matched.duplicated("unique_id", keep=False), "unique_id"].unique()
    if len(duplicate_ids):
        raise ValueError(f"{len(duplicate_ids)} facilities matched more than one House term")
    matched["republican"] = (matched["party"] == "Republican").astype(int)
    return matched


def make_panel(facilities: pd.DataFrame, terms: pd.DataFrame, states: set[str]) -> pd.DataFrame:
    grid = []
    for map_id, first_q, last_q, path, district_col in MAPS:
        geography = pd.DataFrame(read_map(map_id, path, district_col, states).drop(columns="geometry"))
        for quarter in pd.period_range(first_q, last_q, freq="Q"):
            part = geography.copy()
            part["quarter"] = str(quarter)
            part["quarter_end"] = quarter.end_time.normalize()
            grid.append(part)
    panel = pd.concat(grid, ignore_index=True)

    term_panel = panel.merge(terms, on=["state", "district"], how="left", validate="m:m")
    term_panel = term_panel.loc[
        term_panel["quarter_end"].between(term_panel["term_start"], term_panel["term_end"])
    ].copy()
    if term_panel.duplicated(["map_id", "state", "district", "quarter"]).any():
        raise ValueError("A district-quarter matched more than one House term")
    term_panel["republican"] = (term_panel["party"] == "Republican").astype(int)

    announced = facilities.assign(quarter=facilities["Announcement_Date"].dt.to_period("Q").astype(str))
    announced = (
        announced.groupby(["map_id", "state", "district", "quarter"], as_index=False)
        .agg(
            announced_capex_musd=("Estimated_Total_Facility_CAPEX", "sum"),
            announced_projects=("unique_id", "nunique"),
        )
    )
    panel = term_panel.merge(announced, on=["map_id", "state", "district", "quarter"], how="left")
    panel[["announced_capex_musd", "announced_projects"]] = panel[
        ["announced_capex_musd", "announced_projects"]
    ].fillna(0)
    panel["log1p_announced_capex"] = np.log1p(panel["announced_capex_musd"])
    panel["entity"] = panel["map_id"] + "_" + panel["state"] + "_" + panel["district"].astype(str).str.zfill(2)
    return panel.sort_values(["entity", "quarter"]).reset_index(drop=True)


def residualize_two_way(values: pd.Series, panel: pd.DataFrame) -> pd.Series:
    """Project a variable off entity and quarter effects by alternating means."""
    residual = values.astype(float).copy()
    for _ in range(10_000):
        previous = residual.copy()
        residual -= residual.groupby(panel["entity"]).transform("mean")
        residual -= residual.groupby(panel["quarter"]).transform("mean")
        if np.max(np.abs(residual - previous)) < 1e-12:
            return residual
    raise RuntimeError("Two-way fixed-effect residualization did not converge")


def fit_fixed_effects(panel: pd.DataFrame):
    """Entity- and quarter-FE association, clustered by map-specific district.

    Estimation uses the Frisch--Waugh--Lovell equivalent of an OLS regression
    with entity and quarter dummies.  This is numerically more stable than
    forming a 1,300-column dummy matrix before clustered inference.
    """
    y = residualize_two_way(panel["log1p_announced_capex"], panel)
    t = residualize_two_way(panel["republican"], panel)
    return sm.OLS(y, t).fit(cov_type="cluster", cov_kwds={"groups": panel["entity"]})


def read_house_elections() -> pd.DataFrame:
    """Build two-party House-election margins from the local MIT election file."""
    path = ROOT / "data" / "MIT_election_lab" / "1976-2024-house.tab"
    election = pd.read_csv(path, low_memory=False)
    election = election.loc[
        election["year"].isin(ELECTION_TO_MAP)
        & election["stage"].eq("GEN")
        & election["special"].eq(False)
        & election["writein"].eq(False)
    ].copy()
    election["candidatevotes"] = pd.to_numeric(election["candidatevotes"], errors="coerce")
    election = election.dropna(subset=["candidatevotes", "candidate"])

    # Candidates may appear on several ballot lines. Sum those lines and use
    # the largest line to identify their major-party affiliation, per the MIT
    # codebook's fusion-ticket guidance.
    candidate_line = (
        election.groupby(["year", "state_po", "district", "candidate", "party"], as_index=False)
        .agg(candidatevotes=("candidatevotes", "sum"))
    )
    primary_line = (
        candidate_line.sort_values("candidatevotes", ascending=False)
        .drop_duplicates(["year", "state_po", "district", "candidate"])
        .rename(columns={"party": "primary_party"})
    )
    candidate_total = (
        candidate_line.groupby(["year", "state_po", "district", "candidate"], as_index=False)
        .agg(candidatevotes=("candidatevotes", "sum"))
        .merge(
            primary_line[["year", "state_po", "district", "candidate", "primary_party"]],
            on=["year", "state_po", "district", "candidate"],
            validate="1:1",
        )
    )
    candidate_total["major_party"] = np.select(
        [candidate_total["primary_party"].str.startswith("DEMOCRAT"),
         candidate_total["primary_party"].eq("REPUBLICAN")],
        ["Democrat", "Republican"],
        default=pd.NA,
    )
    races = (
        candidate_total.dropna(subset=["major_party"])
        .groupby(["year", "state_po", "district", "major_party"], as_index=False)
        .agg(votes=("candidatevotes", "sum"))
        .pivot(index=["year", "state_po", "district"], columns="major_party", values="votes")
        .reset_index()
        .dropna(subset=["Democrat", "Republican"])
    )
    races["two_party_votes"] = races["Democrat"] + races["Republican"]
    races["running"] = (races["Republican"] - races["Democrat"]) / races["two_party_votes"]
    races["republican_winner"] = (races["running"] > 0).astype(int)
    races["district"] = races["district"].astype(int)
    return races.rename(columns={"state_po": "state", "year": "election_year"})


def make_rd_sample(panel: pd.DataFrame, elections: pd.DataFrame) -> pd.DataFrame:
    """Aggregate district CAPEX during the two years following each election."""
    pieces = []
    for election_year, (first_year, last_year) in ELECTION_TO_POST_YEARS.items():
        sub = panel.loc[
            panel["map_id"].eq(ELECTION_TO_MAP[election_year])
            & panel["quarter"].str[:4].astype(int).between(first_year, last_year)
        ]
        outcome = (
            sub.groupby(["map_id", "state", "district"], as_index=False)
            .agg(
                post_capex_musd=("announced_capex_musd", "sum"),
                post_projects=("announced_projects", "sum"),
            )
        )
        outcome["election_year"] = election_year
        pieces.append(outcome)
    sample = pd.concat(pieces, ignore_index=True).merge(
        elections,
        on=["election_year", "state", "district"],
        how="inner",
        validate="1:1",
    )
    sample["log1p_post_capex"] = np.log1p(sample["post_capex_musd"])
    # Exclude the few cases where the general-election binary party does not
    # match the party in the dated House-term record.  These include party
    # switches and Alaska's 2022 ranked-choice contest, which is not a valid
    # two-party vote-margin RD.  Keeping them would miscode treatment.
    party_at_term_start = []
    for election_year, (first_year, _) in ELECTION_TO_POST_YEARS.items():
        first_quarter = f"{first_year}Q1"
        part = panel.loc[
            panel["map_id"].eq(ELECTION_TO_MAP[election_year])
            & panel["quarter"].eq(first_quarter),
            ["map_id", "state", "district", "party", "republican"],
        ].copy()
        part["election_year"] = election_year
        party_at_term_start.append(part)
    term_party = pd.concat(party_at_term_start, ignore_index=True).rename(
        columns={"party": "term_party", "republican": "term_republican"}
    )
    sample = sample.merge(
        term_party,
        on=["map_id", "state", "district", "election_year"],
        how="left",
        validate="1:1",
    )
    sample["treatment_verified"] = (
        sample["term_party"].isin(["Democrat", "Republican"])
        & sample["republican_winner"].eq(sample["term_republican"])
    )
    return sample


def run_rd(
    sample: pd.DataFrame,
    outcome: str,
    bandwidths=(0.05, 0.10, 0.15),
) -> pd.DataFrame:
    """Local-linear RD screen with separate slopes and election-year effects.

    A final RD should add bias-corrected inference and continuity diagnostics.
    These fixed bandwidths are reported together to avoid selecting on results.
    """
    results = []
    for bandwidth in bandwidths:
        local = sample.loc[sample["running"].abs() <= bandwidth].dropna(subset=[outcome]).copy()
        if local["republican_winner"].nunique() < 2:
            continue
        design = pd.DataFrame(
            {
                "const": 1.0,
                "republican_winner": local["republican_winner"],
                "running": local["running"],
                "winner_x_running": local["republican_winner"] * local["running"],
            },
            index=local.index,
        )
        year_dummies = pd.get_dummies(local["election_year"], prefix="year", drop_first=True, dtype=float)
        design = pd.concat([design, year_dummies.set_index(design.index)], axis=1)
        fit = sm.OLS(local[outcome], design).fit(cov_type="HC1")
        results.append(
            {
                "bandwidth": bandwidth,
                "outcome": outcome,
                "n_district_elections": len(local),
                "republican_wins": int(local["republican_winner"].sum()),
                "democratic_wins": int((1 - local["republican_winner"]).sum()),
                "rd_estimate_log_points": fit.params["republican_winner"],
                "robust_se": fit.bse["republican_winner"],
                "p_value": fit.pvalues["republican_winner"],
                "ci_low": fit.conf_int().loc["republican_winner", 0],
                "ci_high": fit.conf_int().loc["republican_winner", 1],
            }
        )
    return pd.DataFrame(results)


def run_rdrobust(sample: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """Bias-corrected local-polynomial RD with election-cycle covariates.

    This is the primary close-election estimate.  ``rdrobust`` chooses its
    mean-squared-error-optimal bandwidth, applies a triangular kernel, and
    reports robust bias-corrected inference.  Election-cycle indicators absorb
    national variation in observed CIM announcements across the three terms.
    """
    # Import here so the panel-building functions remain usable without the
    # optional RD package.  In the fresh conda environment, rdrobust currently
    # needs NUMBA_DISABLE_JIT=1 because of an upstream Numba cache issue.
    from rdrobust import rdrobust

    local = sample.dropna(subset=[outcome, "running"]).copy()
    cycle_controls = pd.get_dummies(
        local["election_year"].astype(str), prefix="cycle", drop_first=True, dtype=float
    )
    estimate = rdrobust(
        y=local[outcome],
        x=local["running"],
        c=0,
        p=1,
        kernel="triangular",
        masspoints="adjust",
        covs=cycle_controls,
    )
    robust = "Robust"
    return pd.DataFrame(
        [{
            "outcome": outcome,
            "n_district_elections": len(local),
            "effective_n_left": int(estimate.N_h[0]),
            "effective_n_right": int(estimate.N_h[1]),
            "bandwidth_left": float(estimate.bws.loc["h", "left"]),
            "bandwidth_right": float(estimate.bws.loc["h", "right"]),
            "bias_corrected_estimate_log_points": float(estimate.coef.loc[robust, "Coeff"]),
            "robust_bias_corrected_se": float(estimate.se.loc[robust, "Std. Err."]),
            "robust_bias_corrected_p_value": float(estimate.pv.loc[robust, "P>|z|"]),
            "robust_ci_low": float(estimate.ci.loc[robust, "CI Lower"]),
            "robust_ci_high": float(estimate.ci.loc[robust, "CI Upper"]),
        }]
    )


def run_density_test(sample: pd.DataFrame) -> pd.DataFrame:
    """McCrary-style density-continuity diagnostic for the vote margin."""
    from rddensity import rddensity

    density = rddensity(X=sample["running"].dropna(), c=0)
    return pd.DataFrame(
        [{
            "n_district_elections": int(sample["running"].notna().sum()),
            "jackknife_t_statistic": float(density.test["t_jk"]),
            "jackknife_p_value": float(density.test["p_jk"]),
            "bandwidth_left": float(density.h["left"]),
            "bandwidth_right": float(density.h["right"]),
        }]
    )


def read_pre_election_covariates() -> pd.DataFrame:
    """Read ACS characteristics measured in the election year and map vintage."""
    acs = pd.read_csv(ROOT / "data" / "posteda" / "acs_data.csv")
    acs["state_fips"] = acs["geography_str"].str.extract(r"state:(\d+)")[0].str.zfill(2)
    acs["district"] = pd.to_numeric(
        acs["geography_str"].str.extract(r"congressional district:(\d+)")[0],
        errors="coerce",
    )
    acs["state"] = acs["state_fips"].map(FIPS_TO_STATE)
    covariates = ["med_household_income", "unemployed", "total_pop", "gini"]
    acs = acs.loc[acs["year"].isin(ELECTION_TO_MAP), ["year", "state", "district", *covariates]].copy()
    acs = acs.rename(columns={"year": "election_year"})
    acs["district"] = acs["district"].astype(int)
    return acs


def local_randomization_sign_balance(sample: pd.DataFrame, bandwidths=(0.05, 0.10, 0.15)) -> pd.DataFrame:
    """Binomial side-of-cutoff check; not a substitute for a full density test."""
    rows = []
    for bandwidth in bandwidths:
        local = sample.loc[sample["running"].abs() <= bandwidth]
        wins = int(local["republican_winner"].sum())
        rows.append(
            {
                "bandwidth": bandwidth,
                "n": len(local),
                "republican_wins": wins,
                "democratic_wins": len(local) - wins,
                "two_sided_binomial_p": binomtest(wins, n=len(local), p=0.5).pvalue,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(ROOT / "data" / "manufacturing_energy_and_industry_facility_metadata.csv", header=3)
    raw["Announcement_Date"] = pd.to_datetime(raw["Announcement_Date"], errors="coerce")
    raw["Estimated_Total_Facility_CAPEX"] = pd.to_numeric(raw["Estimated_Total_Facility_CAPEX"], errors="coerce")

    # Use only points explicitly marked as known locations and observations with
    # a dated, nonmissing announced CAPEX.  This prevents assigning approximate
    # locations to potentially wrong historical districts.
    facilities = raw.loc[
        raw["LatLon_Valid"].eq(True)
        & raw["Announcement_Date"].between("2018-01-01", "2024-12-31")
        & raw["Estimated_Total_Facility_CAPEX"].notna()
        & raw["Latitude"].notna()
        & raw["Longitude"].notna()
    ].copy()
    eligible_facilities = len(facilities)
    states = set(facilities["State"].dropna())
    facilities = attach_historical_geography(facilities, states)
    facilities = facilities.rename(columns={"State": "facility_state"})
    spatially_matched = facilities["state"].notna().sum()
    facilities = facilities.loc[facilities["state"].eq(facilities["facility_state"])].copy()
    state_consistent = len(facilities)
    terms = read_terms()
    facilities = attach_party(facilities, terms)
    panel = make_panel(facilities, terms, states)

    facilities.to_csv(OUT / "facilities_historical_party.csv", index=False)
    panel.to_csv(OUT / "district_quarter_panel.csv", index=False)

    result = fit_fixed_effects(panel)
    elections = read_house_elections()
    rd_sample = make_rd_sample(panel, elections)
    two_party_elections = len(rd_sample)
    rd_sample = rd_sample.loc[rd_sample["treatment_verified"]].copy()
    rd_results = run_rd(rd_sample, "log1p_post_capex")
    rdrobust_results = run_rdrobust(rd_sample, "log1p_post_capex")
    density_results = run_density_test(rd_sample)
    covariates = read_pre_election_covariates()
    rd_with_covariates = rd_sample.merge(
        covariates, on=["election_year", "state", "district"], how="left", validate="1:1"
    )
    covariate_results = pd.concat(
        [run_rd(rd_with_covariates, covariate) for covariate in [
            "med_household_income", "unemployed", "total_pop", "gini"
        ]],
        ignore_index=True,
    )
    sign_balance = local_randomization_sign_balance(rd_sample)
    rd_sample.to_csv(OUT / "rd_district_election_sample.csv", index=False)
    rd_results.to_csv(OUT / "rd_results.csv", index=False)
    rdrobust_results.to_csv(OUT / "rdrobust_results.csv", index=False)
    density_results.to_csv(OUT / "rd_density_test.csv", index=False)
    covariate_results.to_csv(OUT / "rd_covariate_balance.csv", index=False)
    sign_balance.to_csv(OUT / "rd_sign_balance.csv", index=False)
    switches = (
        panel.groupby("entity")["republican"].nunique().gt(1).sum()
    )
    diagnostics = pd.DataFrame(
        {
            "metric": [
                "historically_matched_facilities",
                "eligible_facilities_before_spatial_match",
                "facilities_inside_historical_district_polygon",
                "facilities_with_state_consistent_historical_polygon",
                "district_quarters",
                "map_specific_districts",
                "entities_with_party_switch",
                "republican_coefficient",
                "republican_clustered_se",
                "republican_p_value",
                "rd_two_party_district_elections",
                "rd_treatment_verified_district_elections",
                "rd_sample_with_pre_election_acs_covariates",
            ],
            "value": [
                len(facilities),
                eligible_facilities,
                spatially_matched,
                state_consistent,
                len(panel),
                panel["entity"].nunique(),
                switches,
                result.params["republican"],
                result.bse["republican"],
                result.pvalues["republican"],
                two_party_elections,
                len(rd_sample),
                len(rd_with_covariates.dropna(subset=["med_household_income", "unemployed", "total_pop", "gini"])),
            ],
        }
    )
    diagnostics.to_csv(OUT / "diagnostics.csv", index=False)
    (OUT / "fixed_effects_summary.txt").write_text(result.summary().as_text())
    print(diagnostics.to_string(index=False))
    print("\nRD feasibility and local-linear estimates:")
    print(rd_results.to_string(index=False))
    print("\nBias-corrected RD estimate (primary):")
    print(rdrobust_results.to_string(index=False))
    print("\nRunning-variable density-continuity test:")
    print(density_results.to_string(index=False))
    print("\nLocal-randomization sign-balance check:")
    print(sign_balance.to_string(index=False))
    print("\nRD covariate-continuity checks:")
    print(covariate_results.to_string(index=False))
    print(f"\nWrote outputs to {OUT}")


if __name__ == "__main__":
    main()
