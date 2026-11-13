"""Build coordinate, party, and district-election assignment ledgers."""
import json
import sys
from functools import lru_cache
from pathlib import Path
import geopandas as gpd
import pyproj
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
DATA=ROOT/"data"
sys.path.insert(0,str(ROOT/"scripts"))
from historical_party_panel import FIPS_TO_STATE, MAPS, read_map
from elections import roster
from exposure import classify, SOURCE

YEARS={2018:116,2020:117,2022:118}
KEY=["year","state","district"]


def geographies():
    pyproj.network.set_network_enabled(False)
    states=set(FIPS_TO_STATE.values())-{"DC"}
    maps={int(key[-3:]):read_map(key,p,col,states).to_crs(4269)
          for key,_,__,p,col in MAPS}
    nc=gpd.read_file(HERE/"nc_117.zip").to_crs(4269)
    nc=nc.assign(state="NC",district=nc.DISTRICT.astype(int),map_id="cd117")
    maps[117]=pd.concat([maps[116].query('state!="NC"'),nc[maps[116].columns]])
    for n,g in maps.items():
        maps[n]=g[g.district.lt(90)].reset_index(drop=True)
        assert len(maps[n])==435 and np.isfinite(maps[n].total_bounds).all()
    return maps


def spatial(f,maps):
    valid=f[f.LatLon_Valid.eq(True) & f.Latitude.notna() & f.Longitude.notna()]
    pts=gpd.GeoDataFrame(valid[["unique_id","State"]],geometry=gpd.points_from_xy(
        valid.Longitude,valid.Latitude),crs=4326).to_crs(4269)
    assert np.isfinite(pts.geometry.x).all() and np.isfinite(pts.geometry.y).all()
    rows=[]
    for congress,g in maps.items():
        j=gpd.sjoin(pts,g,how="left",predicate="intersects")
        ambiguous=j.unique_id.duplicated(keep=False)
        j["location_reason"]=np.select([ambiguous,j.state.isna(),j.State.ne(j.state)],
            ["multiple polygons","outside polygons","state mismatch"],default="matched")
        j=j.drop_duplicates("unique_id").copy()
        j["boundary_distance_deg"]=[p.distance(g.geometry.iloc[int(i)].boundary)
            if pd.notna(i) else np.nan for p,i in zip(j.geometry,j.index_right)]
        j.loc[j.location_reason!="matched","district"]=np.nan
        rows.append(j.assign(congress=congress)[["unique_id","congress","district","location_reason","boundary_distance_deg"]])
    return pd.concat(rows,ignore_index=True)


def terms():
    records=[]
    for filename in sorted((DATA/"redesign_raw/legislators").glob("*.json")):
        for person in json.loads(filename.read_text()):
            for t in person["terms"]:
                if t["type"]!="rep" or t["end"]<"2017" or t["start"]>"2025": continue
                affiliations=t.get("party_affiliations") or [t]
                for a in affiliations:
                    end=min(pd.Timestamp(t["end"]),pd.Timestamp(a["end"])+pd.Timedelta(days=1))
                    records.append((t["state"],t["district"],pd.Timestamp(a["start"]),end,a["party"]))
    frame=pd.DataFrame(records,columns=["state","district","start","end","party"]).drop_duplicates()
    return dict(tuple(frame.groupby(["state","district"])))


def lookup(t,state,district,date):
    group=t.get((state,district))
    if group is None: return None
    matches=group.loc[(group.start<=date)&(date<group.end),"party"].unique()
    assert len(matches)<=1,(state,district,date,matches)
    return matches[0] if len(matches) else None


def build(out):
    raw=pd.read_csv(DATA/"manufacturing_energy_and_industry_facility_metadata.csv",skiprows=5,low_memory=False)
    f=raw[raw.Segment.eq("Manufacturing")].copy()
    assert len(raw)==8532 and len(f)==581 and f.unique_id.is_unique
    f["date"]=pd.to_datetime(f.Announcement_Date)
    f=f.rename(columns={"Estimated_Total_Facility_CAPEX":"capex","US Representative Party":"legacy_party"})
    cross=f[["Technology","Subcategory"]].drop_duplicates().sort_values(["Technology","Subcategory"]).copy()
    cross[["exposure","rationale"]]=[classify(t,p) for t,p in cross.itertuples(index=False,name=None)]
    cross.assign(source=SOURCE).to_csv(out/"crosswalk_45x.csv",index=False)
    f=f.merge(cross,on=["Technology","Subcategory"],validate="m:1").rename(columns={"State":"state"})
    maps=geographies()
    loc=spatial(f.rename(columns={"state":"State"}),maps)
    t=terms()
    f["congress"]=np.select([f.date<"2018-01-01",f.date<"2019-01-03",f.date<"2021-01-03",
                              f.date<"2023-01-03",f.date<"2025-01-01"],[0,115,116,117,118],default=0)
    f=f.merge(loc,on=["unique_id","congress"],how="left",validate="1:1")
    f["location_reason"]=f.location_reason.fillna(f.congress.map(lambda c:"outside dates" if not c else "invalid coordinates"))
    f["historical_party"]=[lookup(t,r.state,r.district,r.date) for r in f.itertuples()]
    f["quarter"]=f.date.dt.to_period("Q").astype(str)
    f["near_boundary"]=f.boundary_distance_deg.lt(.001) & f.district.notna()
    f["year"]=f.congress.map({v:k for k,v in YEARS.items()})
    e=roster(DATA)
    assert e[KEY].duplicated().sum()==0 and len(e[e.state!="DC"])==1305
    assert not e.query('reason==""').audit.isin(["review",""]).any()
    e["winner"]=np.where(e.running.isna(),"",np.where(e.running>0,"R","D"))
    f=f.merge(e[KEY+["winner","reason"]],on=KEY,how="left",validate="m:1")
    f["post_status"]=f.location_reason.where(f.location_reason!="matched",
        np.where(f.year.isna(),"outside post window",f.reason.fillna("district missing from roster").replace("","included")))
    # Jan 1-2 between Congresses are outside both prespecified post windows.
    gap=(f.date.dt.year.isin([2019,2021,2023])) & f.date.dt.month.eq(1) & f.date.dt.day.le(2)
    f.loc[gap & f.location_reason.eq("matched"),"post_status"]="transition gap"
    f.to_csv(out/"facility_assignments.csv",index=False)
    f.loc[f.post_status.ne("included"),["unique_id","state","date","congress","district","post_status"]].to_csv(out/"facility_exclusions.csv",index=False)
    loc.to_csv(out/"coordinate_audit.csv",index=False)
    e.to_csv(out/"election_roster.csv",index=False)
    d=e[e.reason.eq("")].copy()
    post=f[f.post_status.eq("included")].copy()
    assert post.unique_id.is_unique
    post["clear"]=post.exposure.eq("clear")
    post["broad"]=post.exposure.ne("outside")
    largest=post.capex.idxmax()
    post["minus_largest"]=post.capex.mask(post.index==largest,0)
    post.loc[[largest],["unique_id","state","district","year","capex"]].to_csv(out/"largest_project.csv",index=False)
    agg=post.groupby(KEY).agg(projects=("unique_id","size"),known=("capex","count"),dollars=("capex","sum"),
        clear=("clear","sum"),broad=("broad","sum"),minus_largest=("minus_largest","sum"))
    d=d.merge(agg.reset_index(),how="left",on=KEY,validate="1:1")
    d[agg.columns]=d[agg.columns].fillna(0)
    assert d.projects.sum()==len(post) and np.isclose(d.dollars.sum(),post.capex.sum())
    d["unknown"]=d.projects-d.known
    for col in ["projects","known","unknown","clear","broad"]: d["any_"+col]=d[col].gt(0).astype(int)
    d["no_project"]=1-d.any_projects
    d["only_unknown"]=d.any_projects*(1-d.any_known)
    d["disclosure"]=d.known.div(d.projects.replace(0,np.nan))
    d["log_capex"]=np.log1p(d.dollars)
    d["log_minus_largest"]=np.log1p(d.minus_largest)
    pre=[]
    for year,congress in YEARS.items():
        sub=f.loc[(f.date.dt.year==year)&(f.date.dt.month<=9),["unique_id","state","capex"]]
        sub=sub.merge(loc.query("congress==@congress"),how="left",on="unique_id",validate="1:1")
        pre.append(sub.assign(year=year))
    pre=pd.concat(pre)
    pre.to_csv(out/"placebo_assignments.csv",index=False)
    sums=pre.groupby(KEY).agg(pre_dollars=("capex","sum"),pre_count=("unique_id","size"))
    d=d.merge(sums,on=KEY,how="left",validate="1:1")
    d[["pre_dollars","pre_count"]]=d[["pre_dollars","pre_count"]].fillna(0)
    d["pre_log_capex"]=np.log1p(d.pre_dollars)
    # Predetermined ACS years: verify previous vintage's footprint against election district.
    # Require 99% polygon Jaccard overlap, including coast-line generalization of the 118th.
    acs=pd.read_csv(DATA/"posteda/acs_data.csv")
    acs["state"]=acs.geography_str.str.extract(r"state:(\d+)")[0].str.zfill(2).map(FIPS_TO_STATE)
    acs["district"]=pd.to_numeric(acs.geography_str.str.extract(r"district:(\d+)")[0])
    acs["vintage"]=pd.to_numeric(acs.geography.str.extract(r"\((\d+)(?:th|st|nd|rd) Congress")[0])
    acs["year"]+=1
    covs=["med_household_income","unemployed","total_pop","gini"]
    d=d.merge(acs[acs.year.isin(YEARS)][KEY+covs+["vintage"]],on=KEY,how="left",validate="1:1")
    shapes={n:g.set_index(["state","district"]).geometry.simplify(.0003).make_valid() for n,g in maps.items()}
    @lru_cache(maxsize=None)
    def overlap(a,b,k):
        if a==b: return 1.
        if a not in shapes or k not in shapes[a].index or k not in shapes[b].index: return 0.
        x,y=shapes[a][k],shapes[b][k]
        return x.intersection(y).area/x.union(y).area
    inc=[]; scores=[]
    for r in d.itertuples():
        k=(r.state,r.district); n=YEARS[r.year]
        scores.append(overlap(r.vintage,n,k))
        old=lookup(t,*k,pd.Timestamp(r.year,11,1)) if overlap(n-1,n,k)>=.99 else None
        inc.append({"Democrat":0,"Republican":1}.get(old,np.nan))
    d["acs_overlap"]=scores
    d.loc[d.acs_overlap<.99,covs]=np.nan
    d["incumbent_R"]=inc
    d.to_csv(out/"district_elections.csv",index=False)
    annual=f[f.congress>0]
    flow={"raw_records":len(raw),"all_manufacturing":len(f),"dated_2018_24":len(annual),
          "valid_coordinates":annual.LatLon_Valid.sum(),"valid_coordinates_and_value":annual[annual.LatLon_Valid].capex.count(),
          "historical_district":annual.district.count(),"near_boundary":annual.near_boundary.sum(),
          "historical_party":annual.historical_party.count(),"post_election_projects":len(post),
          "post_with_value":post.capex.count(),"district_elections":len(d)}
    assert list(flow.values())[:5]==[8532,581,577,534,491]
    pd.Series(flow,name="records").rename_axis("step").to_csv(out/"sample_flow.csv")
    groups=[["Technology"],["year"],["winner"],["year","winner","Technology"]]
    disclosure=[]
    for keys in groups:
        a=post.groupby(keys).capex.agg(["size","count"]).reset_index()
        disclosure.append(a.assign(group="_".join(keys),rate=a["count"]/a["size"]))
    pd.concat(disclosure,ignore_index=True).to_csv(out/"disclosure.csv",index=False)
    return f,d
