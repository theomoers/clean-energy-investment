"""Feasibility only. National House turnover is not randomized treatment.

Rebuild from frozen local inputs. Every district enters before project joins.
Uses equal-area bidirectional geographic overlap, dated member affiliations,
state-clustered descriptive DiD diagnostics, and complete zero-project cells.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import geopandas as gp
import numpy as np
import pandas as pd
import pyproj
import shapely
import statsmodels.api as sm
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'revision_analysis'))
from panel import geographies


def member_terms():
    records = []
    for p in sorted((ROOT/'data/redesign_raw/legislators').glob('*.json')):
        for person in json.loads(p.read_text()):
            for t in person['terms']:
                if t['type'] != 'rep' or t['end'] < '2015' or t['start'] > '2025':
                    continue
                for a in t.get('party_affiliations') or [t]:
                    # Affiliation end dates are inclusive, term end dates exclusive.
                    end = min(pd.Timestamp(t['end']), pd.Timestamp(a['end'])+pd.Timedelta(days=1))
                    records.append([t['state'], int(t['district']), pd.Timestamp(a['start']), end,
                                    a['party'], person['id']['bioguide'], person['name'].get('official_full',
                                    person['name']['first']+' '+person['name']['last'])])
    d = pd.DataFrame(records, columns=['state','district','start','end','party','member','name']).drop_duplicates()
    from historical_party_panel import FIPS_TO_STATE
    return d[d.state.isin(set(FIPS_TO_STATE.values())-{'DC'}) & d.district.ge(0)]


def member_at(terms, state, district, date):
    a = terms[(terms.state == state) & (terms.district == district) &
              (terms.start <= date) & (date < terms.end)]
    assert len(a) <= 1, (state,district,date,a)
    return ('','', '') if a.empty else tuple(a.iloc[0][['party','member','name']])


def crosswalk(maps, out):
    projected = {n:g.to_crs(6933).assign(geometry=lambda x:x.geometry.make_valid()).reset_index(drop=True)
                 for n,g in maps.items()}
    rows = []
    for new, old in [(116,115),(117,116),(118,117),(118,116)]:
        a,b = projected[new], projected[old]
        l,r = b.sindex.query(a.geometry, predicate='intersects')
        keep = a.state.values[l] == b.state.values[r]
        l,r = l[keep],r[keep]
        area = shapely.area(shapely.intersection(a.geometry.values[l],b.geometry.values[r]))
        matches = pd.DataFrame({'i':l,'j':r,'intersection':area})
        best = matches.sort_values('intersection',ascending=False).drop_duplicates('i').sort_values('i')
        for z in best.itertuples():
            x,y = a.iloc[z.i], b.iloc[z.j]
            rows.append([new, old, x.state, x.district, y.district,
                         z.intersection/x.geometry.area,z.intersection/y.geometry.area])
    d = pd.DataFrame(rows,columns=['new_congress','old_congress','state','new_district','old_district',
                                  'share_new','share_old'])
    d['continuity'] = d[['share_new','share_old']].min(axis=1)
    assert d.groupby(['new_congress','old_congress']).size().eq(435).all()
    d.to_csv(out/'geographic_crosswalk.csv',index=False)
    return d


def transition_switches(terms, cw, out):
    records = []
    for y,new,old in [(2019,116,115),(2021,117,116),(2023,118,117)]:
        a = cw.query('new_congress==@new & old_congress==@old')
        for r in a.itertuples():
            prior = member_at(terms,r.state,r.old_district,pd.Timestamp(y-1,12,15))
            after = member_at(terms,r.state,r.new_district,pd.Timestamp(y,1,15))
            numbered = member_at(terms,r.state,r.new_district,pd.Timestamp(y-1,12,15))
            records.append([y,r.state,r.new_district,r.old_district,r.continuity,*prior,*after,
                            numbered[0],numbered[1]])
    d = pd.DataFrame(records,columns=['year','state','new_district','old_district','continuity',
        'prior_party','prior_member','prior_name','after_party','after_member','after_name',
        'same_number_prior_party','same_number_prior_member'])
    major = ['Democrat','Republican']
    d['party_changed'] = d.prior_party.ne(d.after_party) & d.prior_party.isin(major) & d.after_party.isin(major)
    d['same_number_party_changed'] = d.same_number_prior_party.ne(d.after_party) & d.same_number_prior_party.isin(major) & d.after_party.isin(major)
    d['stable_geography'] = d.continuity.ge(.99)
    d.to_csv(out/'seat_transition_ledger.csv',index=False)
    summary=[]
    for y,a in d.groupby('year'):
        v=a[a.stable_geography & a.prior_party.isin(major) & a.after_party.isin(major)]
        summary.append([y,len(a),int(a.same_number_party_changed.sum()),len(v),int(v.party_changed.sum()),
                        int((v.party_changed & v.after_party.eq('Democrat')).sum()),
                        int((v.party_changed & v.after_party.eq('Republican')).sum())])
    pd.DataFrame(summary,columns=['year','seat_numbers','same_number_changes_not_geography_adjusted',
        'comparable_geography_major_party_seats','verified_geography_party_changes','R_to_D','D_to_R']).to_csv(out/'switch_summary.csv',index=False)
    return d


def sample_roster(terms, maps, cw, transition, first, threshold):
    new = 116 if transition == 2019 else 118
    prior_maps = [115] if transition == 2019 else ([116,117] if first==2019 else [117])
    rows=[]
    for r in maps[new].itertuples():
        links={new:r.district}; overlaps=[]
        for old in prior_maps:
            a=cw[(cw.new_congress==new)&(cw.old_congress==old)&(cw.state==r.state)&(cw.new_district==r.district)].iloc[0]
            links[old]=int(a.old_district); overlaps.append(a.continuity)
        previous=member_at(terms,r.state,links[prior_maps[-1]],pd.Timestamp(transition-1,12,15))
        after=member_at(terms,r.state,r.district,pd.Timestamp(transition,1,15))
        reason=''
        if min(overlaps)<threshold: reason='geography changed'
        elif previous[0] not in ['Democrat','Republican'] or after[0] not in ['Democrat','Republican']: reason='vacant or other party at transition'
        elif previous[1]!=after[1] or previous[0]!=after[0]: reason='member or party changed at transition'
        # Continuous-member audit is stricter. Partition actual intervals, not just snapshots.
        continuous=True
        for year in range(first,transition+2):
            congress=115 if year<2019 else 116 if year<2021 else 117 if year<2023 else 118
            district=links[congress]
            start=pd.Timestamp(year,1,3) if year%2 else pd.Timestamp(year,1,1)
            end=pd.Timestamp(year+1,1,1)
            ts=terms[(terms.state==r.state)&(terms.district==district)&(terms.start<end)&(terms.end>start)].sort_values('start')
            cursor=start
            for t in ts.itertuples():
                if t.start>cursor or t.member!=after[1] or t.party!=after[0]:continuous=False
                cursor=max(cursor,t.end)
            if cursor<end:continuous=False
        rows.append([transition,first,threshold,r.state,r.district,after[0],after[1],after[2],min(overlaps),
                     previous[0],previous[1],int(after[0]==('Democrat' if transition==2019 else 'Republican')),
                     reason,continuous,*[links.get(k,np.nan) for k in [115,116,117]]])
    return pd.DataFrame(rows,columns=['transition','first_year','threshold','state','district','party','member','name',
        'min_overlap','prior_party','prior_member','gainer','transition_exclusion','continuous_member',
        'old115','old116','old117'])


def assignments(raw,maps,out):
    valid=raw[raw.LatLon_Valid.eq(True)&raw.Latitude.notna()&raw.Longitude.notna()].copy()
    pts=gp.GeoDataFrame(valid[['unique_id','State']],geometry=gp.points_from_xy(valid.Longitude,valid.Latitude),crs=4326).to_crs(4269)
    rows=[]
    for n in [115,116,117,118]:
        j=gp.sjoin(pts,maps[n],how='left',predicate='intersects')
        duplicate=j.unique_id.duplicated(keep=False)
        j['reason']=np.select([duplicate,j.state.isna(),j.State.ne(j.state)],['multiple polygons','outside polygons','state mismatch'],default='matched')
        j.loc[j.reason.ne('matched'),'district']=np.nan
        j=j.drop_duplicates('unique_id')
        rows.append(pd.DataFrame(j[['unique_id','state','district','reason']]).assign(congress=n))
    d=pd.concat(rows,ignore_index=True)
    d.to_csv(out/'facility_geography.csv',index=False)
    return d


def diagnostic_fit(frame,y,state_fe=True):
    both=frame.groupby('state').gainer.nunique()
    both=set(both[both.eq(2)].index)
    if state_fe:frame=frame[frame.state.isin(both)].copy()
    n=frame.state.nunique()
    unavailable=dict(estimate=np.nan,se=np.nan,lo=np.nan,hi=np.nan,p=np.nan,states=n,
                     districts=len(frame),states_with_both_arms=len(both),estimable=False)
    if n<3 or frame.gainer.nunique()<2:return unavailable
    x=pd.DataFrame({'const':1.,'gainer':frame.gainer.astype(float)},index=frame.index)
    if state_fe:x=pd.concat([x,pd.get_dummies(frame.state,drop_first=True,dtype=float)],axis=1)
    if len(x)<=x.shape[1]+1:return unavailable
    fit=sm.OLS(frame[y].astype(float),x).fit(cov_type='cluster',cov_kwds={'groups':frame.state},use_t=True)
    n=frame.state.nunique(); b=fit.params['gainer'];se=fit.bse['gainer'];crit=stats.t.ppf(.975,n-1)
    return dict(estimate=b,se=se,lo=b-crit*se,hi=b+crit*se,p=fit.pvalues['gainer'],states=n,
                districts=len(frame),states_with_both_arms=len(both),
                estimable=bool(np.linalg.matrix_rank(x)>np.linalg.matrix_rank(x.drop(columns='gainer'))))


def joint_pre(annual,outcome,trans):
    pre=annual[annual.year<trans]
    wide=pre.pivot(index=['state','district','gainer'],columns='year',values=outcome).reset_index()
    both=wide.groupby('state').gainer.nunique();wide=wide[wide.state.isin(both[both.eq(2)].index)].copy()
    years=sorted(pre.year.unique());g=wide.state.nunique()
    x=np.column_stack([np.ones(len(wide)),wide.gainer,pd.get_dummies(wide.state,drop_first=True,dtype=float)])
    if g<3 or len(wide)<=x.shape[1]+1:return dict(joint_p=np.nan,restrictions=0,joint_F=np.nan)
    y=np.column_stack([wide[k]-wide[trans-1] for k in years[:-1]])
    bread=np.linalg.pinv(x.T@x);beta=bread@x.T@y;res=y-x@beta
    influence=(x@bread[:,1])[:,None]*res
    sums=pd.DataFrame(influence,index=wide.index).groupby(wide.state).sum().to_numpy()
    cov=sums.T@sums*g/(g-1)*(len(x)-1)/(len(x)-x.shape[1]);rank=np.linalg.matrix_rank(cov)
    if not rank:return dict(joint_p=np.nan,restrictions=0,joint_F=np.nan)
    f=float(beta[1]@np.linalg.pinv(cov)@beta[1]/rank)
    return dict(joint_p=float(stats.f.sf(f,rank,g-1)),restrictions=rank,joint_F=f)


def analyze(roster,raw,loc,out):
    support=[]; tests=[]; events=[]; panels=[]; district_rows=[]; splits=[];joint=[]
    definitions=[('manufacturing',raw.Segment.eq('Manufacturing')),
                 ('clean_power',raw.Segment.eq('Energy and Industry')&raw.Technology.isin(['Solar','Wind','Storage'])),
                 ('legacy_all',pd.Series(True,index=raw.index))]
    for key,rr in roster.groupby(['transition','first_year','threshold']):
        trans,first,threshold=key; n=116 if trans==2019 else 118
        for sample in ['transition_stable','continuous_member']:
            r=rr[rr.transition_exclusion.eq('') & (rr.continuous_member if sample=='continuous_member' else True)].copy()
            for sector,mask in definitions:
                f=raw[mask & raw.date.dt.year.between(first,trans+1)].merge(loc[loc.congress.eq(n)&loc.reason.eq('matched')],on='unique_id',validate='1:1')
                f=f.merge(r[['state','district','gainer','old115','old116','old117']],on=['state','district'],validate='m:1')
                # Constant common core. A facility must lie in every linked old polygon,
                # even when its announcement predates or follows that map vintage.
                for old in [115,116,117]:
                    col=f'old{old}'
                    if not r[col].notna().any():continue
                    oldloc=loc[loc.congress.eq(old)&loc.reason.eq('matched')][['unique_id','district']].rename(columns={'district':'mapped_old'})
                    f=f.merge(oldloc,on='unique_id',how='left',validate='1:1')
                    f=f[f[col].isna()|f[col].eq(f.mapped_old)].drop(columns='mapped_old')
                assert f.unique_id.is_unique
                # January 1-2 of each new Congress is omitted from the transition year.
                f=f[~(f.date.dt.year.eq(trans)&f.date.dt.month.eq(1)&f.date.dt.day.le(2))].copy()
                f['year']=f.date.dt.year;f['quarter']=f.date.dt.to_period('Q').astype(str)
                f['known']=f.Estimated_Total_Facility_CAPEX.notna()
                years=list(range(first,trans+2)); quarters=pd.period_range(f'{first}Q1',f'{trans+1}Q4',freq='Q')
                grid=r.merge(pd.DataFrame({'quarter':quarters.astype(str),'year':quarters.year,'q_index':np.arange(len(quarters))}),how='cross')
                agg=f.groupby(['state','district','quarter']).agg(projects=('unique_id','size'),known=('known','sum'),dollars=('Estimated_Total_Facility_CAPEX','sum'))
                grid=grid.merge(agg,on=['state','district','quarter'],how='left',validate='1:1')
                grid[['projects','known','dollars']]=grid[['projects','known','dollars']].fillna(0)
                grid['unknown']=grid.projects-grid.known;grid['log_capex']=np.log1p(grid.dollars)
                assert grid.projects.sum()==len(f) and not grid.duplicated(['state','district','quarter']).any()
                tag=dict(transition=trans,first_year=first,threshold=threshold,sample=sample,sector=sector)
                panels.append(grid.assign(sample=sample,sector=sector))
                for arm,a in r.groupby('gainer'):
                    z=f[f.gainer.eq(arm)];pre=z[z.year<trans];post=z[z.year>=trans]
                    prekeys=set(zip(pre.state,pre.district));postkeys=set(zip(post.state,post.district))
                    bydist=post.groupby(['state','district']).size()
                    s=dict(**tag,gainer=arm,districts=len(a),states=a.state.nunique(),pre_projects=len(pre),post_projects=len(post),
                           pre_active=len(prekeys),post_active=len(postkeys),active_both=len(prekeys&postkeys),
                           pre_known=int(pre.known.sum()),post_known=int(post.known.sum()),
                           post_only_undisclosed=sum(v.known.sum()==0 for _,v in post.groupby(['state','district'])),
                           event_equivalent_districts=float(bydist.sum()**2/(bydist**2).sum()) if len(bydist) else 0.,
                           largest_district_share=float(bydist.max()/bydist.sum()) if len(bydist) else 0.)
                    support.append(s)
                annual=grid.groupby(['state','district','gainer','year'],as_index=False)[['projects','known','dollars','unknown']].sum()
                annual['log_capex']=np.log1p(annual.dollars)
                baseline=annual[annual.year.eq(trans-1)].set_index(['state','district'])
                for outcome in ['projects','log_capex','dollars']:
                    joint.append(dict(**tag,outcome=outcome,**joint_pre(annual,outcome,trans)))
                    pre=annual[annual.year<trans].groupby(['state','district'])[outcome].mean()
                    post=annual[annual.year>=trans].groupby(['state','district'])[outcome].mean()
                    delta=r.set_index(['state','district'])[['gainer']].assign(delta=post-pre).reset_index()
                    for state_fe in [False,True]:
                        tests.append(dict(**tag,outcome=outcome,test='post_minus_pre_relative_change',state_fe=state_fe,
                                          **diagnostic_fit(delta,'delta',state_fe)))
                    for year in years:
                        d=annual[annual.year.eq(year)].set_index(['state','district'])[['gainer',outcome]].copy()
                        d['change']=d[outcome]-baseline[outcome]
                        d=d.reset_index()
                        if year==trans-1:res=dict(estimate=0.,se=0.,lo=0.,hi=0.,p=np.nan,states=d.state.nunique(),estimable=True)
                        else:res=diagnostic_fit(d,'change',True)
                        events.append(dict(**tag,outcome=outcome,year=year,event_year=year-trans,**res))
                        if year<trans-1:tests.append(dict(**tag,outcome=outcome,test=f'pre_year_{year}_vs_{trans-1}',state_fe=True,**res))
                    # District pre-period quarterly slope. Difference expressed per year.
                    preq=grid[grid.year<trans]
                    slopes=[]
                    for (st,dist),g in preq.groupby(['state','district']):
                        slopes.append([st,dist,int(g.gainer.iloc[0]),np.polyfit(g.q_index,g[outcome] if outcome!='log_capex' else g.log_capex,1)[0]*4])
                    slopes=pd.DataFrame(slopes,columns=['state','district','gainer','slope'])
                    tests.append(dict(**tag,outcome=outcome,test='pre_quarterly_slope_per_year',state_fe=True,**diagnostic_fit(slopes,'slope',True)))
                # One district is one forest unit. State-disjoint illustrative holdout.
                if threshold in [.99,.8,.5]:
                    states=np.array(sorted(r.state.unique()));rng=np.random.default_rng(48123+trans);rng.shuffle(states)
                    teststates=set(states[:len(states)//2]);r['fold']=np.where(r.state.isin(teststates),'test','train')
                    for fold,g in r.groupby('fold'):
                        for arm,a in g.groupby('gainer'):
                            z=f.merge(a[['state','district']],on=['state','district'],validate='m:1')
                            post=z[z.year>=trans]
                            splits.append(dict(**tag,fold=fold,gainer=arm,districts=len(a),states=a.state.nunique(),
                                post_projects=len(post),post_active=post.groupby(['state','district']).ngroups))
                district_rows.append(annual.assign(**{k:v for k,v in tag.items() if k not in annual.columns}))
    for name,rows in [('support',support),('diagnostics',tests),('event_study',events),('holdout_support',splits),('joint_pretests',joint)]:
        pd.DataFrame(rows).to_csv(out/f'{name}.csv',index=False)
    pd.concat(panels,ignore_index=True).to_csv(out/'district_quarter_panel.csv',index=False)
    pd.concat(district_rows,ignore_index=True).to_csv(out/'district_year_panel.csv',index=False)
    return pd.DataFrame(events)


def figures(event,out):
    for trans in [2019,2023]:
        fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
        for ax,sector in zip(axes,['manufacturing','clean_power']):
            threshold=.99 if trans==2019 else .5
            d=event[(event.transition==trans)&event.first_year.eq(2017 if trans==2019 else 2019)&
                    event.threshold.eq(threshold)&event['sample'].eq('continuous_member')&
                    event.sector.eq(sector)&event.outcome.eq('projects')].sort_values('year')
            ax.errorbar(d.year,d.estimate,yerr=[d.estimate-d.lo,d.hi-d.estimate],fmt='o-',capsize=3)
            ax.axhline(0,color='gray',lw=1);ax.axvline(trans-.5,color='gray',ls='--')
            ax.set(title=sector.replace('_',' ').title(),xlabel='Announcement year',ylabel='Gainers minus losers, change\nrelative to last pre-year (projects/district)')
            ax.set_xticks(d.year)
        fig.suptitle(f'House majority transition {trans} | fixed common core, continuous member, {threshold:.0%} minimum overlap')
        fig.savefig(out/f'pretrends_{trans}.png',dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'results')
    parser.add_argument('--crosswalk-cache',type=Path)
    parser.add_argument('--facilities',type=Path,default=ROOT/'data/manufacturing_energy_and_industry_facility_metadata.csv')
    args=parser.parse_args();out=args.out
    out.mkdir(parents=True,exist_ok=False)
    pyproj.network.set_network_enabled(False)
    maps=geographies(); terms=member_terms(); terms.to_csv(out/'dated_member_terms.csv',index=False)
    if args.crosswalk_cache:
        cw=pd.read_csv(args.crosswalk_cache);cw.to_csv(out/'geographic_crosswalk.csv',index=False)
    else:cw=crosswalk(maps,out)
    transition_switches(terms,cw,out)
    roster=pd.concat([sample_roster(terms,maps,cw,t,f,h) for t,f in [(2019,2017),(2023,2021),(2023,2019)]
                      for h in [.99,.95,.8,.5]],ignore_index=True)
    roster.to_csv(out/'district_roster.csv',index=False)
    raw=pd.read_csv(args.facilities,skiprows=5,low_memory=False)
    assert raw.unique_id.is_unique
    raw['date']=pd.to_datetime(raw.Announcement_Date,format='mixed')
    raw.groupby([raw.date.dt.year,'Segment']).size().rename('projects').to_csv(out/'raw_annual_coverage.csv')
    loc=assignments(raw,maps,out)
    event=analyze(roster,raw,loc,out);figures(event,out)
    files=[args.facilities.resolve(),*sorted((ROOT/'data/redesign_raw/legislators').glob('*.json')),
           *sorted((ROOT/'data/redesign_raw/boundaries').glob('*.zip')),ROOT/'revision_analysis/nc_117.zip',Path(__file__)]
    pd.DataFrame([{'file':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]).to_csv(out/'input_hashes.csv',index=False)
    print(pd.read_csv(out/'switch_summary.csv').to_string(index=False))
    print(pd.read_csv(out/'support.csv').query('threshold==.99 and sample=="continuous_member" and sector!="legacy_all"').to_string(index=False))


if __name__=='__main__':main()
