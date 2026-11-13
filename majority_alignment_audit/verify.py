"""Reconcile generated panels and independently check headline coefficient."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from audit import ROOT

HERE=Path(__file__).resolve().parent
KEY=['transition','first_year','threshold','sample','sector','state','district']


def main():
    checks={}
    for release in ['frozen','latest']:
        p=HERE/release
        q=pd.read_csv(p/'district_quarter_panel.csv',low_memory=False);a=pd.read_csv(p/'district_year_panel.csv');s=pd.read_csv(p/'support.csv')
        assert not q.duplicated(KEY+['quarter']).any()
        assert not a.duplicated(KEY+['year']).any()
        assert np.allclose(q.projects,q.known+q.unknown)
        assert (q[['projects','known','unknown','dollars']]>=0).all().all()
        agg=q.groupby(KEY+['year'])[['projects','known','unknown','dollars']].sum().sort_index()
        actual=a.set_index(KEY+['year'])[agg.columns].sort_index()
        assert np.allclose(agg,actual)
        assert np.allclose(a.log_capex,np.log1p(a.dollars))
        for row in s.itertuples():
            z=a[(a.transition==row.transition)&a.first_year.eq(row.first_year)&a.threshold.eq(row.threshold)&a['sample'].eq(row.sample)&a.sector.eq(row.sector)&a.gainer.eq(row.gainer)]
            assert z[z.year>=row.transition].projects.sum()==row.post_projects
            assert z[z.year<row.transition].projects.sum()==row.pre_projects
            assert len(z[['state','district']].drop_duplicates())==row.districts
        checks[release+'_panels_reconciled']=True
    original=HERE/'latest';replica=HERE/'replication_latest'
    hashes={}
    for p in original.iterdir():
        if p.is_file():
            assert p.read_bytes()==(replica/p.name).read_bytes(),p.name
            hashes[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
    checks['fresh_identical_artifacts']=len(hashes)
    checks['latest_artifact_sha256']=hashes
    # Independent FWL point-estimate check, no reuse of estimation routine.
    d=pd.read_csv(HERE/'power_plants/district_year_panel.csv')
    d=d[d.transition.eq(2023)&d.threshold.eq(.5)&d['sample'].eq('continuous_member')&d.sector.eq('wind_solar')]
    pre=d[d.year<2023].groupby(['state','district']).net_mw.mean()
    post=d[d.year>=2023].groupby(['state','district']).net_mw.mean()
    x=d[['state','district','gainer']].drop_duplicates().set_index(['state','district']).assign(delta=post-pre).reset_index()
    x['t']=x.gainer-x.groupby('state').gainer.transform('mean')
    x['y']=x.delta-x.groupby('state').delta.transform('mean')
    b=(x.t*x.y).sum()/(x.t*x.t).sum()
    estimates=pd.read_csv(HERE/'power_plants/diagnostics.csv')
    e=estimates[estimates.transition.eq(2023)&estimates.threshold.eq(.5)&estimates['sample'].eq('continuous_member')&estimates.sector.eq('wind_solar')&estimates.state_fe&estimates.test.eq('post_minus_pre_relative_change')].estimate.item()
    assert np.isclose(b,e,atol=1e-10)
    checks['independent_capacity_coefficient']=float(b)
    # Spot-check politically recognizable continuous members and actual dated terms.
    r=pd.read_csv(HERE/'latest/district_roster.csv').fillna({'transition_exclusion':''})
    witnesses=r[(r.transition==2023)&r.first_year.eq(2019)&r.threshold.eq(.5)&r.transition_exclusion.eq('')&r.continuous_member].sort_values(['gainer','state','district']).groupby('gainer').head(4)
    assert witnesses.prior_member.eq(witnesses.member).all()
    assert witnesses.gainer.eq(witnesses.party.eq('Republican').astype(int)).all()
    witnesses.to_csv(HERE/'political_assignment_witnesses.csv',index=False)
    checks['political_witnesses']=len(witnesses)
    (HERE/'VERIFICATION.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps({k:v for k,v in checks.items() if k!='latest_artifact_sha256'},indent=2))


if __name__=='__main__':main()
