"""Independent reconciliations on saved outputs; optional byte-for-byte fresh-run comparison."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd


def verify(path, replica=None):
    def read(name):
        return pd.read_csv(path/(name+".csv"))
    keys=["year","state","district"]
    district=read("district_elections").set_index(keys).sort_index()
    facilities=read("facility_assignments")
    p=facilities.query('post_status=="included"')
    assert facilities.unique_id.is_unique and district.index.is_unique
    for column,values in {"projects":p.groupby(keys).size(),"dollars":p.groupby(keys).capex.sum(),
                          "unknown":p.capex.isna().groupby([p[k] for k in keys]).sum(),
                          "clear":p.exposure.eq("clear").groupby([p[k] for k in keys]).sum()}.items():
        np.testing.assert_allclose(district[column], values.reindex(district.index,fill_value=0))
    n=read("main_rd").iloc[0]
    panel=district.reset_index()
    local=panel[(panel.running>-n.h_left)&(panel.running<n.h_right)]
    audit=read("effective_winner_audit")
    assert set(map(tuple,audit[keys].values))==set(map(tuple,local[keys].values))
    np.testing.assert_allclose(audit.running,(audit.R_votes-audit.D_votes)/(audit.R_votes+audit.D_votes))
    assert audit.winner.eq(np.where(audit.R_votes>audit.D_votes,"R","D")).all()
    assert (audit.audit.notna() & audit.audit.ne("review")).all()
    bins=read("cutoff_plot_bins")
    for (outcome,side),sub in bins.groupby(["outcome","side"]):
        included=local[local.running*side>0]
        assert sub.n.sum()==len(included)
        np.testing.assert_allclose(np.average(sub.y,weights=sub.n),included[outcome].mean())
    for r in read("policy_counts").itertuples():
        s=panel[(panel.year==2022)&(panel.winner==r.side)&(panel.running.abs()<r.bandwidth)]
        assert (r.n,r.event_districts,r.projects)==(len(s),(s[r.definition]>0).sum(),s[r.definition].sum())
    sens=read("sensitivity")
    np.testing.assert_allclose(sens.iloc[0][["estimate","low","high"]].astype(float),
                               n[["estimate","low","high"]].astype(float))
    if replica:
        left={p.name:p for p in path.iterdir() if p.is_file()}
        right={p.name:p for p in replica.iterdir() if p.is_file()}
        assert left.keys()==right.keys()
        for name in left:
            assert hashlib.sha256(left[name].read_bytes()).digest()==hashlib.sha256(right[name].read_bytes()).digest(),name
        print(f"{len(left)} artifacts byte-identical in independent runs.")
    print(f"Reconciled {len(p)} project assignments, {len(district)} district-elections and {len(audit)} effective winner audits.")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results",type=Path)
    parser.add_argument("--replica",type=Path)
    args=parser.parse_args()
    verify(args.results,args.replica)
