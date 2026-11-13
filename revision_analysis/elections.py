"""Build the election roster and reconcile candidate ballot lines to Clerk district tables."""
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def district_blocks(pdf, state_names):
    converter = shutil.which("pdftotext") or str(Path.home() / "miniconda3/envs/pypsa-earth/bin/pdftotext")
    doc = subprocess.check_output([converter, "-layout", str(pdf), "-"]).decode()
    blocks = defaultdict(list)
    state, district, active = None, None, False
    for raw in doc.splitlines():
        line = raw.strip()
        heading = re.split(r"—|–|--", line)[0].strip()
        if heading in state_names:
            if state != state_names[heading]:
                active, district = False, None
            state = state_names[heading]
        if re.match(r"FOR (?:UNITED STATES )?REPRESENTATIVE", line):
            active = True
        elif "Recapitulation of Votes" in line or line.startswith("FOR PRESIDENT") or line.startswith("FOR UNITED STATES SENATOR"):
            active = False
        if active:
            match = re.match(r"^(\d+)\.\s+\S", line)
            if match:
                district = int(match[1])
            if "AT LARGE" == line:
                district = 0
            if district is not None and line:
                blocks[state, district].append(line)
    return blocks


def party(name):
    if name.startswith("DEMOCRAT"):
        return "D"
    if name == "REPUBLICAN":
        return "R"
    return ""


def roster(data):
    e = pd.read_csv(data/"MIT_election_lab/1976-2024-house.tab", low_memory=False)
    e = e[e.year.isin([2018, 2020, 2022]) & e.stage.eq("GEN") & ~e.special]
    names = dict(zip(e.state.str.upper(), e.state_po))
    out = []
    for year, cycle in e.groupby("year"):
        blocks = district_blocks(data/f"election_stats/{year}election.pdf", names)
        for (state, district), race in cycle.groupby(["state_po", "district"]):
            row = dict(year=year, state=state, district=int(district), reason="", audit="")
            out.append(row)
            if state=="DC":
                row["reason"]="nonvoting delegate"; continue
            ballot = race[~race.writein & race.candidate.notna() & race.party.notna()].copy()
            ballot["major"] = ballot.party.map(party)
            candidates = ballot.groupby("candidate").agg(votes=("candidatevotes","sum"),
                major=("major",lambda x: "".join(sorted(set(x)-{""})))) .sort_values("votes",ascending=False)
            if (year,state,district)==(2018,"NC",9):
                row["reason"]="2018 result not certified"; continue
            if (year,state)==(2022,"AK"):
                # Official November 30 final tabulation, p.3. Clerk reports first preferences.
                row.update(R_name="SARAH PALIN",D_name="MARY SATTLER PELTOLA",R_votes=112471,
                           D_votes=137263,running=(112471-137263)/249734,
                           audit="AK final round: https://www.elections.alaska.gov/results/22GENR/US%20REP.pdf")
                continue
            if len(candidates)<2 or set(candidates.major.iloc[:2])!={"R","D"}:
                row["reason"]="no decisive R versus D pair among leading candidates"; continue
            if state=="LA" and (candidates.major.value_counts().reindex(["R","D"],fill_value=0)>1).any():
                row["reason"]="Louisiana multiparty primary with multiple same-party candidates"; continue
            block="\n".join(blocks[state,district]).replace("’","'")
            proofs=[]
            for name, cand in candidates.iloc[:2].iterrows():
                side=cand.major
                lines=ballot[ballot.candidate==name]
                row[side+"_name"]=name
                row[side+"_votes"]=int(cand.votes)
                valid=True
                for line in lines.itertuples():
                    number=f"{line.candidatevotes:,}"
                    if not line.candidatevotes: continue
                    pattern = "Democrat" if line.major=="D" else ("(?:Republican|GOP)" if line.major=="R" else re.escape(line.party.split()[0]))
                    valid &= bool(re.search(pattern+r"[^\n]*?(?:\n\s*)?(?<![\d,])"+re.escape(number)+r"(?![\d,])",block,re.I))
                proofs.append(valid)
            row["audit"]="Clerk district ballot lines agree" if all(proofs) else "review"
            if state=="NY":
                totals={}
                side=None
                for entry in blocks[state,district]:
                    match=re.search(r"^(.*?)\s*\.{2,}\s*([\d,]+)$",entry)
                    if not match: continue
                    label,number=match.groups()
                    if re.search(r",\s*(Democrat|Republican)\s*$",label,re.I):
                        side="R" if label.lower().endswith("republican") else "D"
                        if side in totals: raise ValueError((year,state,district,entry))
                        totals[side]=int(number.replace(",",""))
                    elif side and "," not in label and not re.search(r"blank|scatter|void|write.in|other",label,re.I):
                        totals[side]+=int(number.replace(",",""))
                    else: side=None
                if set(totals)=={"D","R"}:
                    for side in totals:
                        row["MIT_"+side+"_votes"]=row[side+"_votes"]
                        row[side+"_votes"]=totals[side]
                    row["audit"]="Clerk candidate plus fusion continuation lines"
            if (year,state,district) in {(2018,"SD",0),(2018,"ME",2)}:
                d,r=(121033,202695) if state=="SD" else (142440,138931)
                assert f"{d:,}" in block and f"{r:,}" in block
                row.update(D_votes=d,R_votes=r,audit="Clerk corrected district totals")
            row["running"]=(row["R_votes"]-row["D_votes"])/(row["R_votes"]+row["D_votes"])
            if row["running"]==0: row["reason"]="tie"
    return pd.DataFrame(out)
