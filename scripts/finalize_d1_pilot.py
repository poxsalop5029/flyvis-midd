"""Finalize D1 pilot tables from saved reduction outputs; performs no FlyVis inference."""
from pathlib import Path
import base64, hashlib, importlib.metadata as im, json, os, platform, sys
import numpy as np
import pandas as pd
import torch

ROOT=next((p for p in (Path.cwd().resolve(),*Path.cwd().resolve().parents)
           if (p/"data/precomputed/supplementary/biological_constraints/D1/pilot/pilot_integrations_progress.csv").is_file()),None)
if ROOT is None: raise FileNotFoundError("Pilot integration ledger not found.")
OUT=ROOT/"data/precomputed/supplementary/biological_constraints/D1/pilot"
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
progress=pd.read_csv(OUT/"pilot_integrations_progress.csv")
expected={(f"flow/0000/{i:03d}",phase,cond) for i in (0,5,10) for phase in range(12) for cond in ("original","mirror")}
got=set(zip(progress.model_id,progress.phase_index,progress.condition))
assert len(progress)==72 and progress.completed.all() and got==expected, "The exact 72 integrations are not complete."
metrics=pd.read_csv(OUT/"pilot_model_phase_metrics.csv")
metrics["model_short_id"]=metrics.model_short_id.astype(int).map(lambda x:f"{x:03d}")
assert len(metrics)==36 and not metrics.duplicated(["model_short_id","phase_index"]).any()
assert set(metrics.model_short_id)=={"000","005","010"}
assert np.allclose(metrics.phase_deg,metrics.phase_index*1.25)
decoded_path=ROOT/"data/precomputed/fig04/official50_phase_pairs.csv"
decoded=pd.read_csv(decoded_path)
decoded=decoded.loc[decoded.group.eq("3MIDD_048")].copy()
decoded["model_short_id"]=decoded.model_id.astype(int).map(lambda x:f"{x:03d}")
decoded=decoded.loc[decoded.model_short_id.isin(("000","005","010"))].copy()
decoded["phase_deg"]=decoded.phase_index.astype(float)*1.25
decoded=decoded.rename(columns={"A":"A_flow"})
assert len(decoded)==36 and not decoded.duplicated(["model_short_id","phase_index"]).any()
joined=metrics.merge(decoded[["model_short_id","phase_index","phase_deg","A_flow","T_original","T_mirror","S","G","actual_reversal"]],
    on=["model_short_id","phase_index"],how="left",validate="one_to_one",suffixes=("","_decoded"))
assert len(joined)==36 and joined.A_flow.notna().all()
assert np.allclose(joined.phase_deg,joined.phase_deg_decoded,atol=1e-12)
assert np.isfinite(joined[["ON_A","OFF_A","I_ONOFF","A_flow"]].to_numpy(float)).all()
joined.to_csv(OUT/"pilot_joined_decoded_flow.csv",index=False)
on=joined.ON_A.to_numpy(float); off=joined.OFF_A.to_numpy(float)
onr=float(np.sqrt(np.mean(on**2))); offr=float(np.sqrt(np.mean(off**2)))
ratio=onr/offr if offr else float("inf")
raw=pd.read_csv(OUT/"pilot_source_current_raw_summary.csv")
total=raw.loc[raw.source_type.eq("__selected_sources_sum__")]
cancel=float(total.abs_mean_over_mean_absolute.median())
qa=[]
for key,v in (("ON_A",on),("OFF_A",off),("I_ONOFF",joined.I_ONOFF.to_numpy(float))):
    qa.append({"metric":key,"mean":v.mean(),"std":v.std(ddof=0),"median":np.median(v),
        "min":v.min(),"max":v.max(),"rms":np.sqrt(np.mean(v*v)),"n":len(v)})
qa.extend([{"metric":"ON_OFF_RMS_ratio","mean":ratio,"median":ratio},
    {"metric":"median_spatial_cancellation_ratio","mean":cancel,"median":cancel}])
pd.DataFrame(qa).to_csv(OUT/"pilot_scale_qa.csv",index=False)
model_means=joined.groupby("model_short_id")[["I_ONOFF","A_flow"]].mean()
rho=float(model_means.I_ONOFF.corr(model_means.A_flow,method="spearman"))
shape=pd.read_csv(OUT/"pilot_current_shape_qa.csv")
assert len(shape)==72 and shape.current_finite.all() and shape.all_edge_indices_match.all()
assert shape.current_shape.nunique()==1 and shape.current_dtype.nunique()==1
assert (shape.SourceCurrentView_pairs==24).all()
scale_issue=not .1<=ratio<=10
spatial_issue=cancel<.01
flyvis_dist=im.distribution("flyvis")
direct=json.loads((flyvis_dist._path/"direct_url.json").read_text())
source_root=Path(__import__("flyvis").__file__).resolve().parent
record_ok=True
for f in flyvis_dist.files or ():
    h=f.hash
    if h is None: continue
    path=flyvis_dist.locate_file(f)
    if not path.is_file(): record_ok=False; break
    actual=base64.urlsafe_b64encode(hashlib.new(h.mode,path.read_bytes()).digest()).decode().rstrip("=")
    if actual!=h.value: record_ok=False; break
assert record_ok, "Installed FlyVis RECORD hash mismatch."
assert flyvis_dist.version=="1.2.0" and direct.get("vcs_info",{}).get("commit_id")=="92b3845cc426dd309a1a0e1b3890156c42e14021"
assert not direct.get("dir_info",{}).get("editable",False)
technical=(record_ok and len(shape)==72 and len(joined)==36 and shape.current_finite.all() and shape.all_edge_indices_match.all())
decision="PILOT_FAIL" if not technical else ("NEEDS_DESIGN_REVIEW" if scale_issue or spatial_issue else "PILOT_PASS")
shape["device"]=str(torch.get_default_device())
shape.to_csv(OUT/"pilot_current_shape_qa.csv",index=False)
srcs=[source_root/"network"/"network.py",source_root/"network"/"dynamics.py",
      source_root/"utils"/"activity_utils.py",source_root/"connectome"/"connectome.py"]
srcs += [ROOT/"scripts/generate_d1_source_currents.py",ROOT/"scripts/finalize_d1_pilot.py",ROOT/"notebooks/supplementary/biological_constraints_D1_acquisition.ipynb"]
input_paths=[ROOT/"data/precomputed/fig04/phase_stimulus_manifest.csv",decoded_path,
    ROOT/"data/precomputed/supplementary/biological_constraints/B1/t4t5_activity_model_phase.csv",
    ROOT/"notebooks/supplementary/biological_constraints_D1_acquisition.ipynb"]
output_names=("pilot_source_current_raw_summary.csv","pilot_target_node_distribution.csv",
    "pilot_model_phase_metrics.csv","pilot_current_shape_qa.csv","pilot_scale_qa.csv",
    "pilot_joined_decoded_flow.csv","pilot_integrations_progress.csv")
outputs={n:sha(OUT/n) for n in output_names}
source_hashes={str(p):sha(p) for p in srcs}
input_hashes={str(p):sha(p) for p in input_paths}
wall=float(progress.elapsed_seconds.max())
sizes=sum((OUT/n).stat().st_size for n in output_names)
manifest={
 "environment_gate":"PINNED_ENV_PASS","FlyVis_version":flyvis_dist.version,
 "FlyVis_commit":direct["vcs_info"]["commit_id"],"FlyVis_path":str(source_root),
 "FlyVis_install_mode":"noneditable VCS install","checkout_cleanliness":"noneditable exact-commit package; no mutable checkout",
 "installed_RECORD_hashes_verified":record_ok,
 "Python":platform.python_version(),"Python_executable":sys.executable,
 "torch":im.version("torch"),"device":str(torch.get_default_device()),
 "FLYVIS_ROOT_DIR":str(Path.home()/"flyvis"/"data"),
 "models":json.dumps(["flow/0000/000","flow/0000/005","flow/0000/010"]),
 "phases_deg":json.dumps([i*1.25 for i in range(12)]),
 "stimulus":"3MIDD C=48 Original/Mirror; 382 px; inner/outer diameters 55/382; BoxEye extent/kernel 15/13",
 "frames":50,"dt_seconds":0.02,"fade_in_seconds":1.0,"analysis_frames":"10-49 inclusive",
 "expected_integrations":72,"completed_integrations":len(progress),"wall_seconds":wall,
 "peak_observed_rss_bytes":int(shape.rss_bytes.max()),
 "source_ON":"Mi1,Tm3","target_ON":"T4a-d","source_OFF":"Tm1,Tm2,Tm4,Tm9","target_OFF":"T5a-d",
 "aggregation":"sum signed source edges per target node; mean frames; mean nodes within subtype; equal-weight four subtypes",
 "ON_A":"(R_ON_original-R_ON_mirror)/2","OFF_A":"(R_OFF_original-R_OFF_mirror)/2",
 "I_ONOFF":"ON_A-OFF_A; no sign flip or normalization",
 "current_response_shape":shape.current_shape.iloc[0],"current_dtype":shape.current_dtype.iloc[0],
 "current_dimensions":"frame × edge; official API squeezes single sample/batch",
 "current_semantics":"signed synaptic weight × presynaptic activation; sign is included in edge weight",
 "SourceCurrentView":"selects per-source edge columns; does not aggregate source types",
 "SourceCurrentView_checks_per_integration":24,"all_source_indices_match":True,
 "ON_A_RMS":onr,"OFF_A_RMS":offr,"ON_OFF_RMS_ratio":ratio,
 "median_spatial_cancellation_ratio":cancel,"scale_issue":scale_issue,"spatial_cancellation_issue":spatial_issue,
 "exploratory_Spearman_model_mean_N3":rho,"inferential_statistics":"not performed",
 "decoded_join_keys":"model_short_id + phase_index","decoded_join_rows":len(joined),"decoded_join_exact":True,
 "input_sha256":json.dumps(input_hashes,sort_keys=True),"source_sha256":json.dumps(source_hashes,sort_keys=True),
 "output_sha256_excluding_manifest":json.dumps(outputs,sort_keys=True),
 "full_cohort_workload":"1200 integrations (50×12×2)",
 "estimated_full_cohort_wall_seconds":wall*(1200/72),
 "estimated_full_cohort_disk_bytes":int(sizes*50/3),
 "persistent_response_cache":"none; progress ledger was written per integration",
 "decision":decision,
}
pd.DataFrame([{"key":k,"value":json.dumps(v) if isinstance(v,(dict,list,tuple)) else str(v)}
    for k,v in manifest.items()]).to_csv(OUT/"pilot_reproducibility_manifest.csv",index=False)
print({"decision":decision,"integrations":"72/72","wall_seconds":wall,
 "ON_A_RMS":onr,"OFF_A_RMS":offr,"ON_OFF_RMS_ratio":ratio,
 "spatial_cancellation_ratio":cancel,"descriptive_spearman_rho_N3":rho,
 "decoded_join":"36/36 exact","estimated_full_cohort_seconds":wall*1200/72,
 "estimated_disk_bytes":int(sizes*50/3)})
