#!/usr/bin/env python3
"""Run the prespecified D1 official-50 source-current experiment.

Each integration is atomically written to official50_work/integrations, so a
failed run can resume without repeating completed model/phase/condition tuples.
"""
from __future__ import annotations
import argparse, base64, hashlib, importlib.metadata as im, json, os, platform, resource, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

VERSION='1.2.0'; COMMIT='92b3845cc426dd309a1a0e1b3890156c42e14021'
SHORT=tuple(f'{i:03d}' for i in range(50)); MODELS=tuple(f'flow/0000/{x}' for x in SHORT)
PHASES=np.arange(12,dtype=float)*1.25; CONDITIONS=('original','mirror')
ON_SRC=('Mi1','Tm3'); OFF_SRC=('Tm1','Tm2','Tm4','Tm9')
ON_TGT=tuple(f'T4{x}' for x in 'abcd'); OFF_TGT=tuple(f'T5{x}' for x in 'abcd')
DT=.02; FRAMES=50; FADE=1.; START=10; STOP=50; SEED=20261006

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def rss():
    try:
        import psutil
        return int(psutil.Process().memory_info().rss)
    except Exception:
        x=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(x if sys.platform=='darwin' else x*1024)
def root_repo():
    for p in (Path.cwd().resolve(),*Path.cwd().resolve().parents):
        if (p/'data/precomputed/fig04/official50_phase_pairs.csv').is_file(): return p
    raise FileNotFoundError('Run from flyvis-midd repository.')
def atomic_json(path,obj):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,sort_keys=True,allow_nan=False)+'\n'); tmp.replace(path)
def append_csv(path,rows):
    if rows: pd.DataFrame(rows).to_csv(path,mode='a',header=not path.exists(),index=False)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run-inference',action='store_true'); a=ap.parse_args()
    if not a.run_inference: raise SystemExit('Safe default: pass --run-inference to run 1200 integrations.')
    root=root_repo(); out=root/'data/precomputed/supplementary/biological_constraints/D1'; work=out/'official50_work'; idir=work/'integrations'; idir.mkdir(parents=True,exist_ok=True)
    final_names=('on_off_model_phase_metrics.csv','on_off_model_summary.csv','on_off_vs_decoded_flow.csv','on_off_primary_stats.csv','on_off_secondary_stats.csv','on_off_stage_exploratory.csv','on_off_reproducibility_manifest.csv','on_off_source_contributions.csv')
    existing=[n for n in final_names if (out/n).exists()]
    if existing: raise FileExistsError(f'Refusing to overwrite finalized D1 outputs: {existing}')
    roots=[Path(os.environ['FLYVIS_ROOT_DIR']).expanduser()] if os.environ.get('FLYVIS_ROOT_DIR') else []
    roots += [Path.home()/'flyvis'/'data']; data_root=next((p.resolve() for p in roots if (p/'results/flow/0000/000').is_dir()),None)
    if data_root is None: raise FileNotFoundError('Set FLYVIS_ROOT_DIR to FlyVis pretrained data.')
    os.environ['FLYVIS_ROOT_DIR']=str(data_root)
    import flyvis, torch
    from flyvis.network import NetworkView
    from flyvis.utils.activity_utils import SourceCurrentView
    from flyvis.connectome.connectome import ReceptiveFields
    dist=im.distribution('flyvis'); direct=json.loads((dist._path/'direct_url.json').read_text())
    commit=direct.get('vcs_info',{}).get('commit_id'); editable=bool(direct.get('dir_info',{}).get('editable',False))
    rec_ok=True
    for f in dist.files or ():
        if f.hash is None: continue
        q=dist.locate_file(f)
        if not q.is_file(): rec_ok=False; break
        actual=base64.urlsafe_b64encode(hashlib.new(f.hash.mode,q.read_bytes()).digest()).decode().rstrip('=')
        if actual!=f.hash.value: rec_ok=False; break
    if not (flyvis.__version__==VERSION and commit==COMMIT and not editable and rec_ok and hasattr(flyvis.Network,'current_response')):
        raise RuntimeError(f'Pinned package gate failed: version={flyvis.__version__}, commit={commit}, editable={editable}, RECORD={rec_ok}')
    phase_path=root/'data/precomputed/fig04/phase_stimulus_manifest.csv'; decoded_path=root/'data/precomputed/fig04/official50_phase_pairs.csv'
    b1_path=root/'data/precomputed/supplementary/biological_constraints/B1/t4t5_activity_model_phase.csv'
    inputs={str(p):sha(p) for p in (phase_path,decoded_path,b1_path)}
    phase_manifest=pd.read_csv(phase_path)
    from flyvis_midd.stimuli import build_3midd_sequences,render_sector_image
    from flyvis.datasets.rendering import BoxEye
    class Dataset:
        def __init__(self,movie): self.movie=movie; self.dt=None
        def __len__(self): return 1
        def __getitem__(self,i): return {'lum':self.movie}
    def make_movie(phase,condition):
        seq=build_3midd_sequences(48)[condition]
        image,_=render_sector_image(seq,image_size=382,inner_diameter=55,outer_diameter=382,phase_deg=float(phase))
        pi=int(round(float(phase)/1.25)); key=f'3MIDD_048_p{pi:02d}__{condition}'
        rows=phase_manifest.loc[phase_manifest.condition.eq(key)]
        if len(rows)!=1: raise AssertionError(f'Expected one canonical manifest row for {key}')
        row=rows.iloc[0]
        if hashlib.sha256(image.tobytes()).hexdigest()!=row.image_sha256: raise AssertionError(f'Stimulus image hash mismatch: {key}')
        movie=np.repeat(image[None,None,:,:],FRAMES,axis=1).astype(np.float32)/255.
        tensor=torch.as_tensor(movie); rendered=BoxEye(extent=15,kernel_size=13)(tensor)
        if tuple(rendered.shape)!=(1,FRAMES,1,721): raise ValueError(f'Unexpected BoxEye shape {tuple(rendered.shape)}')
        rh=hashlib.sha256(rendered.detach().cpu().numpy().tobytes()).hexdigest()
        # The canonical image hash is enforced exactly. The phase manifest's
        # rendered hash predates this runner and its byte serialization is not
        # documented; preserve both values instead of falsely treating them as equal.
        return Dataset(rendered[0]), {'key':key,'image_sha256':row.image_sha256,
            'rendered_sha256_observed':rh,'rendered_sha256_manifest':row.rendered_sha256,
            'rendered_hash_matches_manifest':bool(rh==row.rendered_sha256)}
    def validate_network(mid):
        p=data_root/'results'/mid
        if not (p/'best_chkpt').is_file(): raise FileNotFoundError(f'Missing best checkpoint: {p}')
        v=NetworkView(mid,root_dir=data_root/'results'); net=v.init_network('best'); net.eval()
        edges=net.connectome.edges.to_df().reset_index(drop=True)
        pairs=[]
        for sources,targets in ((ON_SRC,ON_TGT),(OFF_SRC,OFF_TGT)):
            for target in targets:
                rf=ReceptiveFields(target,edges)
                for source in sources:
                    n=len(rf[source]) if source in rf.source_types else 0
                    if n==0: raise RuntimeError(f'Missing required edge pair {source}->{target} in {mid}')
                    pairs.append((source,target,n))
        if net.training: raise RuntimeError(f'Network not in eval mode: {mid}')
        return v,net,pairs
    def reduce(net,cur,mid,pi,phase,cond):
        edges=net.connectome.edges.to_df().reset_index(drop=True); nodes=net.connectome.nodes.to_df().reset_index(drop=True)
        if cur.shape!=(FRAMES,len(edges)) or not np.isfinite(cur).all(): raise ValueError(f'Bad current tensor shape/finite: {mid} {cur.shape}')
        if sorted(nodes['index'].astype(int))!=list(range(len(nodes))): raise ValueError('Node index mapping is not contiguous.')
        source_map={**{t:ON_SRC for t in ON_TGT},**{t:OFF_SRC for t in OFF_TGT}}
        node_means={}; source_rows=[]; qa_pairs=[]; win=cur[START:STOP].astype(np.float64,copy=False)
        for target,sources in source_map.items():
            selected_nodes=nodes.loc[nodes.type.eq(target)].sort_values('index'); ids=selected_nodes['index'].to_numpy(dtype=int)
            local={int(x):i for i,x in enumerate(ids)}; rf=ReceptiveFields(target,edges); view=SourceCurrentView(rf,cur); parts={}
            for source in sources:
                if source not in view.source_types: raise RuntimeError(f'Missing source {source}->{target}')
                all_ix=np.asarray(rf[source].index,dtype=int); observed=np.asarray(view[source])
                if observed.shape!=(FRAMES,len(all_ix)) or not np.array_equal(observed,cur[:,all_ix]): raise AssertionError(f'SourceCurrentView mismatch {source}->{target}')
                selected=edges.iloc[all_ix]; mask=(selected.source_type.eq(source)&selected.target_type.eq(target)).to_numpy(); ix=all_ix[mask]; selected=edges.iloc[ix]
                if not len(ix): raise RuntimeError(f'No edges {source}->{target}')
                tgt=selected.target_index.map(local)
                if tgt.isna().any(): raise RuntimeError('Source edges map to unknown target index')
                tgt=tgt.to_numpy(dtype=int); vals=np.zeros((STOP-START,len(ids)),dtype=np.float64)
                for fi,row in enumerate(win): np.add.at(vals[fi],tgt,row[ix])
                per_node=vals.mean(axis=0); parts[source]=per_node
                source_rows.append({'model_id':mid,'model_short_id':mid[-3:],'phase_index':pi,'phase_deg':phase,'condition':cond,
                    'pathway':'ON' if target.startswith('T4') else 'OFF','target_type':target,'source_type':source,
                    'mean':float(per_node.mean()),'sd_across_target_nodes':float(per_node.std()),
                    'rms_across_target_nodes':float(np.sqrt(np.mean(per_node**2))),'n_target_nodes':len(ids),'n_edges':len(ix)})
                qa_pairs.append({'target_type':target,'source_type':source,'n_edges':len(ix),'n_target_nodes':len(ids),'edge_indices_match':True})
            total=np.sum(np.stack([parts[s] for s in sources]),axis=0); node_means[target]=float(total.mean())
            source_rows.append({'model_id':mid,'model_short_id':mid[-3:],'phase_index':pi,'phase_deg':phase,'condition':cond,
                'pathway':'ON' if target.startswith('T4') else 'OFF','target_type':target,'source_type':'__selected_sources_sum__',
                'mean':float(total.mean()),'sd_across_target_nodes':float(total.std()),
                'rms_across_target_nodes':float(np.sqrt(np.mean(total**2))),'n_target_nodes':len(ids),'n_edges':sum(x['n_edges'] for x in qa_pairs if x['target_type']==target)})
        node_means['R_ON']=float(np.mean([node_means[t] for t in ON_TGT])); node_means['R_OFF']=float(np.mean([node_means[t] for t in OFF_TGT]))
        return node_means,source_rows,qa_pairs
    # Check every checkpoint and required source-target mapping before expensive inference.
    model_qc=[]; started=time.perf_counter(); peak=rss()
    for mid in MODELS:
        v,net,pairs=validate_network(mid)
        model_qc.append({'model_id':mid,'checkpoint':str(v.get_checkpoint('best')),'eval_mode':not net.training,
                         'source_target_pairs':len(pairs),'edges_per_pair':json.dumps({f'{s}->{t}':n for s,t,n in pairs},sort_keys=True)})
        del net,v
    pd.DataFrame(model_qc).to_csv(work/'official50_checkpoint_qa.csv',index=False)
    # Exact canonical decoded table, scoped only to 3MIDD C=48.
    dec=pd.read_csv(decoded_path); dec=dec.loc[dec.group.eq('3MIDD_048')].copy()
    dec['model_short_id']=dec.model_id.astype(int).map(lambda i:f'{i:03d}'); dec['phase_deg']=dec.phase_index.astype(float)*1.25; dec=dec.rename(columns={'A':'A_flow'})
    dec=dec.loc[dec.model_short_id.isin(SHORT)].copy()
    if len(dec)!=600 or dec.duplicated(['model_short_id','phase_index']).any(): raise AssertionError(f'Canonical decoded table should contain 600 unique official model-phase rows, got {len(dec)}')
    progress=[]; integrations=len(list(idir.glob('*.json')))
    if integrations>1200: raise RuntimeError('Unexpected excess integration artifacts.')
    for model_i,mid in enumerate(MODELS):
        v,net,_=validate_network(mid)
        for pi,phase in enumerate(PHASES):
            for cond in CONDITIONS:
                key=f'{mid[-3:]}_p{pi:02d}_{cond}'; result_path=idir/f'{key}.json'
                if result_path.exists():
                    saved=json.loads(result_path.read_text())
                    if saved.get('model_id')!=mid or saved.get('phase_index')!=pi or saved.get('condition')!=cond: raise RuntimeError(f'Corrupt staged integration {key}')
                    progress.append({'integration_key':key,'model_id':mid,'phase_index':pi,'phase_deg':phase,'condition':cond,'completed':True,'resumed':True})
                    continue
                dataset,stim_hash=make_movie(phase,cond)
                records=list(net.current_response(dataset,DT,t_pre=0.,t_fade_in=FADE))
                if len(records)!=1: raise RuntimeError(f'Expected one current_response record for {key}')
                stimulus,activity,current=map(np.asarray,records[0])
                if stimulus.shape[0]!=FRAMES or activity.shape[0]!=FRAMES: raise RuntimeError(f'Unexpected time series length for {key}')
                vals,source_rows,pairs=reduce(net,current,mid,pi,float(phase),cond)
                if len(pairs)!=24 or not all(x['edge_indices_match'] for x in pairs): raise RuntimeError(f'Source-target coverage failed for {key}')
                data={'model_id':mid,'phase_index':pi,'phase_deg':float(phase),'condition':cond,'stimulus':stim_hash,
                      'R_ON':vals['R_ON'],'R_OFF':vals['R_OFF'],'subtype_means':{t:vals[t] for t in ON_TGT+OFF_TGT},
                      'source_rows':source_rows,'qa_pairs':pairs,'stimulus_shape':str(stimulus.shape),'activity_shape':str(activity.shape),
                      'current_shape':str(current.shape),'current_dtype':str(current.dtype),'current_finite':bool(np.isfinite(current).all()),
                      'elapsed_seconds':time.perf_counter()-started,'rss_bytes':rss()}
                if not data['current_finite']: raise RuntimeError(f'Nonfinite currents at {key}')
                atomic_json(result_path,data); integrations+=1; peak=max(peak,rss())
                progress.append({'integration_key':key,'model_id':mid,'phase_index':pi,'phase_deg':phase,'condition':cond,'completed':True,'resumed':False})
                if integrations%10==0 or integrations==1200:
                    pd.DataFrame(progress).to_csv(work/'official50_integrations_progress.csv',index=False)
                    print(f'Progress {integrations}/1200; model={mid[-3:]} phase={phase:.2f} condition={cond}; elapsed={time.perf_counter()-started:.1f}s; peak_rss={peak}',flush=True)
                del dataset,records,stimulus,activity,current
        del net,v
    if integrations!=1200 or len(list(idir.glob('*.json')))!=1200: raise RuntimeError(f'Incomplete D1 workload: {integrations}/1200')
    records=[json.loads(p.read_text()) for p in sorted(idir.glob('*.json'))]
    if len(records)!=1200 or len({(r['model_id'],r['phase_index'],r['condition']) for r in records})!=1200: raise AssertionError('Integration uniqueness/completeness failed')
    raw=[]; shapes=[]; stimuli={}
    metric_rows=[]
    bykey={(r['model_id'][-3:],r['phase_index'],r['condition']):r for r in records}
    for r in records:
        raw.extend(r['source_rows']); stimuli[f"{r['model_id'][-3:]}|{r['phase_index']}|{r['condition']}"]=r['stimulus']
        shapes.append({'model_id':r['model_id'],'phase_index':r['phase_index'],'condition':r['condition'],'stimulus_shape':r['stimulus_shape'],
          'activity_shape':r['activity_shape'],'current_shape':r['current_shape'],'current_dtype':r['current_dtype'],'current_finite':r['current_finite'],
          'source_target_pairs':len(r['qa_pairs']),'all_edge_indices_match':all(x['edge_indices_match'] for x in r['qa_pairs']),'rss_bytes':r['rss_bytes']})
    for mid in MODELS:
        for pi,phase in enumerate(PHASES):
            o=bykey[(mid[-3:],pi,'original')]; m=bykey[(mid[-3:],pi,'mirror')]
            on=(o['R_ON']-m['R_ON'])/2; off=(o['R_OFF']-m['R_OFF'])/2
            row={'model_id':mid,'model_short_id':mid[-3:],'phase_index':pi,'phase_deg':phase,
                 'R_ON_original':o['R_ON'],'R_ON_mirror':m['R_ON'],'R_OFF_original':o['R_OFF'],'R_OFF_mirror':m['R_OFF'],
                 'ON_A':on,'OFF_A':off,'I_ONOFF':on-off}
            for t in ON_TGT+OFF_TGT: row[f'{t}_A']=(o['subtype_means'][t]-m['subtype_means'][t])/2
            metric_rows.append(row)
    metrics=pd.DataFrame(metric_rows); source_df=pd.DataFrame(raw); shape_df=pd.DataFrame(shapes)
    # Complete structural/finite QA before writing final tables.
    if len(metrics)!=600 or metrics.duplicated(['model_short_id','phase_index']).any(): raise AssertionError('Expected 600 unique model-phase metrics')
    if metrics.isna().any().any() or not np.isfinite(metrics.select_dtypes(include=[np.number]).to_numpy()).all(): raise AssertionError('Nonfinite model-phase values')
    if source_df.groupby(['model_short_id','phase_index','condition']).size().ne(32).any() or len(source_df)!=38400: raise AssertionError('Required source/target summary coverage failed')
    if shape_df.shape[0]!=1200 or not shape_df.current_finite.all() or not shape_df.all_edge_indices_match.all(): raise AssertionError('Integration QA failed')
    if set(source_df.target_type.unique())!=set(ON_TGT+OFF_TGT): raise AssertionError('T4/T5 subtype coverage failed')
    decoded_cols=['model_short_id','phase_index','phase_deg','A_flow','T_original','T_mirror','S','G','actual_reversal']
    joined=metrics.merge(dec[decoded_cols],on=['model_short_id','phase_index'],how='left',validate='one_to_one',suffixes=('','_decoded'))
    if len(joined)!=600 or joined.A_flow.isna().any() or not np.allclose(joined.phase_deg,joined.phase_deg_decoded): raise AssertionError('Decoded flow join not exact 600/600')
    if not np.isfinite(joined.select_dtypes(include=[np.number]).to_numpy()).all(): raise AssertionError('Decoded joined table contains NaN/Inf')
    out_metrics=metrics.copy(); out_metrics.to_csv(out/'on_off_model_phase_metrics.csv',index=False)
    joined.to_csv(out/'on_off_vs_decoded_flow.csv',index=False); source_df.to_csv(out/'on_off_source_contributions.csv',index=False)
    shape_df.to_csv(work/'official50_current_shape_qa.csv',index=False)
    pd.DataFrame(progress).to_csv(work/'official50_integrations_progress.csv',index=False)
    total_elapsed=time.perf_counter()-started
    manifest={
      'FlyVis_version':flyvis.__version__,'FlyVis_commit':commit,'FlyVis_path':str(Path(flyvis.__file__).resolve()),'editable_install':editable,'installed_RECORD_hashes_verified':rec_ok,
      'Python':platform.python_version(),'torch':torch.__version__,'device':str(torch.get_default_device()),'FLYVIS_ROOT_DIR':str(data_root),
      'model_ids':json.dumps(MODELS),'phases_deg':json.dumps(PHASES.tolist()),'conditions':json.dumps(CONDITIONS),'stimulus':'3MIDD C=48 Original/Mirror canonical B1 single BoxEye',
      'image_dimensions':'382x382','inner_diameter_px':55,'outer_diameter_px':382,'BoxEye_extent':15,'BoxEye_kernel_size':13,'frames':FRAMES,'dt_seconds':DT,'fade_in_seconds':FADE,
      'analysis_frames_inclusive':'10-49','integrations_expected':1200,'integrations_completed':integrations,'model_phase_rows':len(metrics),'decoded_join_rows':len(joined),'source_contribution_rows':len(source_df),'decoded_join_exact':True,
      'source_ON':json.dumps(ON_SRC),'targets_ON':json.dumps(ON_TGT),'source_OFF':json.dumps(OFF_SRC),'targets_OFF':json.dumps(OFF_TGT),
      'aggregation':'sum signed source edge currents per target cell; time mean frames 10-49; mean target cells within subtype; equal-weight mean of four subtypes',
      'ON_A':'(R_ON_original-R_ON_mirror)/2','OFF_A':'(R_OFF_original-R_OFF_mirror)/2','I_ONOFF':'ON_A-OFF_A; no post-hoc sign flip; no fitted source/spatial weights',
      'primary_stats':'P1 model-level phase-mean I vs zero; P2 Spearman model-level mean I vs A_flow; 10000 whole-model bootstrap draws; 20000 permutations; Holm over P1/P2; P3 descriptive only',
      'random_seed':SEED,'input_sha256':json.dumps(inputs,sort_keys=True),'stimulus_image_and_render_hashes':json.dumps(stimuli,sort_keys=True),'rendered_hash_manifest_match_count':sum(x['rendered_hash_matches_manifest'] for x in stimuli.values()),
      'rendered_hash_note':'Image hashes match canonical manifest. Rendered hashes record both observed BoxEye bytes and manifest values; manifest byte serialization provenance is undocumented and current values do not match.',
      'model_checkpoint_qa_sha256':sha(work/'official50_checkpoint_qa.csv'),'current_shape_qa_sha256':sha(work/'official50_current_shape_qa.csv'),
      'wall_seconds_total_including_QA':total_elapsed,'peak_rss_bytes_process':peak,'official50_work_disk_bytes':sum(p.stat().st_size for p in work.rglob('*') if p.is_file()),
      'output_sha256_excluding_manifest':json.dumps({n:sha(out/n) for n in ('on_off_model_phase_metrics.csv','on_off_vs_decoded_flow.csv','on_off_source_contributions.csv')},sort_keys=True),
      'checks':json.dumps({'models_50':True,'phases_12':True,'conditions_2':True,'integrations_1200':True,'metrics_600':True,'source_rows_38400':True,'finite':True,'unique':True,'all_eight_targets':True,'exact_decoded_join_600':True,'analysis_frames_10_49':True,'dt_0.02':True,'fade_1s':True,'no_sign_flip':True,'no_fitted_weights':True,'no_cohort_or_phase_selection':True},sort_keys=True)}
    pd.DataFrame([{'key':k,'value':v if isinstance(v,str) else json.dumps(v,sort_keys=True)} for k,v in manifest.items()]).to_csv(out/'on_off_reproducibility_manifest.csv',index=False)
    print(json.dumps({'INFERENCE_PASS':True,'integrations':integrations,'metrics':len(metrics),'joined':len(joined),'source_rows':len(source_df),'elapsed_seconds':total_elapsed,'peak_rss_bytes':peak,'manifest':str(out/'on_off_reproducibility_manifest.csv')},indent=2),flush=True)
if __name__=='__main__': main()
