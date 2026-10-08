#!/usr/bin/env python3
"""Check adopted numerical invariants and canonical reproduction manifests.

Read-only: never runs inference or rewrites historical audit artifacts.
"""
from pathlib import Path
import json, hashlib
import pandas as pd
import numpy as np
r=Path(__file__).resolve().parents[1];checks=[];numbers=[]
def check(name,value):
    checks.append(dict(check=name,passed=bool(value)))
    if not value: raise AssertionError(name)
def read(path):return pd.read_csv(r/path)
d=read('data/precomputed/fig01/fig1_midd_per_model.csv');groups=['3MIDD_048','3MIDD_207','4MIDD_085_170'];check('canonical counts21/18/12',list(d.groupby('group',sort=False).actual_reversal.sum())==[21,18,12]);check('canonical means agree',np.allclose([d[d.group==g].A.mean() for g in groups],[-.009345612,.008258559,.003209043],atol=1e-9))
p='data/precomputed/fig03/composition_bootstrap_summary.csv';b=read(p)
for source,metric,value,tolerance in [('in_sample','r2',.9295,.00005),('LOLO','cv_r2',.9024,.00005),('LOLO','sign_accuracy',.949,.0005),('in_sample','sign_agreement',.965,.00001)]:
 row=b[(b.source==source)&(b.metric==metric)].iloc[0];check(source+' '+metric+' text rounding',abs(row['mean']-value)<tolerance);numbers.append(dict(metric=source+' '+metric,value=float(row['mean']),ci_low=float(row.ci_low),ci_high=float(row.ci_high),source_file=p))
e=read('data/precomputed/fig04/geometry_geometry_contrasts.csv');q=e[(e.group==groups[0])&(e['transform']=='LR')];check('reflectionA10 S0',q.A_raw_sign_flipped.sum()==10 and q.S_raw_sign_flipped.sum()==0)
l=read('data/precomputed/fig05/learning_threshold_summary.csv');q=l[(l.support=='common_100k')&(l.definition=='first')];check('threshold sensitivity all14 below.005',q[q.threshold.isin([.0005,.001,.002])].n_crossed.eq(14).all());check('higherthreshold12/13/11',[int(q[(q.group==g)&(q.threshold==.005)].n_crossed.iloc[0]) for g in groups]==[12,13,11])
n=read('data/precomputed/fig05/learning_normalized_metrics.csv');end=n[n.optimizer_step==100000];check('seed signs12neg13pos11pos',[int((end[end.group==g].A*(1 if i else -1)>0).sum()) for i,g in enumerate(groups)]==[12,13,11]);check('normalized endpoints match',np.allclose([end[end.group==g].abs_A_norm.median() for g in groups],[.03832,.03493,.02449],atol=.000005))
w=read('data/precomputed/supplementary/figS3/duration_window_metrics.csv');q=w[(w.group==groups[0])&(w.fade_seconds==1)&(w.jitter_pixels==0)&(w.window=='late')].sort_values('model_id');check('late selected3 values',np.allclose(q.A,[-.02707,.009925,-.001720],atol=.000005));check('lateN3notN50',len(q)==3 and abs(q.A).gt(.0005).all())

b=read('data/precomputed/fig02/matched_control_bootstrap_summary.csv')
for g,v in [('3MIDD_048',-.003473),('4MIDD_085_170',-.007615)]:
 check('matched controls '+g,np.isclose(b[(b.group==g)&(b.metric=='D_canonical_vs_random')]['mean'].iloc[0],v,atol=.0000005))
b=read('data/precomputed/fig03/composition_bootstrap_summary.csv')
for metric,v in [('slope',.829171),('intercept',.007369)]:
 check('affine '+metric,np.isclose(b[b.metric==metric]['mean'].iloc[0],v,atol=.0000005))
b=read('data/precomputed/fig03/composition_control_summary.csv')
check('structured null delta',np.isclose(b[b.family=='smooth_potential_ls0.25'].delta_r2.iloc[0],.885534,atol=.0000005))
b=read('data/precomputed/supplementary/figS1/multifield_sign_correspondence.csv')
check('multi physical signs10',b[b.renderer=='multi'].CW_negative.sum()==10 and b[b.renderer=='multi'].CCW_positive.sum()==10)
check('native physical signs10',b[b.renderer=='single'].CW_positive.sum()==10 and b[b.renderer=='single'].CCW_negative.sum()==10)
b=read('data/precomputed/fig04/phase_model_phase_sensitivity.csv')
check('phase variance ratio',np.isclose(b[b.group==groups[0]].ratio.median(),.2636519,atol=.0000001))
b=read('data/precomputed/fig04/geometry_geometry_contrasts.csv')
b=b[(b.group==groups[0])&b['transform'].str.startswith(('x_','y_'))].groupby('model_id')[['normalized_error_A','normalized_error_S']].mean()
check('translation errorA',np.isclose(b.normalized_error_A.mean(),.045827,atol=.000001))
check('translation errorS',np.isclose(b.normalized_error_S.mean(),.0233479,atol=.0000001))
b=read('data/precomputed/fig06/replacement_bootstrap.csv')
for intervention,metric,v in [('T4a','D_primary',.2423115),('T4','propeller_loss',.5968236),('T4_T5','propeller_loss',.7099271),('T4_T5','edge_loss',.816967)]:
 row=b[(b.intervention==intervention)&(b.metric==metric)]
 check('intervention '+intervention+' '+metric,len(row)==1 and np.isclose(row['mean'].iloc[0],v,atol=.000001))
b=read('data/precomputed/supplementary/figS6/sector_model_count_sensitivity.csv')
check('canonical sector not maximum',not b.canonical_is_max.any())
b=read('data/precomputed/supplementary/figS6/sector_bootstrap.csv')
row=b[(b.group==groups[0])&(b.phase_fraction==0)&(b.metric=='D')]
check('sector contrast',np.isclose(row['mean'].iloc[0],-.0299950304,atol=1e-9))
b=read('data/precomputed/fig06/crossswap_decomposition.csv')
for group,metric,effect,v in [(groups[0],'A','network',.1045416),(groups[0],'A','decoder',.0508304),(groups[0],'A','residual',.8446280),('motion_grating','Q_motion','residual',.8342305)]:
 row=b[(b.group==group)&(b.window=='stable')&(b.metric==metric)&(b.effect==effect)]
 check('centered SS '+group+' '+effect,len(row)==1 and np.isclose(row.fraction.iloc[0],v,atol=.0000001))
spec=json.loads((r/'src/flyvis_midd/figure_inputs.json').read_text())
check('six main ten supplement routes',set(spec)=={*[str(i) for i in range(1,7)],*[f'S{i}' for i in range(1,11)]})
for key,d in spec.items():
 m=json.loads((r/d['output_dir']/(d['prefix']+'_integration_manifest.json')).read_text())
 check(key+' canonical notebook',m['canonical_notebook']==d['notebook'])
 check(key+' artifact only',m['inference_performed'] is False and m['revision_notebook_execution_required'] is False)
 for kind in ['inputs_sha256','source_sha256','outputs_sha256']:
  for p,h in m[kind].items():check(key+' '+kind+' '+p,hashlib.sha256((r/p).read_bytes()).hexdigest()==h)
 for p,d in m['summaries'].items():check(key+' summary '+p,hashlib.sha256((r/p).read_bytes()).hexdigest()==d['sha256'])
 metadata=json.loads((r/spec[key]['output_dir']/'metadata.json').read_text())
 check(key+' metadata notebook',metadata['canonical_reproduction']['notebook']==spec[key]['notebook'])
print(json.dumps({'checks':checks,'numbers':numbers,'all_passed':all(x['passed'] for x in checks)},indent=2))
