#!/usr/bin/env python3
"""Prespecified model-level D1 statistics and figures from frozen official-50 tables."""
from pathlib import Path
import hashlib,json,sys,time
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

SEED=20261006; NBOOT=10000; NPERM=20000
ROOT=next((p for p in (Path.cwd().resolve(),*Path.cwd().resolve().parents) if (p/'data/precomputed/supplementary/biological_constraints/D1/on_off_model_phase_metrics.csv').is_file()),None)
if ROOT is None: raise FileNotFoundError('Complete official-50 inference first.')
OUT=ROOT/'data/precomputed/supplementary/biological_constraints/D1'; FIG=ROOT/'figures/supplementary/biological_constraints_D1'; FIG.mkdir(parents=True,exist_ok=True)
rng=np.random.default_rng(SEED)
def holm(ps):
    p=np.asarray(ps,float); ix=np.argsort(p); out=np.empty_like(p); run=0.
    for rank,i in enumerate(ix):
        run=max(run,(len(p)-rank)*p[i]); out[i]=min(1.,run)
    return out
def ci_mean(x, n=NBOOT):
    x=np.asarray(x,float); draws=rng.choice(x,(n,len(x)),replace=True).mean(axis=1)
    return np.quantile(draws,[.025,.975])
def ci_rho(x,y,n=NBOOT):
    x=np.asarray(x,float); y=np.asarray(y,float); vals=[]
    for _ in range(n):
        ix=rng.integers(0,len(x),len(x))
        if np.unique(x[ix]).size<2 or np.unique(y[ix]).size<2: continue
        z=stats.spearmanr(x[ix],y[ix]).statistic
        if np.isfinite(z): vals.append(z)
    if len(vals)<n*.9: raise RuntimeError(f'Too many degenerate bootstrap samples: {len(vals)}/{n}')
    return np.quantile(vals,[.025,.975]),len(vals)
def perm_rho(x,y,n=NPERM):
    obs=stats.spearmanr(x,y).statistic
    null=np.empty(n)
    for i in range(n): null[i]=stats.spearmanr(x,rng.permutation(y)).statistic
    p=(1+np.sum(np.abs(null)>=abs(obs)))/(n+1)
    return float(obs),float(p)
def spearman_row(name,x,y,n=NBOOT):
    x=np.asarray(x,float); y=np.asarray(y,float)
    rho,p=perm_rho(x,y); ci,nvalid=ci_rho(x,y,n)
    return {'test':name,'n_models':len(x),'spearman_rho':rho,'ci95_low':ci[0],'ci95_high':ci[1],
            'bootstrap_valid':nvalid,'permutation_p_raw':p}
def write_fig(stem):
    plt.savefig(FIG/f'{stem}.png',dpi=300,bbox_inches='tight',facecolor='white')
    plt.savefig(FIG/f'{stem}.pdf',bbox_inches='tight',facecolor='white')
    plt.close()

started=time.perf_counter()
metrics=pd.read_csv(OUT/'on_off_model_phase_metrics.csv')
joined=pd.read_csv(OUT/'on_off_vs_decoded_flow.csv')
sources=pd.read_csv(OUT/'on_off_source_contributions.csv')
assert len(metrics)==600 and len(joined)==600 and metrics.model_short_id.nunique()==50
assert metrics.groupby('model_short_id').phase_index.nunique().eq(12).all()
if not np.isfinite(metrics.select_dtypes(include=[np.number]).to_numpy()).all(): raise ValueError('Nonfinite metrics')
model=joined.groupby('model_short_id',sort=True).agg(
    ON_A_mean=('ON_A','mean'),OFF_A_mean=('OFF_A','mean'),I_ONOFF_mean=('I_ONOFF','mean'),
    I_ONOFF_phase_sd=('I_ONOFF','std'),A_flow_mean=('A_flow','mean'),A_flow_phase_sd=('A_flow','std'),
    actual_reversal_fraction=('actual_reversal','mean'),
    ON_A_phase_sd=('ON_A','std'),OFF_A_phase_sd=('OFF_A','std')).reset_index()
model['abs_I_ONOFF_mean']=model.I_ONOFF_mean.abs(); model['abs_A_flow_mean']=model.A_flow_mean.abs()
if len(model)!=50: raise AssertionError('Analysis unit must be 50 models.')
model.to_csv(OUT/'on_off_model_summary.csv',index=False)

# Primary P1: whole-model phase-mean I_ONOFF versus zero.
x=model.I_ONOFF_mean.to_numpy(float); y=model.A_flow_mean.to_numpy(float)
obs=float(x.mean()); bci=ci_mean(x)
null=rng.choice(np.array([-1.,1.]),size=(NPERM,len(x)))@x/len(x)
p1=(1+np.sum(np.abs(null)>=abs(obs)))/(NPERM+1)
p1row={'test':'P1_mean_model_phase_mean_I_ONOFF_vs_zero','n_models':len(x),'mean':obs,'median':float(np.median(x)),
       'sd_between_models':float(np.std(x,ddof=1)),'bootstrap_ci95_low':float(bci[0]),'bootstrap_ci95_high':float(bci[1]),
       'sign_flip_permutation_p_raw':float(p1),'n_permutations':NPERM,'bootstrap_draws':NBOOT}
# Primary P2: model-level Spearman; whole-model paired bootstrap/permutation.
p2=spearman_row('P2_I_ONOFF_mean_vs_A_flow_mean',x,y)
primary=pd.DataFrame([
 {'test':p1row['test'],'n_models':50,'estimate':p1row['mean'],'median':p1row['median'],'sd_between_models':p1row['sd_between_models'],
  'ci95_low':p1row['bootstrap_ci95_low'],'ci95_high':p1row['bootstrap_ci95_high'],'p_raw':p1row['sign_flip_permutation_p_raw'],'p_holm':np.nan,'status':'CONFIRMATORY'},
 {'test':p2['test'],'n_models':50,'estimate':p2['spearman_rho'],'median':np.nan,'sd_between_models':np.nan,
  'ci95_low':p2['ci95_low'],'ci95_high':p2['ci95_high'],'p_raw':p2['permutation_p_raw'],'p_holm':np.nan,'status':'CONFIRMATORY'}])
primary['p_holm']=holm(primary.p_raw.to_numpy())
primary['bootstrap_draws']=NBOOT; primary['permutations']=NPERM
primary['statistic_definition']=['mean across models; sign-flip whole model values','Spearman across 50 model means; permute y across models']
# P3 remains descriptive; no sign mapping for selected synaptic currents was prespecified.
concord=(np.sign(x)==np.sign(y)) & (x!=0) & (y!=0); valid=(x!=0)&(y!=0); k=int(concord.sum()); n=int(valid.sum())
lo=stats.beta.ppf(.025,k,n-k+1) if k>0 else 0.; hi=stats.beta.ppf(.975,k+1,n-k) if k<n else 1.
p3=pd.DataFrame([{'test':'P3_sign_concordance_descriptive_only','n_concordant':k,'n_discordant':n-k,'n_zero_excluded':50-n,
 'proportion_concordant':k/n if n else np.nan,'ci95_low_exact_binomial':lo,'ci95_high_exact_binomial':hi,
 'p_raw':np.nan,'p_holm':np.nan,'status':'NOT_CONFIRMATORY','reason':'No prespecified biological/theoretical mapping from signed source-current imbalance to decoded-flow sign.'}])
primary=pd.concat([primary,p3],ignore_index=True); primary.to_csv(OUT/'on_off_primary_stats.csv',index=False)

# Secondary family: four prespecified associations, corrected within family.
sec=[]
for name,a,b in [('ON_A_mean_vs_A_flow_mean',model.ON_A_mean,model.A_flow_mean),
                 ('OFF_A_mean_vs_A_flow_mean',model.OFF_A_mean,model.A_flow_mean),
                 ('abs_I_ONOFF_mean_vs_abs_A_flow_mean',model.abs_I_ONOFF_mean,model.abs_A_flow_mean),
                 ('I_ONOFF_phase_sd_vs_A_flow_phase_sd',model.I_ONOFF_phase_sd,model.A_flow_phase_sd)]: sec.append(spearman_row(name,a,b))
secdf=pd.DataFrame(sec); secdf['permutation_p_holm']=holm(secdf.permutation_p_raw.to_numpy()); secdf['family']='secondary; Holm across 4 tests'; secdf.to_csv(OUT/'on_off_secondary_stats.csv',index=False)

# Exploratory source/stage: average subtype-level source response equally within stage, then phase mean per model.
raw=sources.loc[sources.source_type.ne('__selected_sources_sum__')].copy()
raw['stage']=raw.apply(lambda r: r['source_type'] if r['source_type'] not in ('Mi1','Tm3','Tm1','Tm2','Tm4','Tm9') else r['source_type'],axis=1)
stage_source=raw.groupby(['model_short_id','phase_index','phase_deg','condition','pathway','source_type'],as_index=False).mean(numeric_only=True)
stage_total=sources.loc[sources.source_type.eq('__selected_sources_sum__')].copy()
stage_total['source_type']=stage_total.pathway.map({'ON':'T4_stage','OFF':'T5_stage'})
stage_all=pd.concat([stage_source,stage_total],ignore_index=True)
piv=stage_all.pivot_table(index=['model_short_id','phase_index','phase_deg','pathway','source_type'],columns='condition',values='mean',aggfunc='first').reset_index()
piv['A_source']=(piv['original']-piv['mirror'])/2
stage_model=piv.groupby(['model_short_id','source_type'],as_index=False).A_source.mean()
stage_w=stage_model.pivot(index='model_short_id',columns='source_type',values='A_source').reset_index().merge(model[['model_short_id','A_flow_mean']],on='model_short_id',validate='one_to_one')
labels=['Mi1','Tm3','T4_stage','Tm1','Tm2','Tm4','Tm9','T5_stage']; stage_rows=[]
for label in labels:
    row=spearman_row(label+'_A_vs_A_flow_mean',stage_w[label],stage_w.A_flow_mean)
    row['stage']=label; row['family']='exploratory; Holm across 8 tests'; stage_rows.append(row)
stagedf=pd.DataFrame(stage_rows); stagedf['permutation_p_holm']=holm(stagedf.permutation_p_raw.to_numpy()); stagedf.to_csv(OUT/'on_off_stage_exploratory.csv',index=False)

# Figures: model is the unit in every plot.
plt.figure(figsize=(7,5));
for _,r in model.iterrows(): plt.plot([0,1],[r.ON_A_mean,r.OFF_A_mean],color='#9aa5b1',alpha=.55,lw=.7)
plt.scatter(np.zeros(50),model.ON_A_mean,s=18,color='#247ba0',label='ON_A',zorder=3); plt.scatter(np.ones(50),model.OFF_A_mean,s=18,color='#f28e2b',label='OFF_A',zorder=3)
plt.axhline(0,color='black',lw=.8,ls=':'); plt.xticks([0,1],['ON_A','OFF_A']); plt.ylabel('Mean signed source-current response'); plt.title('D1 | Phase-averaged ON and OFF responses (N=50 models)'); plt.legend(frameon=False); write_fig('fig_D1_on_off_paired')
plt.figure(figsize=(6.4,5.2)); plt.scatter(model.I_ONOFF_mean,model.A_flow_mean,s=32,alpha=.8,color='#2878b5',edgecolor='white',linewidth=.4)
plt.axhline(0,color='black',lw=.7,ls=':'); plt.axvline(0,color='black',lw=.7,ls=':')
plt.xlabel('Phase-mean ON_A − OFF_A'); plt.ylabel('Phase-mean decoded A_flow'); plt.title('D1 | Model-level pathway imbalance and decoded flow')
plt.text(.03,.97,f"N = 50\nSpearman ρ = {p2['spearman_rho']:.3f}\n95% bootstrap CI [{p2['ci95_low']:.3f}, {p2['ci95_high']:.3f}]",transform=plt.gca().transAxes,va='top',ha='left',bbox={'facecolor':'white','edgecolor':'none','alpha':.85})
write_fig('fig_D2_imbalance_vs_decoded_A')
plt.figure(figsize=(5.5,4.5)); plt.bar(['Concordant','Discordant','Zero excluded'],[k,n-k,50-n],color=['#59a14f','#e15759','#bab0ab']);
for i,v in enumerate([k,n-k,50-n]): plt.text(i,v+.5,str(v),ha='center')
plt.ylabel('Models'); plt.ylim(0,55); plt.title('D1 | Sign concordance (descriptive; not confirmatory)'); write_fig('fig_D3_sign_concordance')
# Across-phase ensemble mean with whole-model bootstrap intervals.
phase_rows=[]
for phase,g in joined.groupby('phase_index',sort=True):
    for var in ('I_ONOFF','A_flow'):
        z=g[var].to_numpy(float); bs=np.array([rng.choice(z,len(z),replace=True).mean() for _ in range(3000)])
        phase_rows.append({'phase_index':int(phase),'phase_deg':float(g.phase_deg.iloc[0]),'measure':var,'mean':float(z.mean()),'ci95_low':float(np.quantile(bs,.025)),'ci95_high':float(np.quantile(bs,.975))})
phase_df=pd.DataFrame(phase_rows); phase_df.to_csv(OUT/'on_off_phase_summary.csv',index=False)
fig,ax=plt.subplots(figsize=(8,4.8));
for var,color,label in [('I_ONOFF','#4e79a7','I_ONOFF'),('A_flow','#e15759','Decoded A_flow')]:
    g=phase_df.loc[phase_df.measure.eq(var)]; ax.errorbar(g.phase_deg,g['mean'],yerr=[g['mean']-g.ci95_low,g.ci95_high-g['mean']],marker='o',capsize=2,lw=1.2,color=color,label=label)
ax.axhline(0,color='black',lw=.7,ls=':'); ax.set_xlabel('Stimulus phase (degrees)'); ax.set_ylabel('Ensemble mean (95% model bootstrap CI)'); ax.set_title('D1 | Phase dependence across the 50-model cohort'); ax.legend(frameon=False); write_fig('fig_D4_phase_dependence')
plt.figure(figsize=(8,5)); ypos=np.arange(len(stagedf)); loerr=stagedf.spearman_rho-stagedf.ci95_low; hierr=stagedf.ci95_high-stagedf.spearman_rho
plt.errorbar(stagedf.spearman_rho,ypos,xerr=[loerr,hierr],fmt='o',color='#6f4e7c',capsize=3); plt.axvline(0,color='black',lw=.7,ls=':'); plt.yticks(ypos,stagedf.stage); plt.xlabel('Spearman ρ with model-mean decoded A (95% model bootstrap CI)'); plt.title('D1 | Source/stage associations (exploratory; Holm-adjusted family)'); plt.tight_layout(); write_fig('fig_D5_exploratory_stages')

# Update the reproducibility manifest with post-inference outputs, statistics and figures.
manifest_path=OUT/'on_off_reproducibility_manifest.csv'; man=pd.read_csv(manifest_path).set_index('key').value.to_dict()
files=[OUT/n for n in ('on_off_model_phase_metrics.csv','on_off_model_summary.csv','on_off_vs_decoded_flow.csv','on_off_primary_stats.csv','on_off_secondary_stats.csv','on_off_stage_exploratory.csv','on_off_source_contributions.csv','on_off_phase_summary.csv')]+sorted(FIG.glob('fig_D*.png'))+sorted(FIG.glob('fig_D*.pdf'))
hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
man['statistical_analysis_seconds']=str(time.perf_counter()-started); man['post_analysis_output_sha256']=json.dumps(hashes,sort_keys=True)
man['statistics_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
# Preserve key-value manifest format.
pd.DataFrame([{'key':k,'value':v} for k,v in man.items()]).to_csv(manifest_path,index=False)
print(json.dumps({'ANALYSIS_PASS':True,'N_models':50,'P1':p1row,'P2':p2,'P3':p3.iloc[0].to_dict(),
 'secondary':secdf.to_dict('records'),'stages':stagedf[['stage','spearman_rho','ci95_low','ci95_high','permutation_p_raw','permutation_p_holm']].to_dict('records'),
 'primary_holm':primary[['test','p_raw','p_holm','status']].to_dict('records'),'figures':[p.name for p in sorted(FIG.glob('fig_D*'))]},indent=2,default=str))
