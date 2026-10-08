#!/usr/bin/env python3
"""Regenerate adopted figures from immutable cached inputs; never loads a model.

CSV/NPZ values and original sign conventions are retained. Every consumed input
and generated output is hashed. Figure 1 vector snapshots are explicitly cached
raster displays because the complete canonical representative flow is absent.
"""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
ROOT=Path(__file__).resolve().parents[2]
GROUPS=['3MIDD_048','3MIDD_207','4MIDD_085_170']
LABELS=['3MIDD C=48','3MIDD C=207','4MIDD 85/170']
COLORS=['#0072B2','#D55E00','#009E73']
INPUTS={};CHECKS=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def register(path):
 p=ROOT/path;INPUTS[str(p.relative_to(ROOT))]=sha(p);return p
def csv(path):return pd.read_csv(register(path))
def check(name,p):
 CHECKS.append(dict(check=name,passed=bool(p)))
 if not p:raise AssertionError(name)
def panel(ax,letter,title,x=None,y=None):
 ax.set_title(title,loc='left',fontsize=12,pad=10)
 ax.text(0,1.14,letter,transform=ax.transAxes,weight='bold',fontsize=13)
 if x:ax.set_xlabel(x)
 if y:ax.set_ylabel(y)
 ax.tick_params(labelsize=11)
def zero(ax):ax.axhline(0,color='.65',lw=.7,zorder=0)
def layout(n=6):
 fig,axs=plt.subplots((n+1)//2,2,figsize=(10.6,3.0*((n+1)//2)),layout='constrained');return fig,np.array(axs).ravel()
def points(ax,values,labels,colors=None,paired=False):
 arrays=[np.asarray(v,float) for v in values];j=.09*np.sin(np.arange(max(len(v) for v in arrays))*2.4)
 if paired and len(set(map(len,arrays)))==1:
  for k in range(len(arrays[0])):ax.plot(np.arange(len(arrays))+j[k],[v[k] for v in arrays],color='.8',lw=.6,zorder=1)
 for i,v in enumerate(arrays):
  c=(colors or COLORS)[i%len(colors or COLORS)];ax.scatter(i+j[:len(v)],v,s=15,color=c,alpha=.65,zorder=2);ax.plot([i-.13,i+.13],[np.mean(v)]*2,color='black',lw=1.5,zorder=3)
 ax.set_xticks(range(len(labels)),labels);zero(ax)
def traj(ax,df,x,y,unit,color,summary='median',label=None):
 for _,q in df.groupby(unit):q=q.sort_values(x);ax.plot(q[x],q[y],color=color,alpha=.19,lw=.7)
 q=df.groupby(x)[y].agg(summary);ax.plot(q.index,q.values,color=color,lw=2,label=label)
def save(fig,key):
 d=ROOT/('figures/fig'+key.zfill(2) if not key.startswith('S') else 'figures/supplementary/fig'+key)
 d.mkdir(parents=True,exist_ok=True);prefix='fig'+key.zfill(2) if not key.startswith('S') else 'fig'+key
 fig.savefig(d/(prefix+'_main.pdf'),bbox_inches='tight');fig.savefig(d/(prefix+'_main.png'),dpi=220,bbox_inches='tight')
 pd.DataFrame(CHECKS).to_csv(d/(prefix+'_adoption_QA.csv'),index=False)
 outputs=[d/(prefix+'_main.pdf'),d/(prefix+'_main.png'),d/(prefix+'_adoption_QA.csv')]
 (d/(prefix+'_adoption_manifest.json')).write_text(json.dumps({'source_script':'src/flyvis_midd/figure_rendering.py','source_sha256':sha(Path(__file__)),'figure':key,'inference_performed':False,'inputs_sha256':INPUTS,'outputs_sha256':{str(p.relative_to(ROOT)):sha(p) for p in outputs},'unit':'model/seed; repeated conditions within unit','notes':'Fig1B is a retained raster snapshot; all other scientific plots use stored numerical inputs.' if key=='1' else ''},indent=2)+'\n')
 print('Saved',key,'inputs',len(INPUTS),'checks',len(CHECKS));return fig

def canonical():
 d=csv('data/precomputed/fig01/fig1_midd_per_model.csv');check('canonical 50 models per group',all(d[d.group==g].model_id.nunique()==50 for g in GROUPS))
 check('canonical reversal 21/18/12',[int(d[d.group==g].actual_reversal.sum()) for g in GROUPS]==[21,18,12]);check('A/S definitions',np.allclose(d.A,(d.T_original-d.T_mirror)/2) and np.allclose(d.S,(d.T_original+d.T_mirror)/2));return d


def fig1():
 d=canonical();q=d[d.group==GROUPS[0]]
 fig=plt.figure(figsize=(11,8),layout='constrained');gs=fig.add_gridspec(2,3,height_ratios=[1.4,1])
 cache=Image.open(register('figures/fig01/source/retained_representative_fields.png'))
 a=fig.add_subplot(gs[0,0]);a.imshow(Image.open(register('figures/fig01/fig1A_midd_stimuli.png')));a.axis('off');panel(a,'A','MIDD stimuli')
 a=fig.add_subplot(gs[0,1:]);a.imshow(cache);a.axis('off');panel(a,'B','Representative vector snapshots (frame 25)')
 a=fig.add_subplot(gs[1,0]);a.bar(['Reversal','Non-reversal'],[21,29],color=[COLORS[0],COLORS[1]]);a.set_ylim(0,50);panel(a,'C','Raw phenotype, N=50',y='Models');a.text(0,22,'21/50',ha='center');a.text(1,30,'29/50',ha='center')
 a=fig.add_subplot(gs[1,1]);b=q.actual_reversal;a.scatter(q.T_original[b],q.T_mirror[b],s=20,color=COLORS[0],label='Reversal');a.scatter(q.T_original[~b],q.T_mirror[~b],s=20,color=COLORS[1],label='Non-reversal');zero(a);a.axvline(0,color='.65',lw=.7);panel(a,'D','Original/Mirror pairs',x='T original (CW +)',y='T mirror (CW +)');a.legend(fontsize=8,frameon=False)
 a=fig.add_subplot(gs[1,2]);points(a,[q.T_original,q.T_mirror,q.T_negative],['Original','Mirror','Negative'],paired=True);panel(a,'E','Nonzero balanced control',y='Tangential flow T (CW +)');return fig

def fig2():
 d=canonical();l=csv('data/precomputed/fig02/fig2B_luminance_A_per_model.csv');f,a=layout();im=np.load(register('data/stimuli/fig1B_boxeye_input/midd_source_images_uint8.npz'))
 sys.path.insert(0,str(ROOT/'src'))
 from flyvis_midd.stimuli import build_3midd_sequences, render_sector_image
 levels=sorted(l.C.unique());canvas=Image.new('RGB',(900,560),'white');draw=ImageDraw.Draw(canvas)
 from PIL import ImageFont
 font=ImageFont.load_default(size=18)
 for idx,level in enumerate(levels):
  image,_=render_sector_image(build_3midd_sequences(int(level))['original'],image_size=382,inner_diameter=55,outer_diameter=382)
  x=(idx%5)*180;y=(idx//5)*185;canvas.paste(Image.fromarray(image).convert('RGB').resize((160,160)),(x,y+23));draw.text((x+45,y),f'C={level}',font=font,fill='black')
 a[0].imshow(canvas);a[0].axis('off');panel(a[0],'A','Luminance structure (15 levels)')
 traj(a[1],l,'C','A','model_id',COLORS[0],summary='mean');panel(a[1],'B','Official50 dose-response',x='Intermediate luminance C',y='A (CW-positive basis)');zero(a[1])
 for i,g in enumerate(GROUPS[:2]):
  q=d[d.group==g];a[2].scatter(q.T_original,q.T_mirror,s=15,color=COLORS[i],label=LABELS[i],alpha=.6)
 zero(a[2]);a[2].axvline(0,color='.6',lw=.7);panel(a[2],'C','Original/Mirror, N=50',x='T original',y='T mirror');a[2].legend(fontsize=8,frameon=False)
 q=d[d.group==GROUPS[2]];a[3].scatter(q.T_original,q.T_mirror,color=COLORS[2],s=16);zero(a[3]);a[3].axvline(0,color='.6',lw=.7);panel(a[3],'D','4MIDD: 12/50 reversal',x='T original',y='T mirror')
 points(a[4],[d[d.group==g].T_negative for g in GROUPS],LABELS);panel(a[4],'E','Balanced negative, official50',y='T negative (CW +)')
 c=csv('data/precomputed/fig02/official50_shuffle_model_metrics.csv')
 for i,g in enumerate(GROUPS):
  q=c[c.group==g].sort_values('model_id');off=i*3
  for _,row in q.iterrows():a[5].plot([off,off+1],[row.canonical_reflection_abs_A,row.shuffle_mean_abs_A],color=COLORS[i],alpha=.5,lw=.8)
  a[5].scatter([off]*len(q),q.canonical_reflection_abs_A,color=COLORS[i],s=12);a[5].scatter([off+1]*len(q),q.shuffle_mean_abs_A,color=COLORS[i],marker='s',s=12)
 a[5].set_xticks([.5,3.5,6.5],LABELS);panel(a[5],'F','Same image/LR operator, official50',y='Magnitude (decoded-flow units)');a[5].text(.02,.98,'Circles: canonical image/LR\nSquares: mean of ten shuffle pairs',transform=a[5].transAxes,va='top',fontsize=8);return f

def fig3():
 m=csv('data/precomputed/fig03/composition_model_metrics.csv');v=csv('data/precomputed/fig03/composition_cv_metrics.csv');n=csv('data/precomputed/fig03/composition_null_comparison.csv');c=csv('data/precomputed/fig03/fig3_3MIDD_components_per_model.csv');p=csv('data/precomputed/fig03/fig3_canonical_pair_predictions.csv');f,a=layout()
 a[0].axis('off');a[0].text(.5,.58,r'$A_{4,pred}=A_3(C_1)-A_3(C_2)$',ha='center',fontsize=17);a[0].text(.5,.27,'Centered eight fields; N=10\n21 repeated pairs per model',ha='center',fontsize=11);panel(a[0],'A','Component predictor')
 traj(a[1],c,'c','a3','model_id',COLORS[0],summary='mean');zero(a[1]);panel(a[1],'B','Calibrated: positive CCW / negative CW',x='Luminance C',y='A3 (implemented scalar)')
 a[2].scatter(p.a4_predicted,p.a4_observed,s=10,alpha=.35,color=COLORS[0]);lo=min(p.a4_predicted.min(),p.a4_observed.min());hi=max(p.a4_predicted.max(),p.a4_observed.max());xx=np.array([lo,hi]);a[2].plot(xx,xx,'--',color='.5',label='Identity');a[2].plot(xx,m.intercept.mean()+m.slope.mean()*xx,color=COLORS[1],label='Mean affine fit');panel(a[2],'C','In-sample scalar relation',x='Component prediction (CCW +)',y='Observed A4 (CCW +)');a[2].legend(fontsize=8,frameon=False)
 cv=v[v.split=='LOLO'].sort_values('model_id');points(a[3],[m.sort_values('model_id').r2,cv.cv_r2],['Fitted','LOLO'],paired=True);panel(a[3],'D','Model-wise validation',y='R²');a[3].text(.02,.03,'LOLO mean 0.9024; 95% CI 0.8588–0.9399',transform=a[3].transAxes,fontsize=8)
 q=csv('data/precomputed/fig03/comparator_model_metrics.csv');points(a[4],[q[q.predictor==p].sort_values('model_id').cv_r2 for p in ['P0','P1','P2','P3']],['Own','Ensemble','Donors','Delta C'],paired=True);panel(a[4],'E','Target-model information improves prediction',y='Conditional LOLO R²')
 a[5].scatter(p.a4_predicted,p.normalized_residual,color=COLORS[2],s=10,alpha=.5);zero(a[5]);panel(a[5],'F','Departures from identity',x='Component prediction (CCW +)',y='Residual / A3 range');check('CV canonical mean',abs(cv.cv_r2.mean()-.902378)<1e-6);return f

def fig4():
 p=csv('data/precomputed/fig04/official50_phase_pairs.csv');pm=csv('data/precomputed/fig04/official50_phase_model_metrics.csv');ic=csv('data/precomputed/fig04/official50_phase_reliability.csv');geom=csv('data/precomputed/fig04/receptor_geometry.csv');roi=csv('data/precomputed/fig04/radial_radial_model_phase_metrics.csv');e=csv('data/precomputed/fig04/geometry_geometry_contrasts.csv');f,a=layout()
 traj(a[0],p,'phase_index','A','model_id',COLORS[0],summary='mean')
 n_changed=int(ic.sign_changing_models.iloc[0]);n_models=int(ic.n_models.iloc[0])
 panel(a[0],'A','C48: twelve phases, official50',x='Phase index (one 15° period)',y='A (CW +)');zero(a[0])
 a[0].set_ylim(top=max(.06,float(p.A.max())+.025));a[0].text(.98,.98,f'{n_changed}/{n_models} models changed sign',transform=a[0].transAxes,ha='right',va='top',fontsize=8,bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=2))
 old=csv('data/precomputed/fig04/phase_reliability_icc_bootstrap.csv')
 value=float(ic.icc_absolute.iloc[0]);ci_low=float(ic.ci_low.iloc[0]);ci_high=float(ic.ci_high.iloc[0])
 a[1].errorbar([0],[value],yerr=[[value-ci_low],[ci_high-value]],fmt='o',capsize=5);a[1].set_xticks([0],['Official50']);a[1].set_ylim(-.1,1);panel(a[1],'B','Fixed-phase absolute agreement',y='ICC(A,1), conditional 95% CI');a[1].text(.03,.96,f'ICC(A,1) = {value:.3f}\n95% CI [{ci_low:.3f}, {ci_high:.3f}]',transform=a[1].transAxes,ha='left',va='top',fontsize=8,bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=2))
 a[2].scatter(pm.A_phase0,pm.A_mean,s=17,color=COLORS[0]);zero(a[2]);a[2].axvline(0,color='.6',lw=.7);panel(a[2],'C','Phase averaging changes phenotype',x='Phase0 A (CW +)',y='Mean over twelve phases (CW +)')
 q=e[(e.group==GROUPS[0])&e['transform'].str.contains('minus|plus')].groupby('model_id')[['normalized_error_A','normalized_error_S']].mean();points(a[3],[q.normalized_error_A,q.normalized_error_S],['A','S'],paired=True);panel(a[3],'D','Fixed ±3-pixel placement, N=10',y='Mean normalized error / G')
 a[4].scatter(geom.x[~geom.outer_ROI],geom.y[~geom.outer_ROI],s=5,c='.8');a[4].scatter(geom.x[geom.outer_ROI],geom.y[geom.outer_ROI],s=5,c=COLORS[0]);a[4].set_aspect('equal');panel(a[4],'E','Actual receptor sampling ROI',x='x (retinal-image pixels)',y='y (pixels)')
 q=roi[(roi.group==GROUPS[0])&(roi.phase_index==0)&roi.ROI.isin(['full','outer'])].pivot(index='model_id',columns='ROI',values='A');points(a[5],[q.full,q.outer],['Full annulus','Outer ROI'],paired=True);panel(a[5],'F','Period arc ≥2 neighbor spacings',y='A (CW +), separate N=10');return f

def fig5():
 n=csv('data/precomputed/fig05/learning_normalized_metrics.csv');support=csv('data/precomputed/fig05/fig05_seed_support.csv');e=csv('data/precomputed/fig05/fig05_EPE_step_summary.csv');f,a=layout()
 a[0].hlines(support.seed,0,support.last_regular_step/1000,color=COLORS[0],linewidth=0.8);a[0].axvline(100,color='black',ls='--');panel(a[0],'A','Unequal regular checkpoint support',x='Training step (×1000)',y='Custom seed')
 a[1].plot(e.optimizer_step/1000,e['median'],color='black');a[1].fill_between(e.optimizer_step/1000,e.q25,e.q75,color='.8');a[1].axvline(100,color='.4',ls='--');a[1].axhline(csv('data/precomputed/fig05/official50_sintel_epe.csv').fixed_eval_epe.median(),color=COLORS[1],ls=':',label='Official50 median');a[1].legend(fontsize=9);panel(a[1],'B','Diagnostic, non-held-out Sintel',x='Training step (×1000)',y='Endpoint error')
 common=n[n.optimizer_step<=100000].copy();common['step_k']=common.optimizer_step/1000;common['abs_A']=common.A.abs()
 for i,g in enumerate(GROUPS):
  q=common[common.group==g];traj(a[2],q,'step_k','abs_A','seed',COLORS[i],label=LABELS[i]);traj(a[3],q,'step_k','abs_A_norm','seed',COLORS[i],label=LABELS[i])
 panel(a[2],'C','Raw magnitude, common N=14',x='Training step (×1000)',y='|A|');panel(a[3],'D','Magnitude-normalized trajectories',x='Training step (×1000)',y='|A| / paired G');a[2].legend(fontsize=8,frameon=False)
 end=n[n.optimizer_step==100000]
 sign=csv('data/precomputed/fig05/learning_stability_sign_tendency_100k.csv').set_index('group')
 sign_labels=[f"{label}\n{int(sign.loc[group,'reported_sign_count'])}/{int(sign.loc[group,'n_seeds'])} {sign.loc[group,'reported_direction']}" for group,label in zip(GROUPS,LABELS)]
 points(a[4],[end[end.group==g].A for g in GROUPS],sign_labels);a[4].tick_params(axis='x',labelsize=8);panel(a[4],'E','Fixed-phase sign tendencies, N=14',y='A at step 100k');points(a[5],[end[end.group==g].abs_A_norm for g in GROUPS],LABELS);panel(a[5],'F','Common-support normalized endpoints',y='|A| / G at step 100k');check('common support14',all(len(end[end.group==g])==14 for g in GROUPS));return f

def fig6(data):
 """Five-panel, inference-free reconstruction from canonical compact tables."""
 from matplotlib.patches import FancyBboxPatch, Rectangle

 required = {'panel_b_intervention', 'panel_c_physical', 'panel_d_pairing', 'panel_e_direction', 'fig6_QA'}
 if data is None or not required.issubset(data):
  raise ValueError('Fig. 6 rendering requires validated canonical panel data')
 for _, row in data['fig6_QA'].iterrows():
  check(row['check'], row['passed'])

 # Match the existing main-figure typography while allocating more area to E.
 fig = plt.figure(figsize=(12.2, 8.1), layout='constrained')
 grid = fig.add_gridspec(2, 12, height_ratios=[1.0, 1.08], hspace=.34)
 ax_a = fig.add_subplot(grid[0, :5])
 ax_b = fig.add_subplot(grid[0, 5:8])
 ax_c = fig.add_subplot(grid[0, 8:])
 ax_d = fig.add_subplot(grid[1, :6])
 ax_e = fig.add_subplot(grid[1, 6:])

 def panel6(ax, letter, title, x=None, y=None):
  ax.set_title(title, loc='left', fontsize=9.5, pad=6)
  ax.text(0, 1.13, letter, transform=ax.transAxes, weight='bold', fontsize=12.5)
  if x: ax.set_xlabel(x, fontsize=9.2, labelpad=4)
  if y: ax.set_ylabel(y, fontsize=9.0, labelpad=-3 if letter in {'B', 'C'} else 4)
  ax.tick_params(labelsize=8.7)

 # Analysis map, not a causal pathway diagram. The lower framed lane makes
 # the distinct network × decoder test part of the schematic itself.
 ax_a.set_xlim(0, 1); ax_a.set_ylim(0, 1); ax_a.axis('off')
 boxes = [(.01, .70, .20, .17, 'FlyVis\nnetwork'),
         (.27, .70, .20, .17, 'Decoder-input\nactivity'),
         (.53, .70, .17, .17, 'Decoder'),
         (.77, .70, .21, .17, 'Decoded\nflow')]
 for x0, y0, width, height, label in boxes:
  patch = FancyBboxPatch((x0, y0), width, height, boxstyle='round,pad=0.012',
                         facecolor='#f3f5f7', edgecolor='#536273', linewidth=1.0)
  ax_a.add_patch(patch)
  ax_a.text(x0+width/2, y0+height/2, label, ha='center', va='center', fontsize=8.2)
 for x0, x1 in [(.215, .265), (.475, .525), (.705, .765)]:
  ax_a.annotate('', xy=(x1, .785), xytext=(x0, .785),
                arrowprops=dict(arrowstyle='->', lw=1.0, color='.3'))
 ax_a.text(.37, .55, 'T4 / T4+T5 replacement (B/C)', ha='center', va='center', fontsize=8.0)
 ax_a.annotate('', xy=(.37, .69), xytext=(.37, .59),
               arrowprops=dict(arrowstyle='-|>', linestyle='--', color=COLORS[0], lw=1.15))

 lane = FancyBboxPatch((.035, .07), .93, .31, boxstyle='round,pad=0.012',
                       facecolor='#fafafa', edgecolor='.62', linewidth=.9,
                       linestyle=(0, (3, 2)))
 ax_a.add_patch(lane)
 ax_a.text(.50, .325, 'Network–decoder pairing test (D/E)',
           ha='center', va='center', fontsize=8.0, color='.25')
 for x0, label in [(.13, 'Network i'), (.62, 'Decoder j')]:
  patch = FancyBboxPatch((x0, .14), .25, .105, boxstyle='round,pad=0.01',
                         facecolor='white', edgecolor='#536273', linewidth=.9)
  ax_a.add_patch(patch)
  ax_a.text(x0+.125, .1925, label, ha='center', va='center', fontsize=8.0)
 ax_a.annotate('', xy=(.615, .1925), xytext=(.39, .1925),
               arrowprops=dict(arrowstyle='->', lw=1.0, color='.3'))
 ax_a.text(.50, .105, 'all network i × decoder j pairs', ha='center', va='center', fontsize=7.2, color='.3')
 panel6(ax_a, 'A', 'Intervention and pairing tests')

 # B: primary signed endpoint, normalized by the paired no-op flow magnitude.
 b = data['panel_b_intervention']
 x_positions = {'T4': 0, 'T4_T5': 1}
 colors = {'T4': COLORS[0], 'T4_T5': COLORS[1]}
 for _, q in b.groupby('model_id', sort=True):
  q = q.set_index('intervention')
  if {'T4', 'T4_T5'}.issubset(q.index):
   ax_b.plot([0, 1], [q.loc['T4', 'normalized_change'], q.loc['T4_T5', 'normalized_change']],
             color='.84', lw=.65, zorder=1)
 for intervention, xpos in x_positions.items():
  q = b[b.intervention == intervention].sort_values('model_id')
  jitter = .055 * np.sin(np.arange(len(q)) * 2.4)
  ax_b.scatter(xpos + jitter, q.normalized_change, s=23, color=colors[intervention],
               alpha=.78, zorder=2)
  ax_b.plot([xpos-.16, xpos+.16], [q.normalized_change.mean()] * 2,
            color='black', lw=1.35, zorder=3)
 ax_b.set_xticks([0, 1], ['T4', 'T4+T5']); zero(ax_b)
 panel6(ax_b, 'B', 'C=48 MIDD intervention, N=10', x='Decoder-input replacement',
        y='ΔA / G_no-op')

 # C: the same endpoint, with the two physical stimulus classes visibly grouped.
 c = data['panel_c_physical']
 categories = [('propeller', 'T4', 0.0), ('propeller', 'T4_T5', .95),
              ('edge', 'T4', 2.25), ('edge', 'T4_T5', 3.20)]
 for condition, intervention, xpos in categories:
  q = c[(c.condition == condition) & (c.intervention == intervention)].sort_values('model_id')
  jitter = .045 * np.sin(np.arange(len(q)) * 2.4)
  ax_c.scatter(xpos + jitter, q.normalized_change, s=21, color=colors[intervention],
               alpha=.78, zorder=2)
  ax_c.plot([xpos-.12, xpos+.12], [q.normalized_change.mean()] * 2,
            color='black', lw=1.25, zorder=3)
 ax_c.set_xlim(-.55, 3.75)
 ax_c.set_xticks([0, .95, 2.25, 3.20], ['T4', 'T4+T5', 'T4', 'T4+T5'])
 ax_c.tick_params(axis='x', labelsize=8.2, pad=3)
 ax_c.axvline(1.60, color='.82', lw=.7, zorder=0)
 ax_c.set_ylim(top=1.1)
 ax_c.text(.475, .98, 'Propeller', transform=ax_c.get_xaxis_transform(),
           ha='center', va='top', fontsize=8.5, color='.25')
 ax_c.text(2.725, .98, 'Moving edge', transform=ax_c.get_xaxis_transform(),
           ha='center', va='top', fontsize=8.5, color='.25')
 zero(ax_c)
 panel6(ax_c, 'C', 'Physical-motion contrasts, N=10')

 # D: network is the unit; the nine mismatches are averaged within network.
 d = data['panel_d_pairing'].sort_values('network_id')
 for _, row in d.iterrows():
  ax_d.plot([0, 1], [row.matched_abs_A, row.mismatched_mean_abs_A],
            color='.80', lw=.75, zorder=1)
 ax_d.scatter(np.zeros(len(d)), d.matched_abs_A, color=COLORS[0], s=23, zorder=2)
 ax_d.scatter(np.ones(len(d)), d.mismatched_mean_abs_A, color=COLORS[1], s=23, zorder=2)
 for xpos, values in enumerate([d.matched_abs_A, d.mismatched_mean_abs_A]):
  ax_d.plot([xpos-.15, xpos+.15], [values.mean()] * 2, color='black', lw=1.45, zorder=3)
 ax_d.set_xticks([0, 1], ['Matched decoder', 'Mean of 9 mismatched decoders'])
 zero(ax_d)
 panel6(ax_d, 'D', 'Matched vs mean mismatched |A|, N=10',
        x='One paired observation per network', y='|A| (stable window)')

 # E: retain the complete categorical 10×10 physical-direction matrix.
 e = data['panel_e_direction']
 network_ids = sorted(e.network_id.unique())
 decoder_ids = sorted(e.decoder_id.unique())
 matrix = e.pivot(index='network_id', columns='decoder_id', values='both_correct').reindex(
     index=network_ids, columns=decoder_ids).to_numpy(dtype=bool)
 ax_e.imshow(matrix, cmap=matplotlib.colors.ListedColormap(['#f2f2f2', '#007f5f']),
             vmin=0, vmax=1, aspect='equal', interpolation='nearest')
 ax_e.set_xticks(range(len(decoder_ids)), [f'{v:03d}' for v in decoder_ids], rotation=90)
 ax_e.set_yticks(range(len(network_ids)), [f'{v:03d}' for v in network_ids])
 ax_e.set_xlabel('Decoder ID\nMatched 10/10; mismatched 9/90', labelpad=4)
 ax_e.set_ylabel('Network ID')
 ax_e.xaxis.label.set_size(9.3)
 ax_e.yaxis.label.set_size(10.5)
 ax_e.tick_params(labelsize=9.5)
 for i in range(10):
  ax_e.add_patch(Rectangle((i-.5, i-.5), 1, 1, fill=False,
                           edgecolor=COLORS[2], linewidth=1.15))
 panel6(ax_e, 'E', 'Both physical directions correct')
 return fig

def supp3():
 t=csv('data/precomputed/supplementary/figS3/duration_paired_timecourses.csv');w=csv('data/precomputed/supplementary/figS3/duration_window_metrics.csv');u=csv('data/precomputed/supplementary/figS3/duration_uniform_late_control.csv');f,a=layout(8);short=csv('data/precomputed/supplementary/figS3/05_A_S_temporal_profiles.csv');
 for i,g in enumerate(GROUPS):
  q=short[short.group==g].groupby('frame').A.mean();a[0].plot(q.index*.02,q,color=COLORS[i],label=LABELS[i])
 panel(a[0],'A','Separate official50 short-window context',x='Recorded time (s)',y='Mean A');a[0].legend(fontsize=8,frameon=False)
 base=t[(t.group==GROUPS[0])&(t.fade_seconds==1)&(t.jitter_pixels==0)]
 for i,(mid,q) in enumerate(base.groupby('model_id')):
  for k,col in [(1,'A'),(2,'S')]:a[k].plot(q.time_seconds,q[col],color=COLORS[i],label=f'{mid:03d}')
  a[3].plot(q.time_seconds,q.S+q.A,color=COLORS[i]);a[3].plot(q.time_seconds,q.S-q.A,color=COLORS[i],ls='--')
 for k,y in [(1,'A'),(2,'S'),(3,'T')]:panel(a[k],chr(65+k),'Selected N=3 continuous static C48',x='Recorded time (s)',y=y);a[k].axvspan(9,10,color='.85',alpha=.5);zero(a[k])
 a[1].legend(fontsize=8,frameon=False)
 q=w[(w.group==GROUPS[0])&(w.fade_seconds==1)&(w.jitter_pixels==0)].pivot(index='model_id',columns='window',values='A');points(a[4],[q.early,q.late],['Early 0.2–1 s','Late 9–10 s'],paired=True);panel(a[4],'E','Late paired output persists',y='C=48 A')
 q=w[(w.group==GROUPS[0])&(w.jitter_pixels==0)&(w.window=='late')];traj(a[5],q,'fade_seconds','A','model_id',COLORS[0]);panel(a[5],'F','Late fade-in comparison',x='Preparation duration (s)',y='Late A')
 q=w[(w.group==GROUPS[0])&(w.fade_seconds==1)&(w.window=='late')]
 for i,(mid,z) in enumerate(q.groupby('model_id')):a[6].plot(z.jitter_pixels,z.A.abs(),'-o',color=COLORS[i],label=f'{mid:03d}')
 panel(a[6],'G','Jitter effect is model-dependent',x='Jitter amplitude (pixels)',y='Late |A|')
 q=q[q.jitter_pixels==0];points(a[7],[q.T_negative,u[u.jitter_pixels==0]['T']],['Balanced negative','Uniform gray'],paired=True);panel(a[7],'H','Nonzero late controls, N=3',y='Late tangential T');check('late3 abovefloor',np.all(abs(q.A)>.0005));return f

def supp5():
 sh=csv('data/precomputed/fig02/official50_shuffle_model_metrics.csv');ctx=csv('data/precomputed/fig02/official50_static_physical_context.csv');raw=csv('data/precomputed/fig02/official50_controls_metrics.csv');p=csv('data/precomputed/fig04/phase_paired.csv');pm=csv('data/precomputed/fig04/official50_phase_model_metrics.csv');f,a=layout(8)
 points(a[0],[sh[sh.group==g].difference_mean for g in GROUPS],['C48','C207','4MIDD']);panel(a[0],'A','Same image/LR operator, official50',y='Canonical |A| − mean shuffle |A|')
 points(a[1],[sh[sh.group==g].canonical_percentile for g in GROUPS],['C48','C207','4MIDD']);panel(a[1],'B','Canonical rank in ten shuffle draws',y='Within-model percentile');a[1].set_ylim(-.05,1.05)
 q=ctx[ctx.group==GROUPS[0]].copy();q['value']=q.A.abs()/q.G;shuffle=sh[sh.group==GROUPS[0]];draw=csv('data/precomputed/fig02/official50_shuffle_draw_metrics.csv');sd=draw[draw.group==GROUPS[0]].groupby('model_id').abs_A_norm.mean()
 vals=[q[q.condition=='canonical image/LR'].value,sd,q[q.condition=='random image/LR'].value,q[q.condition=='physical propeller'].value];points(a[2],vals,['Canonical','Shuffle','Random','Propeller']);a[2].set_yscale('symlog',linthresh=.002);panel(a[2],'C','Paired static/physical context, N=50',y='|Image/LR or CW/CCW contrast| / G')
 uniform=raw[raw.condition=='context_uniform'];negative=raw[raw.condition=='static:3MIDD_048__negative'];points(a[3],[uniform.mean_signed_tangent/uniform.mean_flow_magnitude,negative.mean_signed_tangent/negative.mean_flow_magnitude],['Uniform','Balanced negative'],paired=True);panel(a[3],'D','Unpaired static tangent biases',y='T/G (CW +), distinct from paired A/G')
 for i,g in enumerate(GROUPS):traj(a[4],p[p.group==g],'phase_deg','S','model_id',COLORS[i],summary='mean',label=LABELS[i])
 panel(a[4],'E','Shared phase dependence, N=10',x='Phase (degrees)',y='S (CW +)');zero(a[4]);a[4].legend(fontsize=9)
 points(a[5],[pm.phasewise_reversal_fraction],['Twelve-phase fraction']);panel(a[5],'F','Reversal definitions, official50',y='Fraction of phases with raw reversal');a[5].set_ylim(-.05,1.08);a[5].text(.04,.96,f'Averaged-pair reversal: {pm.reversal_of_phase_averaged_pair.sum()}/50',transform=a[5].transAxes,va='top',fontsize=11)
 e=csv('data/precomputed/fig04/official50_performance_phase_join.csv')
 for j,col in [(6,'A'),(7,'normalized_SD')]:
  a[j].scatter(e.fixed_eval_epe,e[col],s=18,c=COLORS[j%3]);panel(a[j],chr(65+j),'Fixed diagnostic task performance, N=50',x='Sintel EPE (not held out)',y='Phase0 A (CW +)' if col=='A' else 'Phase SD(A) / mean G')
 return f

def supp6():
 d=csv('data/precomputed/supplementary/figS6/sector_paired.csv');geo=csv('data/precomputed/fig04/receptor_geometry.csv');rmed=geo.loc[geo.full_annulus,'radius'].median();spacing=geo.loc[geo.full_annulus,'nearest_neighbor_spacing'].median();f,a=plt.subplots(3,3,figsize=(12,9),layout='constrained')
 for j,g in enumerate(GROUPS):
  q=d[d.group==g].copy();sectors_per_unit=4 if g.startswith('4') else 3;q['sampling_ratio']=rmed*2*np.pi/(q.sector_count/sectors_per_unit)/spacing
  for i,col in enumerate(['A','abs_A_norm','S']):
   ax=a[i,j]
   for phase,style in [(0,'-'),(.5,'--')]:
    z=q[q.phase_fraction==phase];traj(ax,z,'sampling_ratio',col,'model_id',COLORS[j],summary='mean');means=z.groupby('sampling_ratio')[col].mean();ax.plot(means.index,means,ls=style,c=COLORS[j],lw=2,label=f'Phase {phase}')
   zero(ax);panel(ax,chr(65+i*3+j),LABELS[j],x='Unit-period arc / neighbor spacing',y=col+' (CW +)' if col!='abs_A_norm' else '|A| / G')
 a[0,0].legend(fontsize=9);return f

def supp7():
 d=csv('data/precomputed/fig03/comparator_model_metrics.csv');m=csv('data/precomputed/fig03/composition_model_metrics.csv');n=csv('data/precomputed/fig03/composition_control_summary.csv');f,a=layout();labs=['Own A3','Other mean','All donors','Delta C']
 for i,col in enumerate(['cv_r2','cv_rmse','sign_accuracy']):
  vals=[d[d.predictor==p].sort_values('model_id')[col] for p in ['P0','P1','P2','P3']];points(a[i],vals,labs,paired=True);panel(a[i],chr(65+i),'Conditional held-out affine validation',y={'cv_r2':'LOLO R²','cv_rmse':'LOLO RMSE (CCW-positive scalar)','sign_accuracy':'LOLO sign accuracy'}[col]);a[i].tick_params(axis='x',rotation=15)
 for i,col in [(3,'slope'),(4,'intercept')]:points(a[i],[m[col]],[col]);panel(a[i],chr(65+i),'Own-component affine fit',y=col)
 q=n[n.family.str.startswith('smooth')];a[5].errorbar(range(len(q)),q.delta_r2,yerr=[q.delta_r2-q.ci_low,q.ci_high-q.delta_r2],fmt='o',capsize=4);a[5].set_xticks(range(len(q)),q.family.str.replace('smooth_potential_ls','Lengthscale '));panel(a[5],'F','Smooth-null sensitivity; 2000 draws',y='Own minus null CV R²');a[5].text(.02,.03,'Rank p floor 1/2001; Holm floor 4/2001',transform=a[5].transAxes,fontsize=10);return f

def supp8():
 t=csv('data/precomputed/fig05/learning_threshold_summary.csv');n=csv('data/precomputed/fig05/learning_normalized_metrics.csv');p=csv('data/precomputed/fig05/physical_motion_trajectory.csv');controls=csv('data/precomputed/fig05/learning_fixed_control_pairs.csv');st=csv('data/precomputed/fig05/learning_stability_checkpoint_summary.csv');f,a=layout(8);q=t[t.support=='common_100k']
 for i,g in enumerate(GROUPS):
  z=q[(q.group==g)&(q.definition=='first')];a[0].plot(z.threshold,z.n_crossed,'-o',c=COLORS[i],label=LABELS[i]);z=q[(q.group==g)&(q.definition=='persistent_three_regular')];a[1].plot(z.threshold,z.median_crossing_among_crossers/1000,'o-',c=COLORS[i])
 panel(a[0],'A','Fixed threshold sensitivity',x='|A| threshold',y='Crossing seeds / 14');a[0].set_ylim(0,15);a[0].legend(fontsize=9);panel(a[1],'B','Three-checkpoint persistent crossings',x='|A| threshold',y='Step ×1000 among crossers')
 for k,col in [(2,'A'),(3,'M')]:
  for i,g in enumerate(GROUPS):
   z=n[(n.group==g)&(n.optimizer_step<=100000)].copy();z['step_k']=z.optimizer_step/1000;traj(a[k],z,'step_k',col,'seed',COLORS[i])
  zero(a[k]);panel(a[k],chr(65+k),'Common fourteen-seed support',x='Training step ×1000',y=col+' (CW +)')
 for i,name in enumerate(['MIDD','shuffle','negative']):
  q=controls[controls.condition==name].copy();q['step_k']=q.optimizer_step/1000;traj(a[4],q,'step_k','abs_A_norm','seed',COLORS[i],label=name)
 panel(a[4],'E','Fixed-checkpoint static pairs, N=14',x='Step ×1000',y='|Paired half-difference| / G');a[4].legend(fontsize=10)
 q=controls[controls.condition=='uniform singleton'].copy();q['step_k']=q.optimizer_step/1000;q['ratio']=q['T']/q.G;traj(a[5],q,'step_k','ratio','seed',COLORS[1]);panel(a[5],'F','Uniform bias: singleton, not A',x='Step ×1000',y='Uniform T/G (CW +)');zero(a[5])
 for seed,z in p.groupby('seed'):a[6].plot(z.optimizer_step/1000,z.propeller_contrast,'o-',label=f'{seed:03d}');a[7].plot(z.optimizer_step/1000,z.edge_contrast,'s-',label=f'{seed:03d}')
 panel(a[6],'G','Separate three-seed physical trajectory',x='Step ×1000',y='CW/CCW propeller contrast');panel(a[7],'H','Separate three-seed edge trajectory',x='Step ×1000',y='Right/left horizontal contrast');a[6].legend(fontsize=10);return f

def supp9():
 d=csv('data/precomputed/fig06/robust_intervention_model_metrics.csv');d=d[d.condition=='MIDD'];c=csv('data/precomputed/fig06/exp09_coupling_by_model.csv');h=csv('data/precomputed/fig06/exp10_model_level_statistics_corrected.csv');e=csv('data/precomputed/fig06/mve02_sintel_specificity.csv');f,a=layout(8)
 sham=[x for x in d.intervention.unique() if x not in ['T4','T4a','T4b','T4+T5','T4_T5']];points(a[0],[d[d.intervention==x].normalized_change for x in sham],sham);panel(a[0],'A','Common-gray sham/all-input controls',y='C48 ΔA / paired noop G')
 for i,ct in enumerate(['T4a','T4b','T4c','T4d']):
  q=c[c.cell_type==ct];a[1].scatter(i+.05*np.sin(np.arange(len(q))),q.rho_signed,s=17,color=COLORS[i%3])
 a[1].set_xticks(range(4),['T4a','T4b','T4c','T4d']);zero(a[1]);panel(a[1],'B','Separate activity–flow associations',y='Signed within-model Spearman')
 q=h[(h.method=='baseline') & h.intervention_id.str.contains('T4')];q=q[q.intervention_id.isin(['group__T4_T5__baseline','group__T4_family__baseline','single__T4a__baseline','single__T4b__baseline'])];a[2].errorbar(range(len(q)),q.mean_loss,yerr=[q.mean_loss-q.ci95_low,q.ci95_high-q.mean_loss],fmt='o',color=COLORS[0]);a[2].set_xticks(range(len(q)),[x.replace('__baseline','').replace('single__','').replace('group__','') for x in q.intervention_id],rotation=25,fontsize=10);panel(a[2],'C','Separate 21-condition baseline',y='Projected MIDD loss');zero(a[2])
 for i,ct in enumerate(['T4c','T4d']):
  q=e[e.set_name==ct];a[3].scatter(np.full(len(q),i)-.1,q.delta_EPE,color=COLORS[i],label=ct);a[3].scatter(np.full(len(q),i)+.1,q.random_mean_delta_EPE,color=COLORS[i],facecolors='none')
 a[3].set_xticks([0,1],['T4c','T4d']);panel(a[3],'D','Separate recurrent operation: selected N6',y='Sintel EPE change');a[3].text(.02,.98,'Filled target / open random',transform=a[3].transAxes,va='top',fontsize=8)
 for k,col in [(4,'projected_MIDD_gain'),(5,'G_noop')]:
  gs=list(d.intervention.unique());points(a[k],[d[d.intervention==g][col] for g in gs],gs);panel(a[k],chr(65+k),'All common-gray groups; no clipping',y=col.replace('_',' '))
 # Preserve source-family activity curves and paired recurrent MIDD effects as distinct evidence.
 ac=csv('data/precomputed/fig06/exp09_activity_by_model_luminance.csv')
 for i,ct in enumerate(['T4a','T4b','T4c','T4d']):
  z=ac[(ac.cell_type==ct)&ac.response_identifiable];traj(a[6],z,'c','e_normalized','model_id',COLORS[i%3],summary='mean',label=ct)
 panel(a[6],'G','Identifiable activity profiles',x='Luminance C',y='Normalized projected evidence');a[6].legend(fontsize=7,frameon=False,ncol=4)
 rm=csv('data/precomputed/fig06/mve02_midd_matched_random.csv')
 agg=rm.groupby(['seed','cohort','set_name','control']).delta_A.mean().reset_index()
 for i,ct in enumerate(['T4c','T4d']):
  z=agg[agg.set_name==ct];target=z[z.control=='target'].sort_values(['seed','cohort']);random=z[z.control=='matched_random'].sort_values(['seed','cohort'])
  for yy,marker,offset in [(target,'o',-.12),(random,'s',.12)]:a[7].scatter(np.full(len(yy),i)+offset,yy.delta_A,s=18,marker=marker,color=COLORS[i])
 a[7].set_xticks([0,1],['T4c','T4d']);zero(a[7]);panel(a[7],'H','Separate recurrent MIDD: selected N6',y='Phase-averaged ΔA');a[7].text(.03,.98,'Circle target / square mean random',transform=a[7].transAxes,va='top',fontsize=8)
 qa=csv('data/precomputed/fig06/replacement_qa.csv');check('replacement source QA',qa.passed.all());a[0].text(.02,.08,'No-op exact; nontarget unchanged',transform=a[0].transAxes,fontsize=8)
 return f

def supp10():
 d=csv('data/precomputed/fig06/crossswap_metrics.csv');b=csv('data/precomputed/fig06/crossswap_decomposition.csv');m=csv('data/precomputed/fig06/crossswap_matched_model_units.csv');direction=csv('data/precomputed/fig06/crossswap_direction_accuracy.csv');f,a=layout()
 for k,(g,metric,title) in enumerate([(GROUPS[0],'A','MIDD A (CW +)'),('motion_grating','Q_motion','Physical horizontal-grating Q')]):
  q=d[(d.group==g)&(d.window=='stable')];X=q.pivot(index='network_id',columns='decoder_id',values=metric);lim=abs(X.to_numpy()).max();im=a[k].imshow(X,vmin=-lim,vmax=lim,cmap='RdBu_r');f.colorbar(im,ax=a[k],shrink=.7);a[k].set_xticks(range(10),X.columns,rotation=90);a[k].set_yticks(range(10),X.index);panel(a[k],chr(65+k),title,x='Decoder ID',y='Network ID')
  q=b[(b.group==g)&(b.window=='stable')&(b.metric==metric)].set_index('effect').loc[['network','decoder','residual']];a[2+k].errorbar(range(3),q.fraction*100,yerr=[100*(q.fraction-q.ci_low),100*(q.ci_high-q.fraction)],fmt='o',capsize=4);a[2+k].set_xticks(range(3),['Network','Decoder','Residual']);panel(a[2+k],chr(67+k),'Descriptive centered-SS decomposition',y='Fraction of centered SS (%)')
 q=m[(m.group==GROUPS[0])&(m.metric=='A')].sort_values('network_id');points(a[4],[q.matched.abs(),q.mismatched_abs_mean],['Matched','Mean mismatch'],paired=True);panel(a[4],'E','MIDD output magnitude increases in mismatch',y='|Source scalar A|')
 X=direction.pivot(index='network_id',columns='decoder_id',values='both_correct');a[5].imshow(X,cmap='Greys',vmin=0,vmax=1);a[5].set_xticks(range(10),X.columns,rotation=90);a[5].set_yticks(range(10),X.index);panel(a[5],'F','Both physical directions correct',x='Decoder ID (black = correct)',y='Network ID');return f

def render(key, analysis=None):
 INPUTS.clear();CHECKS.clear();plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'figure.dpi':130})
 functions={'1':fig1,'2':fig2,'3':fig3,'4':fig4,'5':fig5,'S3':supp3,'S5':supp5,'S6':supp6,'S7':supp7,'S8':supp8,'S9':supp9,'S10':supp10}
 if key == '6':
  for source in ['data/precomputed/fig06/replacement_metrics.csv',
                 'data/precomputed/fig06/crossswap_metrics.csv',
                 'data/precomputed/fig06/crossswap_direction_accuracy.csv']:
   register(source)
  if analysis is None:
   from .figure_analysis import fig6_panel_data
   analysis=fig6_panel_data({
    'replacement_metrics.csv':csv('data/precomputed/fig06/replacement_metrics.csv'),
    'crossswap_metrics.csv':csv('data/precomputed/fig06/crossswap_metrics.csv'),
    'crossswap_direction_accuracy.csv':csv('data/precomputed/fig06/crossswap_direction_accuracy.csv'),
   })
  fig=fig6(analysis)
 else:
  if key not in functions: raise KeyError(key)
  fig=functions[key]()
 check('source inputs present',bool(INPUTS));return save(fig,key)

if __name__=='__main__':
 for key in sys.argv[1:] or ['1','2','3','4','5','6','S3','S5','S6','S7','S8','S9','S10']:render(key);plt.close('all')


def add_s1_calibration(fig):
    INPUTS.clear();CHECKS.clear()
    for p in (ROOT/'data/precomputed/supplementary/figS1').iterdir():
        if p.is_file(): register(str(p.relative_to(ROOT)))
    d=csv('data/precomputed/supplementary/figS1/multifield_sign_correspondence.csv')
    for ax in fig.axes:
        ax.title.set_fontsize(12)
        ax.xaxis.label.set_fontsize(12); ax.yaxis.label.set_fontsize(12)
        ax.tick_params(labelsize=11)
        if len(ax.get_ylabel()) > 60: ax.set_ylabel('Moving / Static magnitude (log scale)')
        pos=ax.get_position();ax.set_position([pos.x0,pos.y0*.78+.22,pos.width,pos.height*.78])
    ax=fig.add_axes([.12,.055,.34,.13]);bx=fig.add_axes([.59,.055,.34,.13])
    for a,renderer in [(ax,'single'),(bx,'multi')]:
        q=d[d.renderer==renderer].sort_values('model_id')
        points(a,[q.CW_scalar,q.CCW_scalar],['Physical CW','Physical CCW'],paired=True)
        a.set_title('Independent N10 calibration: '+renderer,fontsize=12)
        a.set_ylabel('Implemented scalar')
    check('eight-field physical signs all10',d[d.renderer=='multi'].CW_negative.all() and d[d.renderer=='multi'].CCW_positive.all())
    return save(fig,'S1')
