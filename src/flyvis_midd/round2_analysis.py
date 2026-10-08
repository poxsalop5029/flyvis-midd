"""Model-unit Round 2 scalar comparisons; canonical notebooks and audit share code."""
import numpy as np
import pandas as pd
from .figure_analysis import evaluate_scalar_cv

def composition_comparators(a3, pairs):
    """Fixed original LOLO folds/sign support; all donors averaged within target."""
    mat=a3.pivot(index='model_id',columns='c',values='a3')
    if len(mat)!=10 or mat.shape[1]!=7 or mat.isna().any().any():
        raise ValueError('Expected complete original ten-model, seven-level cohort')
    records=[];donors=[]
    for mid,q in pairs.groupby('model_id',sort=True):
        q=q.sort_values(['c1','c2']);levels=list(mat.columns)
        i=np.array([levels.index(x) for x in q.c1]);j=np.array([levels.index(x) for x in q.c2])
        if len(q)!=21 or len(set(zip(i,j)))!=21:raise ValueError('Expected 21 canonical pairs')
        y=q.a4_observed.to_numpy();own=mat.loc[mid].to_numpy();x=own[i]-own[j]
        mask=abs(x)/np.ptp(own)>=.05
        folds=[(np.flatnonzero((i!=h)&(j!=h)),np.flatnonzero((i==h)|(j==h))) for h in range(7)]
        ensemble=mat.drop(mid).mean().to_numpy()
        xs=[('P0',x),('P1',ensemble[i]-ensemble[j]),('P3',q.c1.to_numpy()-q.c2.to_numpy())]
        for donor,v in mat.drop(mid).iterrows():
            v=v.to_numpy();predictor=v[i]-v[j];z=evaluate_scalar_cv(predictor,y,folds,mask)[0]
            slope,intercept=np.polyfit(predictor,y,1)
            donors.append(dict(model_id=mid,donor_id=donor,cv_r2=z[0],cv_rmse=z[1],normalized_rmse=z[1]/np.ptp(own),sign_accuracy=z[2],slope=slope,intercept=intercept))
        for label,predictor in xs:
            z=evaluate_scalar_cv(predictor,y,folds,mask)[0];slope,intercept=np.polyfit(predictor,y,1)
            records.append(dict(model_id=mid,predictor=label,cv_r2=z[0],cv_rmse=z[1],normalized_rmse=z[1]/np.ptp(own),sign_accuracy=z[2],slope=slope,intercept=intercept))
    donors=pd.DataFrame(donors)
    for mid,q in donors.groupby('model_id',sort=True):
        records.append(dict(model_id=mid,predictor='P2',**q[['cv_r2','cv_rmse','normalized_rmse','sign_accuracy','slope','intercept']].mean()))
    return pd.DataFrame(records),donors

def robust_intervention_metrics(raw):
    """Same paired-noop-magnitude operator for static/physical scalar contrasts."""
    rows=[]
    for mid,q in raw.groupby('model_id',sort=True):
        t=q.pivot(index='condition',columns='intervention',values='mean_signed_tangent')
        u=q.pivot(index='condition',columns='intervention',values='mean_u')
        g=q.pivot(index='condition',columns='intervention',values='mean_flow_magnitude')
        levels=[0,48,85,128,170,207,255]
        all_a={k:np.array([(t.loc[f'MIDD_{v}_original',k]-t.loc[f'MIDD_{v}_mirror',k])/2 for v in levels]) for k in t}
        for k in t:
            if k=='noop':continue
            for name,conditions,values in [('MIDD',['MIDD_48_original','MIDD_48_mirror'],t),('propeller',['CW','CCW'],t),('edge',['Right','Left'],u)]:
                base=(values.loc[conditions[0],'noop']-values.loc[conditions[1],'noop'])/2
                changed=(values.loc[conditions[0],k]-values.loc[conditions[1],k])/2
                magnitude=g.loc[conditions,'noop'].mean()
                if not np.isfinite(magnitude) or magnitude<=0:raise ValueError('Nonpositive/nonfinite paired baseline magnitude')
                delta=changed-base;norm2=np.dot(all_a['noop'],all_a['noop'])
                if norm2<=0:raise ValueError('Undefined secondary projected gain')
                rows.append(dict(model_id=mid,intervention=k,condition=name,baseline=base,intervention_value=changed,G_noop=magnitude,raw_delta=delta,normalized_change=delta/magnitude,baseline_aligned_attenuation=-np.sign(base)*delta/magnitude,projected_MIDD_gain=np.dot(all_a['noop'],all_a[k])/norm2,RMS_normalized_difference=np.sqrt(np.mean((all_a[k]-all_a['noop'])**2))/np.sqrt(np.mean(all_a['noop']**2))))
    return pd.DataFrame(rows)

def receptor_sampling_geometry(centers_yx, units=24, inner=.14, outer=.98):
    """Actual nearest-neighbor distances; color-unit arc is not edge Nyquist."""
    from scipy.spatial.distance import cdist
    centers=np.asarray(centers_yx,float)
    if centers.ndim!=2 or centers.shape[1]!=2 or not np.isfinite(centers).all():raise ValueError('Invalid receptor coordinates')
    if units<=0 or not 0<=inner<outer<=1:raise ValueError('Invalid period or annulus')
    delta=centers-centers.mean(0);radius=np.linalg.norm(delta,axis=1)
    distance=cdist(centers,centers);np.fill_diagonal(distance,np.inf);spacing=distance.min(1)
    if (spacing<=0).any():raise ValueError('Duplicate receptors')
    arc=radius*2*np.pi/units;full=(radius>=inner*radius.max())&(radius<=outer*radius.max())
    return pd.DataFrame(dict(receptor=np.arange(len(radius)),y=centers[:,0],x=centers[:,1],radius=radius,nearest_neighbor_spacing=spacing,period_arc=arc,period_over_spacing=arc/spacing,angular_spacing_rad=spacing/np.maximum(radius,1e-12),full_annulus=full,outer_ROI=full&(arc>=2*spacing)))


# Fixed-phase reliability and checkpoint contracts.
def assert_pairs(d):
 cols=['T_original','T_mirror','A','S'];assert np.isfinite(d[cols]).all().all()
 np.testing.assert_allclose(d.A,(d.T_original-d.T_mirror)/2,atol=1e-14)
 np.testing.assert_allclose(d.S,(d.T_original+d.T_mirror)/2,atol=1e-14)
 if 'actual_reversal' in d:assert np.array_equal(d.actual_reversal,d.T_original*d.T_mirror<0)

def icc_matrix(y):
 """ICC(A,1) primary, ICC(C,1) sensitivity; fixed repeated condition columns.

 Signed moment estimates are retained. Residual includes model×phase effects.
 """
 y=np.asarray(y,float)
 if y.ndim!=2 or min(y.shape)<2 or not np.isfinite(y).all():raise ValueError('ICC requires complete finite target × repeat matrix, each dimension≥2')
 n,k=y.shape;row=y.mean(1);col=y.mean(0);grand=y.mean()
 sr=k*np.sum((row-grand)**2);sc=n*np.sum((col-grand)**2);se=np.sum((y-row[:,None]-col[None,:]+grand)**2);st=np.sum((y-grand)**2)
 np.testing.assert_allclose(sr+sc+se,st,atol=1e-14)
 mr=sr/(n-1);mc=sc/(k-1);me=se/((n-1)*(k-1));dm=mr+(k-1)*me+k*(mc-me)/n;dc=mr+(k-1)*me
 observed_between=np.var(row,ddof=1);observed_within=np.mean(np.var(y,axis=1,ddof=1))
 return dict(n_models=n,n_phases=k,icc_absolute=(mr-me)/dm if dm>0 else np.nan,icc_consistency=(mr-me)/dc if dc>0 else np.nan,
  SS_model=sr,SS_phase=sc,SS_residual=se,SS_total=st,SS_model_fraction=sr/st if st>0 else np.nan,SS_phase_fraction=sc/st if st>0 else np.nan,SS_residual_fraction=se/st if st>0 else np.nan,
  MS_model=mr,MS_phase=mc,MS_residual=me,model_variance=(mr-me)/k,phase_variance=(mc-me)/n,residual_variance=me,
  within_phase_total_variance=(mc-me)/n+me,observed_between_phase_means_variance=observed_between,observed_mean_within_model_phase_variance=observed_within,
  observed_within_to_between=observed_within/observed_between if observed_between>0 else np.nan)

def phase_model_table(d):
 assert_pairs(d);out=[]
 for mid,q in d.groupby('model_id',sort=True):
  q=q.sort_values('phase_index');assert len(q)==12 and np.array_equal(q.phase_index,np.arange(12));a=q.A.to_numpy();mean=a.mean();median=np.median(a);q0=q.iloc[0];to=q.T_original.mean();tm=q.T_mirror.mean();g=q.G.mean()
  assert g>0
  out.append(dict(model_id=f'flow/0000/{int(mid):03d}',group=q.group.iloc[0],n_phases=12,A_phase0=q0.A,A_mean=mean,A_median=median,A_SD=np.std(a,ddof=1),A_IQR=np.quantile(a,.75)-np.quantile(a,.25),A_MAD=np.median(np.abs(a-median)),A_min=a.min(),A_max=a.max(),A_mean_absolute=np.abs(a).mean(),mean_G=g,normalized_SD=np.std(a,ddof=1)/g,normalized_range=np.ptp(a)/g,
    positive_phases=int((a>0).sum()),negative_phases=int((a<0).sum()),zero_phases=int((a==0).sum()),dominant_sign_fraction=max((a>0).sum(),(a<0).sum(),(a==0).sum())/len(a),sign_consistency_with_phase_mean=np.mean(np.sign(a)==np.sign(mean)),phase0_vs_mean_sign_agreement=np.sign(q0.A)==np.sign(mean),A_sign_changes=a.min()<0<a.max(),
    phasewise_reversal_count=int((q.T_original*q.T_mirror<0).sum()),phasewise_reversal_fraction=float((q.T_original*q.T_mirror<0).mean()),mean_T_original=to,mean_T_mirror=tm,mean_S=(to+tm)/2,reversal_of_phase_averaged_pair=to*tm<0,phase0_reversal=q0.T_original*q0.T_mirror<0))
 return pd.DataFrame(out)

def ordered_adjacent(q,step='optimizer_step',spacing=1000):
 q=q.sort_values(step)
 if q[step].duplicated().any():raise ValueError('Duplicate checkpoint step')
 if not np.isfinite(q[step]).all():raise ValueError('Nonfinite checkpoint')
 return q.assign(previous_step=q[step].shift(),previous_A=q.A.shift(),previous_abs_A=np.abs(q.A.shift())).loc[q[step].diff()==spacing].copy()
