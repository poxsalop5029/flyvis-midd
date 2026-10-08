"""Canonical stored-artifact analysis shared by main and supplementary notebooks.

No model loading or revision-code execution. Original repeated-observation units,
RNG sequence, frozen confidence intervals, and source contracts are preserved.
"""
from pathlib import Path
import hashlib
import json
import platform
import numpy as np
import pandas as pd

GROUPS = ['3MIDD_048', '3MIDD_207', '4MIDD_085_170']

def input_manifest(key, root):
    spec = json.loads(Path(__file__).with_name('figure_inputs.json').read_text())[key]
    paths = list(spec['input_paths'])
    # Optional saved summaries are hash-recorded regression references, not render inputs.
    paths.extend(p for p in spec.get('qa_reference_paths', []) if (root/p).is_file())
    return pd.DataFrame([{'path': p, 'sha256': hashlib.sha256((root/p).read_bytes()).hexdigest(),
                          'bytes': (root/p).stat().st_size} for p in paths])

def load_tables(manifest, root):
    return {row.path: pd.read_csv(root/row.path) for row in manifest.itertuples() if row.path.endswith('.csv')}

def table(tables, suffix):
    matches = [value for path, value in tables.items() if path.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f'Expected one source for {suffix}; found {len(matches)}')
    return matches[0]

def percentile_model_bootstrap(values, rng, draws=10000):
    """Historical whole-model mean percentile interval, with caller-owned RNG."""
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or not len(values):
        raise ValueError('Bootstrap requires nonempty finite model-unit values')
    samples = values[rng.integers(len(values), size=(draws, len(values)))].mean(axis=1)
    return float(values.mean()), *np.quantile(samples, [.025, .975])

def evaluate_scalar_cv(x, y, splits, sign_mask):
    """Frozen affine scalar validation; pooled folds are repeated within model."""
    x = np.atleast_2d(x); predictions=[]; observed=[]; masks=[]
    for train, test in splits:
        assert not set(train) & set(test)
        xt=x[:,train]; yt=y[train]; variance=((xt-xt.mean(1)[:,None])**2).sum(1)
        slope=np.divide(((xt-xt.mean(1)[:,None])*(yt-yt.mean())).sum(1),variance,
                        out=np.zeros(len(x)),where=variance>0)
        predictions.append(yt.mean()+slope[:,None]*(x[:,test]-xt.mean(1)[:,None]))
        observed.extend(y[test]); masks.extend(sign_mask[test])
    pred=np.concatenate(predictions,axis=1); obs=np.asarray(observed); mask=np.asarray(masks)
    return np.c_[1-((pred-obs)**2).sum(1)/((obs-obs.mean())**2).sum(),
                 np.sqrt(((pred-obs)**2).mean(1)),
                 (np.sign(pred[:,mask])==np.sign(obs[mask])).mean(1)]

def composition_analysis(tables):
    """Recompute canonical fits/CV and original bootstrap sequence from stored scalars.

    Null predictions are immutable cached draw metrics, not newly tuned draws.
    """
    a=table(tables,'fig3_3MIDD_components_per_model.csv')
    p=table(tables,'fig3_canonical_pair_predictions.csv'); models=[]; cv=[]
    for mid,q in p.groupby('model_id',sort=True):
        q=q.sort_values(['c1','c2']); v=a[a.model_id==mid].sort_values('c')
        levels=v.c.to_numpy(); components=v.a3.to_numpy()
        ij=np.array([[list(levels).index(c1),list(levels).index(c2)] for c1,c2 in q[['c1','c2']].to_numpy()])
        i,j=ij.T; x=components[i]-components[j]; y=q.a4_observed.to_numpy()
        gate=np.abs(x)/np.ptp(components)>=.05
        np.testing.assert_allclose(x,q.a4_predicted,atol=1e-14)
        slope,intercept=np.polyfit(x,y,1); fitted=intercept+slope*x
        models.append(dict(model_id=mid,slope=slope,intercept=intercept,
            r2=1-((y-fitted)**2).sum()/((y-y.mean())**2).sum(),
            normalized_rmse=np.sqrt(((y-x)**2).mean())/np.ptp(components),
            sign_agreement=np.mean(np.sign(x[gate])==np.sign(y[gate]))))
        tests=[np.flatnonzero((i==h)|(j==h)) for h in range(len(levels))]
        lolo=[(np.setdiff1d(np.arange(len(q)),test),test) for test in tests]
        loo=[(np.setdiff1d(np.arange(len(q)),[h]),np.array([h])) for h in range(len(q))]
        for name,splits in [('LOLO',lolo),('leave_one_pair_out',loo)]:
            values=evaluate_scalar_cv(x,y,splits,gate)[0]
            cv.append(dict(model_id=mid,split=name,cv_r2=values[0],cv_rmse=values[1],sign_accuracy=values[2]))
    models=pd.DataFrame(models); cv=pd.DataFrame(cv)
    ref=table(tables,'composition_model_metrics.csv').sort_values('model_id')
    for metric in ['slope','intercept','r2','normalized_rmse','sign_agreement']:
        np.testing.assert_allclose(models[metric],ref[metric],atol=1e-10,rtol=1e-9)
    refcv=table(tables,'composition_cv_metrics.csv').sort_values(['model_id','split'])
    for metric in ['cv_r2','cv_rmse','sign_accuracy']:
        np.testing.assert_allclose(cv.sort_values(['model_id','split'])[metric],refcv[metric],atol=1e-10,rtol=1e-9)
    # Original Rev05 consumed 2000 permutations then 2000x7 normals before CIs.
    rng=np.random.default_rng(2026100305)
    for _ in range(2000): rng.permutation(7)
    rng.normal(size=(2000,7)); boot=[]
    for source,df,metrics in [('in_sample',models,['slope','intercept','r2','sign_agreement','normalized_rmse']),
                              ('LOLO',cv[cv.split=='LOLO'],['cv_r2','cv_rmse','sign_accuracy'])]:
        for metric in metrics:
            mean,lo,hi=percentile_model_bootstrap(df[metric],rng)
            boot.append(dict(source=source,metric=metric,mean=mean,ci_low=lo,ci_high=hi))
    boot=pd.DataFrame(boot); refboot=table(tables,'composition_bootstrap_summary.csv')
    np.testing.assert_allclose(boot[['mean','ci_low','ci_high']],refboot[['mean','ci_low','ci_high']],atol=1e-10)
    draws=table(tables,'composition_null_draw_metrics.csv')
    comparisons=table(tables,'composition_null_comparison.csv'); controls=[]
    for family,q in comparisons.groupby('family',sort=False):
        mean,lo,hi=percentile_model_bootstrap(q.delta_r2,rng)
        controls.append(dict(family=family,delta_r2=mean,ci_low=lo,ci_high=hi))
        for mid,z in q.groupby('model_id'):
            stored=draws[(draws.family==family)&(draws.model_id==mid)].cv_r2.mean()
            np.testing.assert_allclose(z.control_r2.iloc[0],stored,atol=1e-10)
    controls=pd.DataFrame(controls); refctl=table(tables,'composition_control_summary.csv')
    np.testing.assert_allclose(controls[['delta_r2','ci_low','ci_high']],refctl[['delta_r2','ci_low','ci_high']],atol=1e-10)
    result={'affine_fit':models,'held_out_prediction':cv,'model_bootstrap':boot,'structured_nulls':controls}
    if any(path.endswith('comparator_model_metrics.csv') for path in tables):
        from .round2_analysis import composition_comparators
        calculated,donors=composition_comparators(a,p)
        ref=table(tables,'comparator_model_metrics.csv').sort_values(['model_id','predictor'])
        calculated=calculated.sort_values(['model_id','predictor'])
        np.testing.assert_allclose(calculated[['cv_r2','cv_rmse','sign_accuracy']],ref[['cv_r2','cv_rmse','sign_accuracy']],atol=1e-12)
        result.update(comparator_prediction=calculated,all_cross_model_donors=donors)
    return result

def validate_inputs(tables):
    """Within-contract algebra/finite QA, never combine cohorts or pair operators."""
    qa=[]
    for path,df in tables.items():
        if {'T_original','T_mirror','A','S'}.issubset(df):
            mask=df[['T_original','T_mirror','A','S']].notna().all(axis=1)
            good=np.allclose(df.loc[mask,'A'],(df.loc[mask,'T_original']-df.loc[mask,'T_mirror'])/2)
            good &= np.allclose(df.loc[mask,'S'],(df.loc[mask,'T_original']+df.loc[mask,'T_mirror'])/2)
            assert good,path
            qa.append(dict(check='A/S identity: '+path,passed=True))
    return pd.DataFrame(qa,columns=['check','passed'])

def fig6_panel_data(tables):
    """Build and QA the five adopted Fig. 6 panels from compact canonical tables.

    This is deterministic table aggregation only; it never loads a network or
    creates new inferential statistics. Model/pair structures are validated
    before the returned plotting tables can be consumed.
    """
    from .round2_analysis import robust_intervention_metrics

    qa = []
    def passed(name, condition):
        if not bool(condition):
            raise AssertionError(name)
        qa.append(dict(check=name, passed=True))

    raw = table(tables, 'replacement_metrics.csv')
    expected_models = list(range(10))
    passed('intervention cohort is models 000-009', sorted(raw.model_id.unique().tolist()) == expected_models)
    keys = ['model_id', 'condition', 'intervention']
    passed('intervention source has no duplicate keys', not raw.duplicated(keys).any())
    required_conditions = ['MIDD_48_original', 'MIDD_48_mirror', 'CW', 'CCW', 'Right', 'Left']
    required_interventions = ['noop', 'T4', 'T4_T5']
    required = raw[raw.condition.isin(required_conditions) & raw.intervention.isin(required_interventions)]
    expected_index = pd.MultiIndex.from_product(
        [expected_models, required_conditions, required_interventions],
        names=['model_id', 'condition', 'intervention'],
    )
    observed_index = pd.MultiIndex.from_frame(required[keys])
    passed('intervention source has complete required condition grid',
           len(required) == len(expected_index) and set(observed_index) == set(expected_index))
    numeric = ['mean_flow_magnitude', 'mean_signed_tangent', 'mean_u']
    passed('intervention source values are finite', np.isfinite(required[numeric].to_numpy(dtype=float)).all())

    calculated = robust_intervention_metrics(raw)
    panel_b = calculated[(calculated.condition == 'MIDD') & calculated.intervention.isin(['T4', 'T4_T5'])].copy()
    panel_c = calculated[calculated.condition.isin(['propeller', 'edge']) & calculated.intervention.isin(['T4', 'T4_T5'])].copy()
    passed('Panel B contains 10 models for T4 and T4+T5',
           len(panel_b) == 20 and set(panel_b.model_id.unique()) == set(expected_models)
           and panel_b.groupby('intervention').model_id.nunique().to_dict() == {'T4': 10, 'T4_T5': 10})
    passed('Panel C contains propeller and moving-edge data for 10 models',
           len(panel_c) == 40 and set(panel_c.condition) == {'propeller', 'edge'}
           and panel_c.groupby(['condition', 'intervention']).model_id.nunique().eq(10).all())
    passed('Panels B/C normalized endpoint is finite',
           np.isfinite(pd.concat([panel_b, panel_c]).normalized_change.to_numpy()).all())

    # Existing derived endpoint remains a numerical regression reference only.
    if any(path.endswith('robust_intervention_model_metrics.csv') for path in tables):
        stored = table(tables, 'robust_intervention_model_metrics.csv')
        current = pd.concat([panel_b, panel_c], ignore_index=True)
        merged = current.merge(stored, on=['model_id', 'intervention', 'condition'],
                               suffixes=('_new', '_stored'), validate='one_to_one')
        compared = ['baseline', 'intervention_value', 'G_noop', 'raw_delta', 'normalized_change']
        agreement = all(np.allclose(merged[f'{metric}_new'], merged[f'{metric}_stored'],
                                   atol=1e-12, rtol=0) for metric in compared)
        passed('intervention derived-table agreement', len(merged) == 60 and agreement)

    cross = table(tables, 'crossswap_metrics.csv')
    cross = cross[(cross.group == '3MIDD_048') & (cross.window == 'stable')].copy()
    networks = sorted(cross.network_id.unique().tolist())
    decoders = sorted(cross.decoder_id.unique().tolist())
    passed('Panel D uses complete 10 by 10 stable C48 cross-swap',
           len(networks) == 10 and networks == decoders and len(cross) == 100)
    passed('cross-swap pair keys are unique',
           not cross.duplicated(['network_id', 'decoder_id']).any())
    passed('cross-swap stable C48 A values are finite', np.isfinite(cross.A.to_numpy(dtype=float)).all())
    pair_matrix = cross.pivot(index='network_id', columns='decoder_id', values='A').reindex(
        index=networks, columns=decoders)
    passed('cross-swap matrix has no missing pair', pair_matrix.shape == (10, 10) and not pair_matrix.isna().any().any())
    matched_abs = pair_matrix.abs().to_numpy().diagonal()
    mismatch = pair_matrix.abs().copy()
    for network in networks:
        mismatch.loc[network, network] = np.nan
    panel_d = pd.DataFrame({
        'network_id': networks,
        'matched_abs_A': matched_abs,
        'mismatched_mean_abs_A': mismatch.mean(axis=1).to_numpy(),
        'n_mismatched_decoders': mismatch.notna().sum(axis=1).to_numpy(),
    })
    passed('each Panel D network has exactly nine mismatched decoders',
           panel_d.n_mismatched_decoders.eq(9).all() and len(panel_d) == 10)
    if any(path.endswith('crossswap_matched_model_units.csv') for path in tables):
        stored = table(tables, 'crossswap_matched_model_units.csv')
        stored = stored[(stored.group == '3MIDD_048') & (stored.metric == 'A')]
        merged = panel_d.merge(stored, on='network_id', validate='one_to_one')
        passed('matched/mismatch derived-table agreement', len(merged) == 10 and
               np.allclose(merged.matched_abs_A, merged['matched'].abs().to_numpy(), atol=1e-12, rtol=0) and
               np.allclose(merged.mismatched_mean_abs_A, merged.mismatched_abs_mean, atol=1e-12, rtol=0))

    direction = table(tables, 'crossswap_direction_accuracy.csv').copy()
    passed('direction table contains unique complete 10 by 10 pairs',
           len(direction) == 100 and sorted(direction.network_id.unique().tolist()) == networks and
           sorted(direction.decoder_id.unique().tolist()) == decoders and
           not direction.duplicated(['network_id', 'decoder_id']).any())
    passed('direction saved correctness definition is preserved',
           np.array_equal(direction.right_correct.to_numpy(bool), (direction.right_u > 0).to_numpy()) and
           np.array_equal(direction.left_correct.to_numpy(bool), (direction.left_u < 0).to_numpy()) and
           np.array_equal(direction.both_correct.to_numpy(bool),
                          (direction.right_correct & direction.left_correct).to_numpy(bool)) and
           np.array_equal(direction.matched.to_numpy(bool),
                          (direction.network_id == direction.decoder_id).to_numpy()))
    panel_e = direction.pivot(index='network_id', columns='decoder_id', values='both_correct').reindex(
        index=networks, columns=decoders)
    passed('Panel E is a complete 10 by 10 direction matrix',
           panel_e.shape == (10, 10) and not panel_e.isna().any().any())
    diagonal_mask = np.asarray(networks)[:, None] == np.asarray(decoders)[None, :]
    matched_correct = int(panel_e.to_numpy(dtype=bool)[diagonal_mask].sum())
    mismatched_correct = int(panel_e.to_numpy(dtype=bool)[~diagonal_mask].sum())
    passed('direction counts match prespecified 10/10 and 9/90 QA',
           matched_correct == 10 and mismatched_correct == 9)
    if any(path.endswith('crossswap_direction_summary.csv') for path in tables):
        summary = table(tables, 'crossswap_direction_summary.csv').set_index('matched')
        counts = direction.assign(matched=direction.matched.astype(bool)).groupby('matched').agg(
            n_pairs=('both_correct', 'size'), both_correct=('both_correct', 'sum'),
            right_correct=('right_correct', 'sum'), left_correct=('left_correct', 'sum'),
            Q_correct=('Q_correct', 'sum'))
        summary = summary.sort_index().loc[[False, True], ['n_pairs', 'both_correct', 'right_correct', 'left_correct', 'Q_correct']]
        counts = counts.sort_index().loc[[False, True], ['n_pairs', 'both_correct', 'right_correct', 'left_correct', 'Q_correct']]
        passed('direction-summary derived-table agreement',
               np.array_equal(counts.to_numpy(dtype=int), summary.to_numpy(dtype=int)))
    panel_e = panel_e.astype(bool).rename_axis(index='network_id', columns='decoder_id').stack().rename(
        'both_correct').reset_index()
    qa.extend([
        dict(check='direction matched both-correct count = 10', passed=True),
        dict(check='direction mismatched both-correct count = 9', passed=True),
    ])
    return {
        'panel_b_intervention': panel_b.sort_values(['intervention', 'model_id']).reset_index(drop=True),
        'panel_c_physical': panel_c.sort_values(['condition', 'intervention', 'model_id']).reset_index(drop=True),
        'panel_d_pairing': panel_d,
        'panel_e_direction': panel_e,
        'fig6_QA': pd.DataFrame(qa),
    }

def primary_analysis(key,tables):
    if key in ['3','S7']:return composition_analysis(tables)
    if key in ['5','S8']:
        d=table(tables,'normalized_metrics.csv')
        return {'common_support_trajectories':d[d.optimizer_step<=100000],
                'common_100k_endpoints':d[d.optimizer_step==100000]}
    if key in ['4','S5']:
        d=table(tables,'geometry_paired.csv')
        return {'official50_fixed_phase_pairs':table(tables,'official50_phase_pairs.csv'), 'native_radial_metrics':table(tables,'radial_model_phase_metrics.csv')}
    if key == '6':
        return fig6_panel_data(tables)
    if key == 'S9':
        from .round2_analysis import robust_intervention_metrics
        calculated=robust_intervention_metrics(table(tables,'replacement_metrics.csv'))
        ref=table(tables,'robust_intervention_model_metrics.csv')
        keys=['model_id','intervention','condition'];merged=calculated.merge(ref,on=keys,suffixes=('_new','_stored'),validate='one_to_one')
        np.testing.assert_allclose(merged.normalized_change_new,merged.normalized_change_stored,atol=1e-12)
        return {'paired_noop_normalized_changes':calculated}
    return {path:df for path,df in tables.items() if path.startswith('data/precomputed/')}

def adopted_analysis(key,tables):
    result={path:df for path,df in tables.items() if path in json.loads(Path(__file__).with_name('figure_inputs.json').read_text())[key]['supporting_input_paths'] and
            not any(s in path for s in ['null_draw_metrics','normalized_metrics','paired_timecourses','threshold_crossings'])}
    if key in ['2','S5']:
        d=table(tables,'specificity_modelwise.csv')
        np.testing.assert_allclose(d.D_canonical_vs_random,d.canonical_abs_A-d.mean_random_reflection_abs_A,atol=1e-12)
    if key=='S3':
        t=table(tables,'paired_timecourses.csv')
        late=t[t.frame.between(450,499)].groupby(['model_id','group','fade_seconds','jitter_pixels'])[['A','S','G']].mean().reset_index()
        w=table(tables,'window_metrics.csv');w=w[w.window=='late']
        merged=late.merge(w,on=['model_id','group','fade_seconds','jitter_pixels'],suffixes=('_new','_stored'))
        for metric in ['A','S','G']:np.testing.assert_allclose(merged[metric+'_new'],merged[metric+'_stored'],atol=1e-9)
        result['recomputed_late_window']=late
    if key in ['4','S5']:
        paired=table(tables,'phase_paired.csv')
        stats=paired.groupby(['model_id','group'])[['A','S']].var(ddof=0).rename(columns={'A':'var_A','S':'var_S'}).reset_index()
        ref=table(tables,'model_phase_sensitivity.csv').sort_values(['model_id','group'])
        np.testing.assert_allclose(stats[['var_A','var_S']],ref[['var_A','var_S']],atol=1e-14)
        result['recomputed_phase_variance']=stats
    if key in ['6','S9']:
        d=table(tables,'robust_intervention_model_metrics.csv')
        np.testing.assert_allclose(d.normalized_change*d.G_noop,d.intervention_value-d.baseline,atol=1e-12)
    if key=='S10':
        raw=table(tables,'crossswap_metrics.csv'); ss=[]
        for (group,window),df in raw.groupby(['group','window']):
            metric='Q_motion' if group=='motion_grating' else 'A'
            if metric not in df or df[metric].isna().any():continue
            y=df.pivot(index='network_id',columns='decoder_id',values=metric).to_numpy()
            row=y.mean(1); col=y.mean(0); grand=y.mean()
            values=[len(col)*np.sum((row-grand)**2),len(row)*np.sum((col-grand)**2),np.sum((y-row[:,None]-col[None,:]+grand)**2)]
            for effect,value in zip(['network','decoder','residual'],values):ss.append(dict(group=group,window=window,metric=metric,effect=effect,fraction=value/sum(values)))
        ss=pd.DataFrame(ss); ref=table(tables,'decomposition.csv')
        merged=ss.merge(ref,on=['group','window','metric','effect'],suffixes=('_new','_stored'))
        np.testing.assert_allclose(merged.fraction_new,merged.fraction_stored,atol=1e-12)
        result['recomputed_centered_SS']=ss
    return result

def statistical_tables(tables):
    """Frozen model/seed intervals and corrected tests retain their source identity."""
    return {p:d for p,d in tables.items() if any(x in p for x in ['bootstrap','control_summary','threshold_summary','onsets','decomposition','statistics_corrected','comparator_summary','comparator_contrasts','reliability','primary_relations','denominator_audit','shuffle_summary'])}

def write_reproduction_record(key,root,manifest,qa,primary,adopted):
    for row in manifest.itertuples():
        assert hashlib.sha256((root/row.path).read_bytes()).hexdigest()==row.sha256, row.path
    spec=json.loads(Path(__file__).with_name('figure_inputs.json').read_text())[key]
    out=root/spec['output_dir']; prefix=spec['prefix']; summaries={}
    for category,frames in [('primary',primary),('adopted',adopted)]:
        for index,(label,df) in enumerate(frames.items()):
            if len(df)>1000:continue
            file=out/f'{prefix}_integration_{category}_{index:02d}.csv';df.to_csv(file,index=False)
            summaries[str(file.relative_to(root))]=dict(source_or_calculation=label,sha256=hashlib.sha256(file.read_bytes()).hexdigest())
    qa.to_csv(out/f'{prefix}_integration_QA.csv',index=False)
    paths=[out/(prefix+'_main.pdf'),out/(prefix+'_main.png')]
    record=dict(canonical_notebook=spec['notebook'],inference_performed=False,
        revision_notebook_execution_required=False,inputs_sha256=dict(zip(manifest.path,manifest.sha256)),
        source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [root/'src/flyvis_midd/figure_analysis.py',root/'src/flyvis_midd/figure_rendering.py',root/'src/flyvis_midd/figure_inputs.json',root/'src/flyvis_midd/round2_analysis.py']},
        outputs_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},summaries=summaries,
        software=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__),
        statistics=('No inferential tests, confidence intervals, or bootstrap were computed; model is the unit for intervention observations and network is the unit for matched/mismatched |A|, with nine mismatched decoders averaged within each network.' if key=='6' else 'Model/seed units only. Original scalar fit/bootstrap replay and stronger comparators use shared calculations; fixed-design intervals retain their source configs. Phase-grid intervals are conditional.'),
        limitation='Fig1B uses retained rasters; complete representative raw vectors are unavailable.' if key=='1' else '')
    path=out/f'{prefix}_integration_manifest.json';path.write_text(json.dumps(record,indent=2)+'\n');return record
