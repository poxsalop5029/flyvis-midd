"""Round2 scientific-contract checks without model inference."""
from pathlib import Path
import sys,unittest
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
from flyvis_midd.round2_analysis import icc_matrix,assert_pairs,phase_model_table,ordered_adjacent
from flyvis_midd.stimuli import build_3midd_sequences,render_sector_image
class Round2ReliabilityTests(unittest.TestCase):
 def test_fixed_phase_absolute_and_consistency_are_distinct(self):
  y=np.arange(4)[:,None]+100*np.arange(3)[None,:]
  stats=icc_matrix(y);self.assertAlmostEqual(stats['icc_consistency'],1);self.assertLess(stats['icc_absolute'],.01)
  transformed=icc_matrix(-3*y+7)
  for metric in ['icc_absolute','icc_consistency']:self.assertAlmostEqual(stats[metric],transformed[metric])
  self.assertAlmostEqual(stats['SS_total'],stats['SS_model']+stats['SS_phase']+stats['SS_residual'])
 def test_complete_matrix_and_degenerate_behavior(self):
  for y in [np.ones((1,12)),np.array([[1,np.nan],[2,3]]),np.ones(12)]:
   with self.assertRaises(ValueError):icc_matrix(y)
  self.assertTrue(np.isnan(icc_matrix(np.ones((4,12)))['icc_absolute']))
 def test_reversal_mean_predicate_is_not_mean_reversal(self):
  to=np.tile([1.,-1.],6);tm=-to
  d=pd.DataFrame(dict(model_id=0,group='3MIDD_048',phase_index=np.arange(12),T_original=to,T_mirror=tm,A=to,S=0.,G=1.,actual_reversal=True))
  q=phase_model_table(d).iloc[0];self.assertEqual(q.phasewise_reversal_fraction,1);self.assertFalse(q.reversal_of_phase_averaged_pair);self.assertEqual(q.dominant_sign_fraction,.5)
  with self.assertRaises(AssertionError):phase_model_table(d.iloc[:-1])
 def test_pair_algebra_rejects_changed_shared_component(self):
  d=pd.DataFrame(dict(T_original=[2.],T_mirror=[-1.],A=[1.5],S=[.5],actual_reversal=[True]));assert_pairs(d)
  d.loc[0,'S']=.6
  with self.assertRaises(AssertionError):assert_pairs(d)
 def test_checkpoint_order_and_gaps_not_bridged(self):
  d=pd.DataFrame(dict(optimizer_step=[3000,0,1000,5000],A=[3.,0.,1.,5.]));q=ordered_adjacent(d)
  self.assertEqual(q.optimizer_step.tolist(),[1000]);self.assertEqual(q.previous_step.tolist(),[0])
  with self.assertRaises(ValueError):ordered_adjacent(pd.concat([d,d.iloc[[0]]]))
 def test_reflected_phase_mapping_uses_real_generator(self):
  seq=build_3midd_sequences(48);bank=[render_sector_image(seq['mirror'],phase_deg=15*j/12)[0] for j in range(12)]
  for j in range(12):
   im=render_sector_image(seq['original'],phase_deg=15*j/12)[0];ref=np.fliplr(im)
   distances=[np.mean(np.abs(ref.astype(int)-x.astype(int))) for x in bank]
   self.assertEqual(int(np.argmin(distances)),(-j)%12)
   # Away from singular sector boundaries, nonzero phases have exact matching rasters.
   if j%4:self.assertEqual(min(distances),0)
if __name__=='__main__':unittest.main()
