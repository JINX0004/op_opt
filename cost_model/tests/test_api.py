from copy import deepcopy
from pathlib import Path
import json,sys,unittest
from matmul_cost_model import MatmulCostModel,make_config,expand_config,generate_candidates,supported_tilings

HERE=Path(__file__).resolve().parent

class APITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=MatmulCostModel()
        cls.smoke=json.loads((HERE/'data/smoke.json').read_text())

    def test_frozen_eight_calibrated_and_semantic_values(self):
        for r in self.smoke:
            with self.subTest(config=r['config']):
                a=self.model.predict(r['config']);b=self.model.predict(r['config'],semantic=True)
                self.assertAlmostEqual(a['predicted_us'],r['expected_us'],places=7)
                self.assertAlmostEqual(b['predicted_us'],r['expected_semantic_us'],places=7)
                self.assertAlmostEqual(a['predicted_ms']*1000,a['predicted_us'])
                self.assertFalse(a['target_measurements_required'])
                self.assertFalse(a['hardware_execution_performed'])

    def test_plan_roundtrip_aliases_and_no_input_mutation(self):
        raw=deepcopy(self.smoke[3]['config']);before=deepcopy(raw)
        raw['implementation']='ops_nn_deterministic_split_k';raw['dtype']='fp16'
        expanded=expand_config(raw);copy=deepcopy(expanded)
        a=self.model.predict(raw);b=self.model.predict(expanded)
        self.assertEqual(a['predicted_us'],b['predicted_us'])
        self.assertEqual(expanded,copy)
        raw['implementation']=before['implementation'];raw['dtype']=before['dtype']
        self.assertEqual(raw,before)

    def test_case_id_does_not_change_prediction(self):
        raw=deepcopy(self.smoke[1]['config']);a=self.model.predict(raw)
        raw['case_id']='unrelated_ID';b=self.model.predict(raw)
        self.assertEqual(a['predicted_us'],b['predicted_us'])

    def test_launch_and_effective_cores_are_distinct(self):
        r=self.model.predict(self.smoke[1]['config'])
        self.assertEqual(r['cores'],8);self.assertEqual(r['active_compute_cores'],1)
        d=self.model.predict(self.smoke[3]['config']);self.assertEqual(d['active_compute_cores'],24)

    def test_unqualified_inputs_rejected(self):
        original=self.smoke[3]['config']
        edits=[{'dtype':'float32'},{'trans_b':True},{'trans_a':1},{'batch':2},{'cores':0},{'cores':True},
               {'cores':1},{'shape':[16,48,102400]},{'shape':[15,16,102400]},
               {'shape':[16,16,32]},{'source_a_offset_extra_bytes':128},
               {'execution_profile':'cold'},{'measured_us':100}, {'tiling':{'base_m':64,'base_n':64,'base_k':128,'step_k':2}}]
        for edit in edits:
            with self.subTest(edit=edit):
                with self.assertRaises(ValueError):expand_config(dict(original,**edit))

    def test_contradictory_complete_plan_rejected(self):
        original=expand_config(self.smoke[3]['config'])
        changes=[lambda r:r['plan']['tcube'].update(stepKa=3),
                 lambda r:r['plan']['raw_words'].__setitem__(0,r['plan']['raw_words'][0]+1),
                 lambda r:r['plan'].update(dtype='bfloat16'),
                 lambda r:r['plan']['tcube'].update(baseM=128.0)]
        for change in changes:
            r=deepcopy(original);change(r)
            with self.assertRaises(ValueError):expand_config(r)

    def test_explicit_catlass_dimensions(self):
        r=make_config('catlass',[16,16,102400],'fp16',16,{'l1_tile':[16,16,4096],'l0_tile':[16,16,1024]})
        self.assertEqual(r['tile'],'deepk')
        self.assertEqual(self.model.predict(r)['predicted_us'],self.smoke[0]['expected_us'])

    def test_candidates_and_comparison_grouping(self):
        rows=generate_candidates([64,128,512],'bf16',('base','deterministic'),(1,2))
        self.assertTrue(rows)
        result=self.model.compare(rows,top=2)
        self.assertEqual(len(result['groups']),2)
        for group in result['groups']:
            values=[r['predicted_us'] for r in group['ranking']]
            self.assertEqual(values,sorted(values))
        with self.assertRaises(ValueError):self.model.compare([rows[0],rows[0]])

    def test_template_contract_and_namespace_isolation(self):
        self.assertEqual(len(supported_tilings()['catlass']),4)
        self.assertNotIn('costmodel',sys.modules)
        self.assertNotIn('design',sys.modules)
        self.assertFalse(any(name.startswith(('numpy','scipy','torch','acl')) for name in sys.modules))

if __name__=='__main__':unittest.main()
