import unittest
from matmul_cost_model.evaluation import evaluate

class EvaluationTests(unittest.TestCase):
    def predictions(self):
        return [dict(case_id='a',implementation='base',shape=[16,16,1024],dtype='float16',predicted_us=12.),
                dict(case_id='b',implementation='base',shape=[16,16,1024],dtype='float16',predicted_us=11.)]

    def test_stable_reversal_and_mape(self):
        r=evaluate(self.predictions(),{'a':[10.,10.,10.],'b':[20.,20.,20.]})
        self.assertAlmostEqual(r['overall']['mape'],.325)
        self.assertAlmostEqual(r['overall']['max_ape'],.45)
        self.assertEqual(r['overall']['stable5']['inversions'],1)
        self.assertEqual(r['overall']['practical_top1']['max_raw_regret'],1.)
        self.assertFalse(r['parameters_updated'])

    def test_one_run_does_not_claim_stable_sort_quality(self):
        r=evaluate(self.predictions(),{'a':[10.],'b':[20.]})
        self.assertIsNone(r['overall']['stable5'])
        self.assertIsNone(r['overall']['practical_top1'])

    def test_near_tie_strict_retained_stable_excluded(self):
        r=evaluate(self.predictions(),{'a':[10.,10.,10.],'b':[10.1,10.1,10.1]})
        self.assertEqual(r['overall']['strict']['inversions'],1)
        self.assertEqual(r['overall']['stable5']['pairs'],0)

    def test_id_and_invalid_values_rejected(self):
        for times in ({'a':[10.]},{'a':[10.],'b':[0.]},{'a':[10.],'b':[float('nan')]}):
            with self.assertRaises(ValueError):evaluate(self.predictions(),times)

if __name__=='__main__':unittest.main()
