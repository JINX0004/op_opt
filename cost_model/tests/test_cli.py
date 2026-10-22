from contextlib import redirect_stdout,redirect_stderr
from io import StringIO
from pathlib import Path
import json,unittest
from matmul_cost_model.cli import main

ROOT=Path(__file__).resolve().parents[1]

class CLITests(unittest.TestCase):
    def call(self,args):
        out=StringIO();err=StringIO()
        with redirect_stdout(out),redirect_stderr(err):code=main(args)
        return code,out.getvalue(),err.getvalue()

    def test_prediction_flags(self):
        c,out,err=self.call(['predict','--shape','16','16','102400','--dtype','fp16','--implementation','deterministic',
                             '--cores','24','--base-m','128','--base-n','128','--step-k','2'])
        self.assertEqual(c,0,err);self.assertAlmostEqual(json.loads(out)['predicted_us'],91.98508811412263)

    def test_bad_tiling_is_clear_error(self):
        c,out,err=self.call(['predict','--shape','16','16','102400','--implementation','base','--cores','8',
                            '--base-m','64','--base-n','64','--step-k','4'])
        self.assertEqual(c,2);self.assertIn('unreviewed',err);self.assertFalse(out)

    def test_json_does_not_silently_ignore_override_flags(self):
        c,out,err=self.call(['predict','--input',str(ROOT/'examples/fp16_deterministic.json'),'--dtype','bf16'])
        self.assertEqual(c,2);self.assertIn('cannot be combined',err)

    def test_compare_and_evaluate_examples(self):
        c,out,err=self.call(['compare','--input',str(ROOT/'examples/candidate_comparison.json'),'--top','2'])
        self.assertEqual(c,0,err);self.assertEqual(json.loads(out)['total_candidates'],12)
        c,out,err=self.call(['evaluate','--input',str(ROOT/'examples/candidate_comparison.json'),
                             '--measurements',str(ROOT/'examples/measured_times.csv')])
        self.assertEqual(c,0,err);self.assertEqual(json.loads(out)['overall']['conditions'],12)

if __name__=='__main__':unittest.main()
