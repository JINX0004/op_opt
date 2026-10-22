"""Integrity, unit tests and optional full 1198-case frozen parity/accuracy replay."""
from pathlib import Path
import argparse,hashlib,json,sys,time,unittest
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--full',action='store_true');p.add_argument('--output',type=Path)
    p.add_argument('--forbid-root',type=Path,help='Optional isolation audit: deny reading this tree outside the current copy')
    a=p.parse_args();start=time.perf_counter()
    if a.forbid_root:
        forbidden_root=a.forbid_root.resolve()
        def audit(event,args):
            if event=='open' and isinstance(args[0],(str,bytes)):
                mode=args[1]
                if isinstance(mode,str) and any(flag in mode for flag in 'wax') and '+' not in mode:
                    return  # The report may be written outside the audited copy.
                path=Path(args[0]).resolve()
                if path.is_relative_to(forbidden_root) and not path.is_relative_to(ROOT):
                    raise AssertionError('Read outside extracted copy: '+str(path))
        sys.addaudithook(audit)
    manifest=json.loads((ROOT/'MANIFEST.json').read_text())
    for name,digest in manifest['files'].items():
        path=(ROOT/name).resolve()
        assert path.is_relative_to(ROOT),name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,'Integrity mismatch: '+name
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():return 1
    report=dict(unit_tests=result.testsRun,all_unit_tests_passed=True,manifest_files=len(manifest['files']),
                fixed_model_sha256=manifest['model_sha256'],hardware_execution_performed=False)
    if a.full:
        import gzip,socket,subprocess
        from matmul_cost_model import MatmulCostModel
        from matmul_cost_model.evaluation import evaluate
        model=MatmulCostModel()
        def forbidden(*args,**kwargs):raise AssertionError('network/process execution forbidden during standalone inference')
        socket.socket=socket.create_connection=subprocess.Popen=subprocess.run=forbidden
        predictions=[];measurements={};maxerr=0.
        with gzip.open(ROOT/'tests/data/independent_validation.jsonl.gz','rt',encoding='utf-8') as f:cases=[json.loads(line) for line in f]
        for i,r in enumerate(cases):
            v=model.predict(r['config']);delta=abs(v['predicted_us']-r['expected_us']);maxerr=max(maxerr,delta)
            assert delta<1e-7,(v['case_id'],delta)
            predictions.append(v);measurements[v['case_id']]=r['process_us']
            if (i+1)%200==0:print('Frozen parity: '+str(i+1)+'/'+str(len(cases)),file=sys.stderr,flush=True)
        scores=evaluate(predictions,measurements)
        info=json.loads((ROOT/'matmul_cost_model/data/model_info.json').read_text())
        for dtype,values in info['accuracy_scores'].items():
            s=scores['per_dtype'][dtype]
            assert abs(s['mape']-values['mape'])<1e-12
            assert abs(s['max_ape']-values['max_ape'])<1e-12
            for label in ('strict','stable5'):
                assert s[label]==values['orders'][label],(dtype,label)
            assert s['practical_top1']==values['practical_top1']
        for name,module in list(sys.modules.items()):
            if name=='matmul_cost_model' or name.startswith('matmul_cost_model.'):
                if getattr(module,'__file__',None):assert Path(module.__file__).resolve().is_relative_to(ROOT),(name,module.__file__)
        assert 'costmodel' not in sys.modules and 'design' not in sys.modules
        assert not any(n.split('.')[0] in ('numpy','scipy','torch','acl','paramiko') for n in sys.modules)
        report.update(frozen_cases=len(cases),max_prediction_difference_us=maxerr,network_and_process_execution_disabled=True,
                      all_inference_modules_inside_this_copy=True,no_third_party_runtime_modules=True,
                      independent_metrics={d:{k:s[k] for k in ('conditions','mape','max_ape','within5','strict','stable5','practical_top1')} for d,s in scores['per_dtype'].items()})
    report['CPU_seconds']=time.perf_counter()-start
    text=json.dumps(report,ensure_ascii=False,indent=2)
    if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(text+'\n',encoding='utf-8')
    print(text);return 0

if __name__=='__main__':raise SystemExit(main())
