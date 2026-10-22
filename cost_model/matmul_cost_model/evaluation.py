"""Offline comparison with supplied measurements; never calibrates the model."""
from collections import defaultdict
from math import isfinite
from statistics import mean,median
import csv

def read_measurements(path):
    """CSV columns: case_id,measured_us[,process_index]; repeated processes allowed."""
    values=defaultdict(list);seen=set()
    with open(path,encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if not {'case_id','measured_us'}<=set(reader.fieldnames or []):raise ValueError('CSV needs case_id,measured_us; units must be microseconds')
        for row in reader:
            cid=row['case_id'].strip();value=float(row['measured_us'])
            pi=row.get('process_index','').strip() or str(len(values[cid]))
            if not cid or not isfinite(value) or value<=0:raise ValueError('nonempty case_id and positive finite measured_us required')
            if (cid,pi) in seen:raise ValueError('duplicate measurement case_id/process_index: '+cid+'/'+pi)
            seen.add((cid,pi));values[cid].append(value)
    return dict(values)

def _scores(rows):
    errors=[r['ape'] for r in rows];groups=defaultdict(list)
    for r in rows:groups[(r['implementation'],tuple(r['shape']),r['dtype'])].append(r)
    strict=dict(pairs=0,inversions=0,prediction_ties=0)
    stable=dict(pairs=0,inversions=0,prediction_ties=0)
    repeated_pairs=0;top_ok=0;regrets=[];evidence=[]
    for items in groups.values():
        for i,a in enumerate(items):
            for b in items[i+1:]:
                fast,slow=sorted((a,b),key=lambda r:r['actual_us']);gap=slow['actual_us']/fast['actual_us']-1
                if gap<=0:continue
                wrong=fast['predicted_us']>slow['predicted_us']+1e-9
                tie=abs(fast['predicted_us']-slow['predicted_us'])<=1e-9
                strict['pairs']+=1;strict['inversions']+=wrong;strict['prediction_ties']+=tie
                enough=min(len(fast['process_us']),len(slow['process_us']))>=3
                repeated_pairs+=enough
                if enough and gap>.05 and max(fast['process_us'])<min(slow['process_us']):
                    stable['pairs']+=1;stable['inversions']+=wrong;stable['prediction_ties']+=tie
                    if wrong or tie:evidence.append(dict(fast=fast['case_id'],slow=slow['case_id'],actual_gap=gap,prediction_tie=tie))
        selected=min(items,key=lambda r:r['predicted_us']);best=min(items,key=lambda r:r['actual_us'])
        regrets.append(selected['actual_us']/best['actual_us']-1)
        dominated=any(min(len(r['process_us']),len(selected['process_us']))>=3 and max(r['process_us'])<min(selected['process_us']) and selected['actual_us']/r['actual_us']>1.05 for r in items)
        top_ok+=not dominated
    all_repeated=all(len(r['process_us'])>=3 for r in rows)
    return dict(conditions=len(rows),mape=mean(errors),max_ape=max(errors),within5=mean(e<=.05 for e in errors),
                strict=strict,stable5=stable if repeated_pairs or all_repeated else None,
                all_cases_have_three_processes=all_repeated,
                practical_top1=dict(groups=len(groups),selected_not_practically_dominated=top_ok,max_raw_regret=max(regrets)) if all_repeated else None,
                worst_stable5_evidence=sorted(evidence,key=lambda r:r['actual_gap'],reverse=True)[:20])

def evaluate(predictions, measurements):
    """Join only by case_id. Supplied timing data cannot affect predictions."""
    lookup={r['case_id']:r for r in predictions}
    if len(lookup)!=len(predictions):raise ValueError('prediction case_id values must be unique')
    if set(lookup)!=set(measurements):
        raise ValueError('prediction/measurement case_id mismatch: missing='+str(sorted(set(lookup)-set(measurements))[:5])+', extra='+str(sorted(set(measurements)-set(lookup))[:5]))
    if not lookup:raise ValueError('no predictions to evaluate')
    rows=[]
    for cid,p in lookup.items():
        times=measurements[cid]
        if not times or any(not isfinite(t) or t<=0 for t in times):raise ValueError('positive finite measurement lists required')
        predicted=float(p['predicted_us'])
        if not isfinite(predicted) or predicted<=0:raise ValueError('positive finite predictions required')
        actual=median(times)
        rows.append(dict(case_id=cid,implementation=p['implementation'],shape=p['shape'],dtype=p['dtype'],predicted_us=predicted,
                         actual_us=actual,process_us=list(times),ape=abs(predicted/actual-1)))
    return dict(overall=_scores(rows),per_dtype={d:_scores([r for r in rows if r['dtype']==d]) for d in sorted({r['dtype'] for r in rows})},
                per_implementation={k:_scores([r for r in rows if r['implementation']==k]) for k in sorted({r['implementation'] for r in rows})},
                cases=rows,parameters_updated=False,
                measurement_policy='Input is process batch-mean/median time in microseconds; aggregate processes by median. stable5 needs >=3 process values, nonoverlapping ranges and real gap>5%.')
