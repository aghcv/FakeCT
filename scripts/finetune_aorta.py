#!/usr/bin/env python3
"""Run a fixed family-held-out pretrained/baseline/sibling segmentation comparison."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fakect_pretrained_study import load_config,plan,inspect,prepare_case,smoke,fit,evaluate,report


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
 p.add_argument('--stage',choices=('plan','list','prepare','smoke','fit','evaluate','report'),default='plan')
 p.add_argument('--case-id');p.add_argument('--fold');p.add_argument('--arm',choices=('pretrained','baseline_only','siblings'))
 p.add_argument('--ids-only',action='store_true');p.add_argument('--revision',default='review-r1');p.add_argument('--validate-only',action='store_true')
 a=p.parse_args();path=Path(a.config);c=load_config(path);content=path.read_bytes()
 if a.validate_only:
  result=plan(c,content,validate_only=True)
 elif a.stage=='plan':
  lock=plan(c,content);result={'cases':lock['cases'],'folds':[f['id'] for f in lock['folds']]}
 else:
  lock=inspect(c,content,software=a.stage in ('smoke','fit','evaluate'))
  if a.stage=='list':
   ids=[f['id'] for f in lock['folds']] if a.fold else lock['cases']
   if a.ids_only:print('\n'.join(ids));return
   result={'cases':lock['cases'],'folds':[f['id'] for f in lock['folds']]}
  elif a.stage=='prepare':result=prepare_case(c,lock,a.case_id)
  elif a.stage=='smoke':result=smoke(c,lock)
  elif a.stage=='fit':result=fit(c,lock,a.fold,a.arm)
  elif a.stage=='evaluate':result=evaluate(c,lock,a.fold,a.arm)
  else:result=report(c,lock,a.revision)
 print(json.dumps(result,indent=2,default=str))


if __name__=='__main__':main()
