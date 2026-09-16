import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

parser = argparse.ArgumentParser(description='Bounded fakenoise SLURM wrapper checks using shell mocks only.')
parser.add_argument('--repo', type=Path, default=Path.cwd(), help='Checkout containing scripts/slurm/fakenoise_train_gpu.sh')
parser.add_argument('--output', type=Path, default=Path('outputs/evaluation/slurm-wrapper-results.json'))
options = parser.parse_args()
root = options.repo.resolve()
script = root / 'scripts/slurm/fakenoise_train_gpu.sh'
wrapper = r'''
module() {
    printf '%s\0' "$@" > "$MOCK_MODULE_LOG"
    return "${MOCK_MODULE_STATUS:-0}"
}
crun() {
    printf '%s\0' "$@" > "$MOCK_CRUN_LOG"
    return "${MOCK_CRUN_STATUS:-0}"
}
export -f module crun
bash "$@"
'''
results=[]
cases=[
 ('default_local_root', ['--csv','/tmp/pairs.csv','--out-dir','/tmp/run'], {},0,True),
 ('paths_with_spaces', ['--csv','/tmp/data source/pairs.csv','--out-dir','/tmp/output run','--context','2'], {'FAKECT_ENV_PREFIX':'/tmp/env prefix'},0,True),
 ('explicit_root', ['--csv','/tmp/pairs.csv','--out-dir','/tmp/run'], {'FAKECT_ROOT':str(root),'SLURM_SUBMIT_DIR':'/tmp/invalid'},0,True),
 ('submit_dir_root', ['--csv','/tmp/pairs.csv','--out-dir','/tmp/run'], {'SLURM_SUBMIT_DIR':str(root)},0,True),
 ('missing_csv', ['--out-dir','/tmp/run'], {},2,False),
 ('missing_out_dir', ['--csv','/tmp/pairs.csv'], {},2,False),
 ('bad_root', ['--csv','/tmp/pairs.csv','--out-dir','/tmp/run'], {'FAKECT_ROOT':'/tmp/fakect-does-not-exist'},2,False),
 ('module_failure', ['--csv','/tmp/pairs.csv','--out-dir','/tmp/run'], {'MOCK_MODULE_STATUS':'7'},7,False),
 ('crun_failure', ['--csv','/tmp/pairs.csv','--out-dir','/tmp/run'], {'MOCK_CRUN_STATUS':'13'},13,True),
 ('equals_options', ['--csv=/tmp/pairs.csv','--out-dir=/tmp/run'], {},0,True),
 ('missing_csv_but_token_in_out_path', ['--out-dir','/tmp/output --csv run'], {},2,False),
 ('missing_out_value', ['--csv','/tmp/pairs.csv','--out-dir'], {},2,False),
 ('empty_csv_equals', ['--csv=','--out-dir=/tmp/run'], {},2,False),
 ('empty_out_equals', ['--csv=/tmp/pairs.csv','--out-dir='], {},2,False),
 ('csv_value_is_next_option', ['--csv','--out-dir','/tmp/run'], {},2,False),
 ('mixed_csv_equals', ['--csv=/tmp/pairs.csv','--out-dir','/tmp/run'], {},0,True),
 ('mixed_out_equals', ['--csv','/tmp/pairs.csv','--out-dir=/tmp/run'], {},0,True),
 ('empty_csv_argument', ['--csv','','--out-dir','/tmp/run'], {},2,False),
]
for name,args,overrides,expected,expect_crun in cases:
 with tempfile.TemporaryDirectory(prefix='fakect-slurm-test-') as tmp:
  env=dict(os.environ)
  for key in ['FAKECT_ROOT','SLURM_SUBMIT_DIR','FAKECT_ENV_PREFIX','MOCK_MODULE_STATUS','MOCK_CRUN_STATUS','BASH_ENV','ENV']:
   env.pop(key,None)
  env.update(overrides)
  env['MOCK_MODULE_LOG']=str(Path(tmp)/'module.args')
  env['MOCK_CRUN_LOG']=str(Path(tmp)/'crun.args')
  done=subprocess.run(['bash','-c',wrapper,'mock-wrapper',str(script),*args],env=env,cwd='/tmp',capture_output=True,text=True,timeout=10)
  module_path=Path(env['MOCK_MODULE_LOG'])
  module_argv=module_path.read_bytes().decode().split('\0')[:-1] if module_path.exists() else None
  crun_path=Path(env['MOCK_CRUN_LOG'])
  argv=crun_path.read_bytes().decode().split('\0')[:-1] if crun_path.exists() else None
  assert done.returncode==expected,(name,done.returncode,done.stderr)
  assert (argv is not None)==expect_crun,name
  assert (module_argv is not None)==(expected != 2),name
  if module_argv is not None:
   assert module_argv==['load','container_env','tensorflow-gpu/2.17'],(name,module_argv)
  if argv:
   assert argv[-len(args):]==args,(name,argv,args)
   assert argv[0]=='-p'
   assert argv[2]=='python'
   assert argv[3]==str(root/'src/fakenoise.py')
   assert argv[4:10]==['--mode','train','--context','15','--context-step','4']
  results.append({'case':name,'exit_code':done.returncode,'module_called':module_argv is not None,'crun_called':argv is not None,'arguments_preserved': True if argv else None,'stderr':done.stderr.strip()})
options.output.parent.mkdir(parents=True, exist_ok=True)
options.output.write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
