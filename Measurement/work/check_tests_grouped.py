"""Run unchanged test modules in fresh processes to bound Qt fixture memory."""
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
rows=[]
for path in sorted((ROOT/'tests').glob('test_*.py')):
    result=subprocess.run([sys.executable,'-m','unittest',f'tests.{path.stem}'],
        cwd=str(ROOT),capture_output=True,text=True)
    output=result.stdout+result.stderr
    match=re.search(r'Ran (\d+) tests?',output)
    row={'module':path.name,'tests':int(match.group(1)) if match else 0,'returncode':result.returncode,'output':output}
    rows.append(row)
    print(path.name,row['tests'],'PASS' if result.returncode==0 else output,flush=True)
target=ROOT/'outputs/modularization_review/grouped_tests.json'
target.write_text(json.dumps({'total':sum(r['tests'] for r in rows),'all_passed':all(r['returncode']==0 for r in rows),
    'reason':'Fresh processes bound retained Qt test fixture memory; application code and existing test bodies are unchanged.',
    'modules':rows},ensure_ascii=False,indent=2),encoding='utf-8')
sys.exit(0 if all(r['returncode']==0 for r in rows) else 1)
