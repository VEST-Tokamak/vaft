import json, re, shutil, subprocess
from pathlib import Path
import numpy as np

src=Path('/tmp/nice331-cocos-input-v2-20260914/focus')
root=Path('/tmp/nice331-cocos-matrix-v2-20260914')
root.mkdir(exist_ok=True)

def flip_values(path):
    lines=path.read_text().splitlines()
    out=[lines[0]]
    for line in lines[1:]:
        fields=line.split(); fields[0]=f'{-float(fields[0]):.17g}'; out.append(' '.join(fields))
    path.write_text('\n'.join(out)+'\n')

results={}
for bp_sign in (1,-1):
  for fl_sign in (1,-1):
    name=f'bp_{bp_sign:+d}_fl_{fl_sign:+d}'.replace('+','p').replace('-','m')
    case=root/name
    if case.exists(): raise RuntimeError(f'stale {case}')
    shutil.copytree(src/'input',case/'input'); (case/'output').mkdir(); (case/'restart').mkdir()
    text=(case/'input/param.xml').read_text().replace('<algoVacTHonly>0</algoVacTHonly>','<algoVacTHonly>1</algoVacTHonly>')
    (case/'input/param.xml').write_text(text)
    if bp_sign<0: flip_values(case/'input/Bprobes_meas.txt')
    if fl_sign<0: flip_values(case/'input/fluxloops_meas.txt')
    with (case/'stdout.log').open('w') as out,(case/'stderr.log').open('w') as err:
      r=subprocess.run(['/tmp/nice-cocos-build-20260914/nice_recon'],cwd=case,stdout=out,stderr=err,timeout=30)
    log=(case/'stdout.log').read_text()
    def one(pattern):
      m=re.findall(pattern,log); return m[-1] if m else None
    results[name]={'bp_multiplier_on_prepared_file':bp_sign,'flux_multiplier_on_prepared_file':fl_sign,
      'returncode':r.returncode,'barycenter':one(r'Barycenter = \(([^)]+)\)'),
      'ip_vacth_A':float(one(r'Ip from TH = ([^\s]+)')),
      'rms_B_T':float(one(r'residu_B \(rms\) =([^\s]+)')),
      'rms_flux_Wb':float(one(r'residu_Psi \(rms\) =([^\s]+)')),
      'cost_B':float(one(r'cost_B=([^\s]+)')),'cost_flux':float(one(r'cost_Psi=([^\s]+)'))}
(root/'summary.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
