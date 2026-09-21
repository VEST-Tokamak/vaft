import json, re, shutil, subprocess
from pathlib import Path
import numpy as np
import xml.etree.ElementTree as ET

src=Path('/tmp/nice331-cocos-input-v2-20260914/focus')
root=Path('/tmp/nice331-cocos-equivalence-20260914'); root.mkdir(exist_ok=True)

def case(name,cocos):
 d=root/name
 if d.exists(): raise RuntimeError(f'stale {d}')
 shutil.copytree(src/'input',d/'input');(d/'output').mkdir();(d/'restart').mkdir()
 tree=ET.parse(d/'input/param.xml');x=tree.getroot();x.find('algoVacTHonly').text='1'
 if cocos==7:
  x.find('inCOCOS').text=x.find('inoutCOCOS').text=x.find('outCOCOS').text='7'
  lines=(d/'input/Bprobes.txt').read_text().splitlines();out=[lines[0]]
  for line in lines[1:]:
   f=line.split();f[2]=f'{-float(f[2]):.17g}';out.append(' '.join(f))
  (d/'input/Bprobes.txt').write_text('\n'.join(out)+'\n')
  lines=(d/'input/fluxloops_meas.txt').read_text().splitlines();out=[lines[0]]
  for line in lines[1:]:
   f=line.split();f[0]=f'{-float(f[0]):.17g}';out.append(' '.join(f))
  (d/'input/fluxloops_meas.txt').write_text('\n'.join(out)+'\n')
 tree.write(d/'input/param.xml')
 with (d/'stdout.log').open('w') as o,(d/'stderr.log').open('w') as e:
  r=subprocess.run(['/tmp/nice-cocos-build-20260914/nice_recon'],cwd=d,stdout=o,stderr=e,timeout=30)
 return d,r.returncode

a,ra=case('cocos11',11);b,rb=case('equivalent_cocos7',7)
files=['vacth_Bprobes_meas.txt','vacth_Bprobes_comp.txt','vacth_fluxloops_meas.txt',
       'vacth_fluxloops_comp.txt','vacth_dirichlet.txt','vacth_neumann.txt']
comparison={}
for f in files:
 x=np.loadtxt(a/'output'/f,skiprows=1 if f in ('vacth_dirichlet.txt','vacth_neumann.txt') else 0)
 y=np.loadtxt(b/'output'/f,skiprows=1 if f in ('vacth_dirichlet.txt','vacth_neumann.txt') else 0)
 comparison[f]={'max_abs':float(np.max(abs(x-y))), 'relative_l2':float(np.linalg.norm(x-y)/max(np.linalg.norm(x),1e-300))}
def summary(d):
 s=(d/'stdout.log').read_text();one=lambda p:re.findall(p,s)[-1]
 return {'barycenter':one(r'Barycenter = \(([^)]+)\)'), 'ip_vacth_A':float(one(r'Ip from TH = ([^\s]+)')),
         'cost_B':float(one(r'cost_B=([^\s]+)')),'cost_flux':float(one(r'cost_Psi=([^\s]+)'))}
result={'cocos11':summary(a),'equivalent_cocos7':summary(b),'comparison':comparison,'returncodes':[ra,rb]}
(root/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
