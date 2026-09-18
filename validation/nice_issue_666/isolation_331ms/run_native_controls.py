"""Read-only diagnosis of production adapter; isolated native input experiments."""
import json, re, shutil, subprocess
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from vaft.process.electromagnetics import compute_point_response_matrices
from vaft.formula.magnetics import project_poloidal_field

BASE=Path('/tmp/nice-666-corrected-v5/focus')
ROOT=Path('/tmp/nice331-isolation-20260909')
ROOT.mkdir(exist_ok=True)
manifest=json.loads((BASE/'nice_case_manifest.json').read_text())
channels=[c for c in manifest['diagnostic_channels'] if c['enabled'] and c['family'] in ('bpol_probe','flux_loop')]
ip=float(np.loadtxt(BASE/'input/Ip_B0.txt')[0])

def response(r,z,currents):
    rows=[]
    for c in channels:
        g=c['geometry']; p=g if c['family']=='bpol_probe' else g['positions'][0]
        psi,bz,br=compute_point_response_matrices([p['r']],[p['z']],r,z,turns=np.ones(len(r)),components=('psi','bz','br'))
        v=project_poloidal_field(br[0],bz[0],g['poloidal_angle']) if c['family']=='bpol_probe' else psi[0]
        rows.append(v @ currents)
    return np.array(rows)

synthetic=response(np.array([.4]),np.array([0.]),np.array([ip]))
def run(name,params,synthetic_data=False):
    case=ROOT/name
    if case.exists(): raise RuntimeError(f'refusing stale case {case}')
    shutil.copytree(BASE/'input',case/'input'); (case/'output').mkdir();(case/'restart').mkdir()
    tree=ET.parse(case/'input/param.xml'); root=tree.getroot()
    for key,value in params.items():
        el=root.find(key)
        if el is None:el=ET.SubElement(root,key)
        el.text=str(value)
    tree.write(case/'input/param.xml')
    if synthetic_data:
        n=int((case/'input/Icoils.txt').read_text().split()[0])
        (case/'input/Icoils.txt').write_text(str(n)+'\n'+'0\n'*n)
        for family,file,sign in [('bpol_probe','Bprobes_meas.txt',1),('flux_loop','fluxloops_meas.txt',-1)]:
            pairs=[(c,y) for c,y in zip(channels,synthetic) if c['family']==family]
            (case/'input'/file).write_text(str(len(pairs))+'\n'+''.join(f'{sign*y:.17g} {c["uncertainty"]:.17g} 0\n' for c,y in pairs))
    with (case/'stdout.log').open('w') as out,(case/'stderr.log').open('w') as err:
        result=subprocess.run(['/tmp/nice-build-clang2/nice_recon'],cwd=case,stdout=out,stderr=err,timeout=30)
    log=(case/'stdout.log').read_text()
    summary={'returncode':result.returncode,'parameters_changed':params,'synthetic':synthetic_data,
       'barycenter':re.findall(r'Barycenter = \(([^)]+)\)',log),
       'ip_vacth':re.findall(r'Ip from TH = ([^\s]+)',log),
       'direct':re.findall(r'Ip=[^\n]+|[<>]+ direct[^\n]+',log),
       'last_residuals':re.findall(r'relresidX=([^\s]+)',log)[-4:],
       'costs':re.findall(r'costM?=[^\n]+',log),
       'invalid':bool(re.search(r'plasma valid = 0|IsValid\(\)=0',log))}
    if synthetic_data:
        fit=[]
        for family,file in [('bpol_probe','Bprobes'),('flux_loop','fluxloops')]:
            comp=np.loadtxt(case/'output'/f'vacth_{file}_comp.txt',ndmin=1)
            expected=np.array([y for c,y in zip(channels,synthetic) if c['family']==family])
            sigma=np.array([c['uncertainty'] for c in channels if c['family']==family])
            fit.append({'family':family,'rms':float(np.sqrt(np.mean((comp-expected)**2))),
                        'max_sigma':float(np.max(abs(comp-expected)/sigma))})
        summary['forward_recovery']=fit
    print(name,json.dumps(summary),flush=True)
    return summary

results={}
for name,params,syn in [
    ('real_vacth_only',{'algoVacTHonly':1},False),
    ('synthetic_vacth_only',{'algoVacTHonly':1},True),
    ('real_direct1',{'iterMaxDirInitRecon':1},False),
    ('real_direct10',{'iterMaxDirInitRecon':10},False),
    ('real_picard2',{'algoDirect':0},False),
]:
    results[name]=run(name,params,syn)
(ROOT/'summary.json').write_text(json.dumps(results,indent=2))
