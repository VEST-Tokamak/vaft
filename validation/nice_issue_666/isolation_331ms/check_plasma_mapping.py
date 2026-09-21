import json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import lsq_linear
from vaft.process.electromagnetics import compute_point_response_matrices
from vaft.formula.magnetics import project_poloidal_field
from matplotlib.tri import Triangulation

base=Path('/tmp/nice-666-corrected-v5/focus'); out=Path('/tmp/nice331-isolation-20260909')
p=json.loads((base/'nice_case_manifest.json').read_text())
cs=[c for c in p['diagnostic_channels'] if c['enabled'] and c['family'] in ('bpol_probe','flux_loop')]
audit={a['ods_path']:a for a in p['active_response_audit']}
rr,zz=np.meshgrid(np.linspace(.16,.72,15),np.linspace(-.52,.52,17))
rr,zz=rr.ravel(),zz.ravel()
full='--full-support' in sys.argv
if full:
 from matplotlib.path import Path as Polygon
 rr,zz=np.meshgrid(np.linspace(.11,.75,25),np.linspace(-1.17,1.17,49))
 limiter=np.loadtxt(base/'input/limiter.txt',skiprows=1)
 inside=Polygon(limiter).contains_points(np.c_[rr.ravel(),zz.ravel()])
 rr,zz=rr.ravel()[inside],zz.ravel()[inside]
rows=[]
for c in cs:
 g=c['geometry'];loc=g if c['family']=='bpol_probe' else g['positions'][0]
 psi,bz,br=compute_point_response_matrices([loc['r']],[loc['z']],rr,zz,turns=np.ones(len(rr)),components=('psi','bz','br'))
 rows.append(project_poloidal_field(br[0],bz[0],g['poloidal_angle']) if c['family']=='bpol_probe' else psi[0])
A=np.array(rows); y=np.array([c['value']-audit[c['ods_path']]['native'] for c in cs]);sigma=np.array([c['uncertainty'] for c in cs])
ip=next(c for c in p['diagnostic_channels'] if c['family']=='plasma_current')
result={}
for family in ('bpol_probe','flux_loop','both'):
 mask=np.array([family=='both' or c['family']==family for c in cs]); aa=A[mask]*ip['value']/sigma[mask,None];bb=y[mask]/sigma[mask]
 aa=np.vstack((aa,np.ones(len(rr))*ip['value']/ip['uncertainty']));bb=np.r_[bb,ip['value']/ip['uncertainty']]
 fit=lsq_linear(aa,bb,bounds=(0,np.inf),method='bvls',tol=1e-10,max_iter=1000)
 res=(A@fit.x*ip['value']-y)/sigma
 result[family]={'success':bool(fit.success),'ip_A':float(sum(fit.x)*ip['value']),
   'fit_bpol_rms_sigma':float(np.sqrt(np.mean(res[:63]**2))),
   'fit_flux_rms_sigma':float(np.sqrt(np.mean(res[63:]**2))),
   'largest_residuals':[{'path':cs[i]['ods_path'],'residual_sigma':float(res[i]),'observed_plasma':float(y[i])} for i in np.argsort(abs(res))[-5:]]}

# Independent FE response audit of a known 125.8 kA loop at (.4,0).
nodes=np.loadtxt(base/'output/mesh_coord.txt'); tri=np.loadtxt(base/'output/mesh_triangles.txt',dtype=int)
# NICE OutputTxt writes zero-based triangle indices.
triang=Triangulation(nodes[:,0],nodes[:,1],tri)
finder=triang.get_trifinder()
psi,bz,br=compute_point_response_matrices(nodes[:,0],nodes[:,1],[.4],[0.],turns=[1],components=('psi','bz','br'))
u=psi[:,0]*ip['value']/(2*np.pi)
fe=[]
for c in cs:
 g=c['geometry'];loc=g if c['family']=='bpol_probe' else g['positions'][0];r,z=loc['r'],loc['z']; t=int(finder(r,z))
 if t<0: raise ValueError('sensor outside mesh')
 ns=tri[t];coef=np.linalg.solve(np.column_stack((nodes[ns],np.ones(3))),u[ns])
 ps,bzz,brr=compute_point_response_matrices([r],[z],[.4],[0.],turns=[1],components=('psi','bz','br'))
 if c['family']=='bpol_probe':
  finite_element=project_poloidal_field(-coef[1]/r,coef[0]/r,g['poloidal_angle'])
  exact=project_poloidal_field(brr[0,0],bzz[0,0],g['poloidal_angle'])*ip['value']
 else:
  finite_element=(coef@[r,z,1])*2*np.pi;exact=ps[0,0]*ip['value']
 fe.append({'path':c['ods_path'],'family':c['family'],'exact':float(exact),'finite_element':float(finite_element),'error_sigma':float((finite_element-exact)/c['uncertainty'])})
result['finite_element_response']=fe
(out/('plasma_mapping_full.json' if full else 'plasma_mapping.json')).write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='finite_element_response'},indent=2))
for f in ('bpol_probe','flux_loop'):
 rows=[x for x in fe if x['family']==f];print(f,'FE max sigma',max(abs(x['error_sigma']) for x in rows),'relative norm error',np.linalg.norm([x['finite_element']-x['exact'] for x in rows])/np.linalg.norm([x['exact'] for x in rows]))
