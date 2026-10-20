"""Discover a complete shot population, inventory attempts, and resume batch scans.

Run on a server with the canonical FileDB. Writes only inside --output. HSDS and
FileDB observations remain separate. No solver reconstruction is invoked.
"""
from __future__ import annotations
import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import time

import pandas as pd
from vaft.database._equilibrium_monitoring import equilibrium_summary, TABLE_COLUMNS
from vaft.database.filedb import FileDB
from vaft.database.utils import exist_shot


def write_json(path, value):
    staged = path.with_suffix(path.suffix + '.tmp')
    staged.write_text(json.dumps(value, sort_keys=True, indent=2, default=str) + '\n')
    staged.replace(path)


def inventory(source, db, output):
    with redirect_stdout(io.StringIO()):
        discovered = exist_shot(source=source, sort=1, strict=True)
    remote = {int(s) for s in discovered if str(s).isdigit()}
    local = set()
    for parent in (db.root/'omas/efit/magnetic', db.root/'efit/magnetic'):
        if parent.is_dir():
            local.update(int(p.name) for p in parent.iterdir() if p.is_dir() and p.name.isdigit())
    shots = sorted(remote | local)
    rows, attempts = [], []
    for shot in shots:
        manifest_path = db.omas_manifest('efit',shot=shot,family='magnetic')
        row = dict(shot=shot, hsds_shot_present=shot in remote, server_efit_directory=shot in local,
                   normal_shot_number=10000 <= shot < 100000, stage_status='unrecorded',
                   attempted_points=0, usable_recorded=0, inventory_error=None)
        if manifest_path.exists():
            try:
                manifest=json.loads(manifest_path.read_text())
                statuses=manifest.get('slice_statuses',[])
                row.update(stage_status=manifest.get('status','unrecorded'),
                           runtime_status=manifest.get('efit_status'), attempted_points=len(statuses),
                           usable_recorded=sum(s.get('overall_status')=='usable' for s in statuses))
                for status in statuses:
                    attempts.append(dict(shot=shot,time_s=status['time'],overall_status=status.get('overall_status'),
                                         failure_codes_json=json.dumps(status.get('failure_codes',[]))))
            except Exception as exc:
                row['inventory_error']=str(exc)
        rows.append(row)
    pd.DataFrame(rows).to_csv(output/'inventory_shots.csv',index=False)
    pd.DataFrame(attempts,columns=['shot','time_s','overall_status','failure_codes_json']).to_csv(output/'inventory_attempts.csv',index=False)
    write_json(output/'selection.json',dict(source=source,shots=shots,hsds_shots=len(remote),
        server_efit_shots=len(local), union_shots=len(shots), recorded_attempts=len(attempts),
        stage_counts=pd.Series([r['stage_status'] for r in rows]).value_counts().to_dict(),
        generated_at=datetime.now(timezone.utc).isoformat()))
    return shots


def collect_with_retries(shot, *, source, db, vacuum, progress, progress_path,
                         delays=(10, 30, 90)):
    """Retry transient transports without appending partially collected rows."""
    for attempt in range(len(delays) + 1):
        try:
            return equilibrium_summary([shot], mode='server', source=source,
                                       filedb=db, vacuum=vacuum)
        except ConnectionError as exc:
            exhausted = attempt == len(delays)
            write_json(progress_path, {**progress,
                'state': 'failed' if exhausted else 'retrying',
                'retry_count': attempt + 1, 'error': str(exc),
                'updated_at': datetime.now(timezone.utc).isoformat()})
            if exhausted:
                raise
            print('connection_retry', shot, attempt + 1, 'delay_s', delays[attempt],
                  str(exc), flush=True)
            time.sleep(delays[attempt])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--filedb',required=True)
    p.add_argument('--source',default='main')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=10)
    p.add_argument('--inventory-only',action='store_true')
    p.add_argument('--vacuum',action='store_true')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if args.batch_size <= 0: p.error('batch-size must be positive')
    args.output.mkdir(parents=True,exist_ok=True)
    db=FileDB(args.filedb)
    selection=args.output/'selection.json'
    if args.resume and selection.exists():
        shots=json.loads(selection.read_text())['shots']
    else:
        shots=inventory(args.source,db,args.output)
    print('population',len(shots),flush=True)
    if args.inventory_only: return
    # Inspect representative generations and known failures first, then scan all.
    first=[s for s in (39915,48922,48925,48926,48224,39014,38996) if s in shots]
    shots=first+[s for s in shots if s not in set(first)]
    previous_elapsed = 0.0
    progress_path = args.output/'progress.json'
    if args.resume and progress_path.exists():
        previous_elapsed = float(json.loads(progress_path.read_text()).get('elapsed_s', 0))
    started=time.monotonic()
    completed=0
    for offset in range(0,len(shots),args.batch_size):
        batch=shots[offset:offset+args.batch_size]
        destination=args.output/'batches'/f'batch_{offset//args.batch_size:05d}'
        marker=destination/'complete.json'
        if args.resume and marker.exists():
            completed+=len(batch)
            continue
        if destination.exists():
            stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
            destination.rename(destination.with_name(destination.name+'_incomplete_'+stamp))
        destination.mkdir(parents=True)
        populations=[]
        for shot in batch:
            progress = dict(state='running',completed_shots=completed,
                total_shots=len(shots),current_shot=shot,elapsed_s=previous_elapsed+time.monotonic()-started,
                batch=str(destination),updated_at=datetime.now(timezone.utc).isoformat())
            write_json(progress_path, progress)
            result=collect_with_retries(shot, source=args.source, db=db, vacuum=args.vacuum,
                                        progress=progress, progress_path=progress_path)
            for name in TABLE_COLUMNS:
                path=destination/(name+'.csv')
                getattr(result,name).to_csv(path,mode='a',header=not path.exists(),index=False)
            populations.append(result.population())
            completed+=1
            print('completed',completed,'/',len(shots),'shot',shot,'seconds',round(time.monotonic()-started,1),flush=True)
        write_json(destination/'manifest.json',dict(schema_version=1,shots=batch,mode='server',
            source=args.source,vacuum=args.vacuum,magnetics=True,vaft_version=result.metadata['vaft_version']))
        write_json(destination/'population.json',populations)
        write_json(marker,dict(shots=batch,completed_at=datetime.now(timezone.utc).isoformat()))
    write_json(args.output/'progress.json',dict(state='complete',completed_shots=completed,total_shots=len(shots),
        elapsed_s=previous_elapsed+time.monotonic()-started,updated_at=datetime.now(timezone.utc).isoformat()))
    print('complete',completed,flush=True)


if __name__=='__main__':
    main()
