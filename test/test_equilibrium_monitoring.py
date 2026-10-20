"""Storage coverage, time joins and source provenance of equilibrium monitoring."""
from contextlib import nullcontext
import json

import numpy as np
from omas import ODS
import pandas as pd
import pytest

from vaft.database._equilibrium_monitoring import (
    FileDBEvidenceBackend, HSDSEvidenceBackend, ShotEvidence,
    TABLE_COLUMNS, collect_evidence, equilibrium_summary,
)


def ods_with_times(times):
    ods = ODS(consistency_check=False)
    ods['equilibrium.time'] = np.array(times)
    for i, time in enumerate(times):
        ods[f'equilibrium.time_slice.{i}.time'] = time
    return ods


def status(time, outcome='output_failed'):
    return dict(time=time, overall_status=outcome, failure_codes=['missing_gfile'],
                metrics={'ip': 10.0}, provenance={'kfile_sha256': 'abc'})


class Backend:
    def __init__(self, name, observations):
        self.name, self.observations = name, observations
    def read(self, shot):
        return self.observations.get(shot, ShotEvidence(shot, self.name, 'main',
            ods=ODS(consistency_check=False), availability='no_equilibrium'))


def test_no_output_attempts_and_missing_shots_survive():
    failure = ShotEvidence(1, 'filedb', 'main', ods=ODS(consistency_check=False),
        collection={'slice_statuses': [status(.31), status(.32)]},
        manifest={'status': 'no_output'}, availability='no_output')
    result = equilibrium_summary([1, 2], backends=[Backend('filedb', {1: failure})], magnetics=False)
    assert result.shots.shot.tolist() == [1, 2]
    assert result.attempts.time_s.tolist() == [.31, .32]
    assert not result.attempts.has_equilibrium.any()
    assert result.shots.attempt_coverage.tolist() == ['recorded_attempts', 'unknown']
    assert result.metrics.value.tolist() == [10., 10.]


def test_join_is_by_time_and_keeps_status_without_output(monkeypatch):
    import vaft.validation.equilibrium as validation
    monkeypatch.setattr(validation, 'validate_equilibrium', lambda *a, **k: {})
    ods = ods_with_times([.32, .31])
    e = ShotEvidence(1, 'hsds', 'main', ods=ods,
        collection={'slice_statuses': [status(.31), status(.30), status(.32, 'usable')]})
    rows = collect_evidence(e, magnetics=False)['attempts']
    assert [r['time_s'] for r in rows] == [.30, .31, .32]
    assert [r['eq_index'] for r in rows] == [None, 1, 0]
    assert [r['overall_status'] for r in rows] == ['output_failed', 'output_failed', 'usable']


def test_colliding_millisecond_times_are_not_silently_joined():
    e = ShotEvidence(1, 'hsds', 'main', ods=ods_with_times([.3101, .3102]))
    result = equilibrium_summary([1], backends=[Backend('hsds', {1: e})], magnetics=False)
    assert result.shots.availability.iloc[0] == 'analysis_error'
    assert 'ambiguous time' in result.shots.errors_json.iloc[0]
    assert result.attempts.empty


def test_server_and_hsds_observations_remain_separate():
    a = ShotEvidence(1, 'hsds', 'main', ods=ODS(consistency_check=False),
                     collection={'slice_statuses': [status(.31, 'usable')]})
    b = ShotEvidence(1, 'filedb', 'main', ods=ODS(consistency_check=False),
                     collection={'slice_statuses': [status(.31, 'physical_failed')]})
    result = equilibrium_summary([1], mode='server', magnetics=False,
        backends=[Backend('hsds', {1:a}), Backend('filedb', {1:b})])
    assert set(result.attempts.backend) == {'hsds', 'filedb'}
    assert result.attempts.product_id.nunique() == 2
    assert set(result.attempts.overall_status) == {'usable', 'physical_failed'}


def test_constraints_and_kfile_only_times_are_unrecorded():
    constraints = ods_with_times([.31])
    constraints['equilibrium.time_slice.0.constraints.ip.measured'] = 15000.
    e = ShotEvidence(1, 'filedb', 'main', ods=ODS(consistency_check=False), constraints=constraints,
        kfiles=[dict(time=.32, sha256='filehash', namelists={'in1': {'plasma': 18000.}})])
    rows = collect_evidence(e, magnetics=False)
    assert [r['overall_status'] for r in rows['attempts']] == ['unrecorded', 'unrecorded']
    assert [r['kind'] for r in rows['inputs']] == ['constraint_stage', 'kfile_namelists']
    assert json.loads(rows['inputs'][0]['snapshot_json'])['ip']['measured'] == 15000.


def test_whole_product_continuity_not_repeated_as_slice_check(monkeypatch):
    import vaft.validation.equilibrium as validation
    monkeypatch.setattr(validation, 'validate_equilibrium', lambda *a, **k: {
        'time_slices':[0,1], 'verification': {
            'continuity':{'status':'warn','ip_max_relative_step':.9},
            'structure':{'status':'fail','slices':[
                {'time_slice':0,'status':'pass','volume':1},
                {'time_slice':1,'status':'fail','volume':-1}]}}})
    e = ShotEvidence(1, 'hsds', 'main', ods=ods_with_times([.31,.32]))
    rows = collect_evidence(e, magnetics=False)
    whole = [r for r in rows['checks'] if r['check'] == 'verification.continuity']
    assert len(whole) == 1 and whole[0]['attempt_id'] is None
    sliced = [r for r in rows['checks'] if r['check'] == 'verification.structure']
    assert [r['status'] for r in sliced] == ['pass','fail']
    assert any(r['metric']=='ip_max_relative_step' and r['value']==.9 for r in rows['metrics'])


def test_hsds_optional_absence_does_not_erase_equilibrium(monkeypatch):
    from vaft import database
    opened = []
    def open_shot(shot, source, paths):
        opened.append(paths[0])
        if paths[0] == 'equilibrium':
            return nullcontext(ods_with_times([.31]))
        raise FileNotFoundError(paths[0])
    monkeypatch.setattr(database, 'open', open_shot)
    e = HSDSEvidenceBackend(magnetics=False).read(1)
    assert e.availability == 'present'
    assert e.ods['equilibrium.time'][0] == .31
    assert 'camera_visible' not in opened
    assert e.errors


def test_hsds_transport_failure_stops_immediately(monkeypatch):
    from vaft import database
    calls = []
    def fail(shot, **kw):
        calls.append(shot)
        raise TimeoutError('down')
    monkeypatch.setattr(database, 'open', fail)
    with pytest.raises(ConnectionError, match='unreachable'):
        equilibrium_summary([1,2], magnetics=False)
    assert calls == [1]


def test_filedb_no_output_manifest_and_listed_kfile_are_read(tmp_path):
    from vaft.database.filedb import FileDB
    db = FileDB(tmp_path)
    manifest = db.omas_manifest('efit',shot=1,family='magnetic')
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({'status':'no_output','slice_statuses':[status(.31)]}))
    kdir = db.efit(1,family='magnetic',artifact='input')
    kdir.mkdir(parents=True)
    kfile = kdir / 'k000001.00310'
    kfile.write_text('&in1\n plasma=15000, kppcur=1\n/\n')
    (kdir/'kfiles_generated.txt').write_text(str(kfile)+'\n')
    import hashlib
    recorded = status(.31)
    recorded['provenance']['kfile_sha256'] = hashlib.sha256(kfile.read_bytes()).hexdigest()
    manifest.write_text(json.dumps({'status':'no_output','slice_statuses':[recorded]}))
    # This stale unlisted input must not enter the attempted-time denominator.
    (kdir/'k000001.00320').write_text('&in1\n plasma=99\n/\n')
    e = FileDBEvidenceBackend(db,magnetics=False).read(1)
    assert e.availability == 'no_output'
    assert len(e.kfiles) == 1
    rows = collect_evidence(e,magnetics=False)
    assert len(rows['attempts']) == 1
    assert json.loads(rows['inputs'][0]['snapshot_json'])['in1']['plasma'] == 15000


def test_export_and_plot_missing_observations(tmp_path):
    result = equilibrium_summary([1,2], backends=[Backend('filedb',{})], magnetics=False)
    result.export(tmp_path)
    assert set(TABLE_COLUMNS).issubset({p.stem for p in tmp_path.glob('*.csv')})
    assert json.loads((tmp_path/'manifest.json').read_text())['schema_version'] == 1
    import matplotlib
    matplotlib.use('Agg')
    from vaft.plot.equilibrium_monitoring import write_monitoring_figures
    paths = write_monitoring_figures(result,tmp_path)
    assert paths and all(p.stat().st_size > 1000 for p in paths)


def test_public_api_and_cli_dispatch(monkeypatch,tmp_path):
    from vaft import database
    import vaft.database._equilibrium_monitoring as module
    import vaft.cli.summary as cli
    called = []
    def fake(shots, **kw):
        called.append((list(shots),kw))
        return equilibrium_summary([1],backends=[Backend('test',{})],magnetics=False)
    monkeypatch.setattr(module,'equilibrium_summary',fake)
    database.equilibrium_summary([1])
    assert callable(database.equilibrium_summary)
    assert cli.main(['monitor','--shot-range','1:2','--mode','server','--filedb',str(tmp_path),
                     '--output',str(tmp_path/'out'),'--no-magnetics']) == 0
    assert called[-1][0] == [1,2]
    assert called[-1][1]['mode'] == 'server'


@pytest.mark.parametrize('kwargs', [{'mode':'other'}, {'mode':'server'}, {'family':'invalid'}])
def test_invalid_storage_selection_is_rejected(kwargs):
    with pytest.raises(ValueError):
        equilibrium_summary([1],**kwargs)


def test_mismatched_input_generation_is_refused():
    e = ShotEvidence(1,'filedb','main',ods=ODS(consistency_check=False),
        collection={'slice_statuses':[status(.31)]},
        kfiles=[dict(time=.31,sha256='new-generation',namelists={})])
    result = equilibrium_summary([1],backends=[Backend('filedb',{1:e})],magnetics=False)
    assert result.shots.availability.iloc[0] == 'analysis_error'
    assert 'generation mismatch' in result.shots.errors_json.iloc[0]
    assert result.inputs.empty


def test_population_does_not_treat_absence_as_failure():
    e = ShotEvidence(1,'filedb','main',ods=ODS(consistency_check=False),
        collection={'slice_statuses':[status(.31,'usable'),status(.32)]})
    result = equilibrium_summary([1,2],backends=[Backend('filedb',{1:e})],magnetics=False)
    group = result.population()['groups'][0]
    assert group['requested_shots'] == 2
    assert group['recorded_attempts'] == 2
    assert group['usable_fraction_recorded'] == .5


def test_vacuum_is_window_scoped_and_opt_in(monkeypatch):
    import vaft.validation.vacuum_benchmark as vacuum
    calls=[]
    def benchmark(*a,**kw):
        calls.append(kw)
        return {'windows':{'validation':[.1,.2]},'metrics':{'channels':[
            {'name':'C1','family':'probe','status':'evaluated','residual':.2}]}}
    monkeypatch.setattr(vacuum,'run_benchmark_case',benchmark)
    e=ShotEvidence(1,'test','main',ods=ODS(consistency_check=False))
    assert not collect_evidence(e,magnetics=False)['checks']
    rows=collect_evidence(e,magnetics=False,vacuum=True)
    assert len(calls)==1
    assert rows['checks'][0]['scope']=='window'
    assert rows['checks'][0]['attempt_id'] is None
    assert rows['channels'][0]['scope']=='vacuum_window'


def test_unavailable_vacuum_carries_reason(monkeypatch):
    import vaft.validation.vacuum_benchmark as vacuum
    def fail(*a,**kw):
        raise vacuum.BenchmarkError('insufficient wall history')
    monkeypatch.setattr(vacuum,'run_benchmark_case',fail)
    rows=collect_evidence(ShotEvidence(1,'test','main',ods=ODS(consistency_check=False)),
                          magnetics=False,vacuum=True)
    assert rows['checks'][0]['status']=='not_available'
    assert 'wall history' in rows['checks'][0]['result_json']


def test_cli_can_partition_a_large_query(monkeypatch,tmp_path):
    import vaft.database._equilibrium_monitoring as module
    import vaft.cli.summary as cli
    calls=[]
    def fake(shots,**kw):
        calls.append(list(shots))
        return equilibrium_summary(shots,backends=[Backend('test',{})],magnetics=False)
    monkeypatch.setattr(module,'equilibrium_summary',fake)
    cli.main(['monitor','--shot-range','1:5','--batch-size','2','--output',str(tmp_path)])
    assert calls == [[1,2],[3,4],[5]]
    assert (tmp_path/'shots_5_5'/'shots.csv').exists()


@pytest.mark.parametrize('family,stage,source',[('electron','electron_efit','electron-efit'),
                                             ('kinetic','kinetic_efit','kinetic-efit')])
def test_corrective_family_manifest_is_read_without_inventing_usable(tmp_path,family,stage,source):
    from vaft.database.filedb import FileDB
    db=FileDB(tmp_path)
    manifest=db.omas_manifest(stage,shot=1)
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({'status':'no_output','attempts':[
        {'time_ms':310,'status':'no_output','converged':False,'reason':'negative pressure'}]}))
    e=FileDBEvidenceBackend(db,source=source,family=family,magnetics=False).read(1)
    result=equilibrium_summary([1],family=family,backends=[Backend('filedb',{1:e})],magnetics=False)
    assert result.attempts.time_s.tolist()==[.31]
    assert result.attempts.overall_status.tolist()==['stage_no_output']
    assert result.population()['groups'][0]['usable_fraction_recorded'] is None
    assert result.metadata['source']==source


def test_population_runner_retries_transport_and_records_exhaustion(monkeypatch, tmp_path):
    from workflow.equilibrium_monitoring import run_population as runner
    calls = []
    sleeps = []
    def transient(*args, **kwargs):
        calls.append(args)
        if len(calls) < 3:
            raise ConnectionError('interrupted binary response')
        return 'collected'
    monkeypatch.setattr(runner, 'equilibrium_summary', transient)
    monkeypatch.setattr(runner.time, 'sleep', sleeps.append)
    path = tmp_path / 'progress.json'
    kwargs = dict(source='main', db=object(), vacuum=True,
                  progress={'completed_shots': 10}, progress_path=path, delays=(1, 2))
    assert runner.collect_with_retries(42738, **kwargs) == 'collected'
    assert sleeps == [1, 2]
    assert len(calls) == 3
    assert json.loads(path.read_text())['state'] == 'retrying'
    def unavailable(*args, **kwargs):
        raise ConnectionError('still unavailable')
    monkeypatch.setattr(runner, 'equilibrium_summary', unavailable)
    with pytest.raises(ConnectionError):
        runner.collect_with_retries(42738, **kwargs)
    assert json.loads(path.read_text())['state'] == 'failed'
