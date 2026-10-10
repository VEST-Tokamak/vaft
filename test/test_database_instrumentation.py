"""Opt-in HSDS I/O instrumentation (#1869): records inside record_io, changes nothing outside it."""

from __future__ import annotations

import requests
from requests.adapters import BaseAdapter

from vaft.database import instrumentation
from vaft.database.instrumentation import RUN_ID_HEADER, active_recorder, record_io, record_subprocess


class _CannedAdapter(BaseAdapter):
    """Answers every request locally: 200 with a body, or a 404; records what it was sent."""

    def __init__(self, body: bytes = b"0123456789", with_length: bool = True):
        super().__init__()
        self.body, self.with_length, self.seen = body, with_length, []

    def send(self, request, **kwargs):
        self.seen.append(request)
        response = requests.Response()
        response.status_code = 404 if request.url.endswith("/missing") else 200
        response._content = self.body
        if self.with_length:
            response.headers["Content-Length"] = str(len(self.body))
        response.url, response.request = request.url, request
        return response

    def close(self):
        pass


def _session(adapter):
    session = requests.Session()
    session.mount("http://", adapter)
    return session


def test_nothing_is_patched_or_recorded_outside_the_context():
    original = requests.Session.send
    adapter = _CannedAdapter()
    _session(adapter).get("http://hsds.test/domains")
    assert requests.Session.send is original and active_recorder() is None
    assert RUN_ID_HEADER not in adapter.seen[0].headers
    record_subprocess("hsget", "hdf5://main/1/", 1.0, 10)  # a no-op without a recorder


def test_requests_are_recorded_with_the_run_id_and_restored_after():
    original = requests.Session.send
    adapter = _CannedAdapter()
    session = _session(adapter)
    with record_io(run_id="bench-1") as recorder:
        assert requests.Session.send is not original
        session.get("http://hsds.test/datasets/d-1/value?select=[0:10]")
        session.put("http://hsds.test/datasets/d-1/value", data=b"abcd")
        session.get("http://hsds.test/missing")
        record_subprocess("hsget", "hdf5://main/39915/master.h5", 0.5, 2048)
    assert requests.Session.send is original
    assert all(r.headers[RUN_ID_HEADER] == "bench-1" for r in adapter.seen)
    summary = recorder.summary()
    assert summary["request_count"] == 3
    assert summary["requests_by_method"] == {"GET": 2, "PUT": 1}
    assert summary["error_count"] == 1
    assert summary["request_body_bytes"] == 4
    assert summary["response_bytes_observed"] == 30 and summary["responses_without_length"] == 0
    assert summary["header_latency_p50_s"] is not None and summary["header_latency_p95_s"] >= summary["header_latency_p50_s"]
    assert summary["subprocess_count"] == 1 and summary["subprocess_bytes"] == 2048
    assert recorder.requests[0]["path"] == "/datasets/d-1/value"
    assert summary["ended_utc"] is not None


def test_a_response_without_content_length_is_unknown_not_zero():
    with record_io() as recorder:
        _session(_CannedAdapter(with_length=False)).get("http://hsds.test/x")
    summary = recorder.summary()
    assert summary["response_bytes_observed"] == 0 and summary["responses_without_length"] == 1
    assert recorder.requests[0]["response_bytes_observed"] is None


def test_nested_recorders_attribute_requests_to_the_innermost_and_restore_once():
    original = requests.Session.send
    session = _session(_CannedAdapter())
    with record_io(run_id="outer") as outer:
        session.get("http://hsds.test/a")
        with record_io(run_id="inner") as inner:
            session.get("http://hsds.test/b")
        assert requests.Session.send is not original
        session.get("http://hsds.test/c")
    assert requests.Session.send is original and instrumentation._original_send is None
    assert [r["path"] for r in outer.requests] == ["/a", "/c"]
    assert [r["path"] for r in inner.requests] == ["/b"]


def test_an_empty_recorder_summarises_without_percentiles():
    with record_io() as recorder:
        pass
    summary = recorder.summary()
    assert summary["request_count"] == 0 and summary["header_latency_p50_s"] is None
