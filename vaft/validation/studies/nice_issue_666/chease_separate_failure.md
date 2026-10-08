# Separate pre-existing CHEASE tolerance failure

Reproduced independently on 2026-09-08:

```sh
python -m pytest test/test_chease_adapter.py::test_run_chease_gfile_and_equivalent_ods_input_agree -q
```

Both CHEASE runs succeed. Their near-zero ZMAXIS values differ by
3.18049e-11 (4.598697e-7 versus 4.598379e-7). The existing comparison uses
rtol=1e-7 and atol=0, giving relative difference 6.91654557e-5 and a failed
assertion at test/test_chease_adapter.py:253. This is separate from NICE
issue #666. Neither CHEASE code nor its tolerance was changed.
