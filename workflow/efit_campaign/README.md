# EFIT campaign shot selection (#1331)

The #891 working setting (`statistical_891`, the EFIT default since 2026-10-01) is run over shots chosen for three reasons:
- **importance**: record-class Ip or a long pulse;
- **magnetics quality**: like 39915's;
- **Thomson coverage**: needed for the kinetic lineage.

`select_shots.py` joins the sources that answer those questions and labels each shot with a tier.

| tier | what it is |
| --- | --- |
| A | Thomson, and magnetics like 39915's: ≤ 4 condemned probes and ≥ 19 usable outboard probes. The σ-recalibration and kinetic set. |
| B | Thomson and an important discharge (Ip ≥ 250 kA or pulse ≥ 30 ms). Its magnetics fall short of A but are not unfit. |
| C | Thomson, but no magnetics in the database. Needs ingest first. |
| D | No Thomson, but important with A-quality magnetics. Magnetic-only. |

A shot is in the first tier that applies. The thresholds are command-line arguments and are recorded in the output.

## Inputs, in the order they are made

1. **The operational overview, from the database.** It gives peak Ip, pulse duration and shot class. Thomson presence is not a column here, because file presence is not database metadata.

   ```bash
   python workflow/automatic_pipeline_3_data_summary/gen_omas_history.py \
       --shot-range 30000:49999 --source main --output shot_overview.xlsx --rebuild
   ```

2. **Magnetics quality.** Scan every shot that has Thomson and every important shot. An absent shot is recorded as an `absent` row rather than aborting the batch.

   ```bash
   python workflow/magnetics_quality/scan_magnetics_quality.py --shots 44780,41717,... --table mq/important.json
   ```

3. **The raw Thomson and ion uploads.** `--data-root` points at the external diagnostic root, `/srv/vest.diagnostic` on vestserver. Thomson files are recognised by the resolver's own layouts (`thomson_file_shot`), and ion files by `CES_{shot}.mat` / `IDS_{shot}.mat`.

```bash
python workflow/efit_campaign/select_shots.py --overview shot_overview.xlsx \
    --quality mq/ --data-root /srv/vest.diagnostic \
    --output selection.json --markdown selection.md
```

A shot with no quality row is `not_scanned` and never counts as good magnetics.
