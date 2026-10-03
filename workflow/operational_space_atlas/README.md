# Operational-space atlas: Lane V base table (#1456)

`build_efit_base.py` writes the Lane V layer of the conference atlas. It produces one row per state
key of Lane K's State key contract v1 (#1454), containing the operational-space coordinates
computed by VAFT from that row's own EFIT product. The columns are named by the quantity identity
of `vaft.formula.boundaries`.

```bash
python workflow/operational_space_atlas/build_efit_base.py \
    --state ~/runs/campaign/atlas/v1/state.csv \
    --filedb ~/runs/campaign/filedb \
    --out ~/runs/campaign/atlas/lane_v
```

- Each slice is matched by time (1 us; the state times come from the same product), never by
  index. Unmatched rows are kept with `base_status = time_unmatched`. A product whose sha256 no
  longer matches the state table gives `base_status = product_changed`.
- $l_i(3)$ and $\beta_N$ are the DD values of `update_equilibrium_global_quantities_beta_li`. This
  needs #1477, the fix of #1462. Each row is cross-checked against a grid integral of $B_p^2$
  (`li3_grid_integral`) and `derive_global_descriptors` (`beta_normal_descriptors`). The result is
  in `li_beta_crosscheck`, and the MANIFEST counts the rows that disagree.
- The script reads only the campaign FileDB and Lane K's table. It writes only `--out`
  (`efit_base.csv` and `MANIFEST.json`).

The figures are in `notebooks/conference_operational_space_atlas.ipynb`.
