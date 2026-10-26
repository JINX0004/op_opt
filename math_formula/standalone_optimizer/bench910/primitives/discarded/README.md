These raw pilot files are retained for provenance but **must not be used to
calibrate `SolveBase`**.

- `launch_read_pre_sync_raw.jsonl` predates explicit MTE2-to-scalar
  synchronization and the 32-byte separation of per-core validation-token
  stores. It includes flat-read comparisons only.
- `nd2nz_read_pre_stride_fix_raw.jsonl` and
  `repeat_selected_pre_stride_fix_raw.jsonl` were recorded while mode 4
  allocated 64 KiB of source data per AIC regardless of the requested
  transfer amount. Larger reads could access beyond the allocation.

The corresponding bounds-valid ND→NZ data is in
`../nd2nz_read_corrected_raw.jsonl` and
`../repeat_selected_corrected_raw.jsonl`. The analyzer never loads files
from this directory.
