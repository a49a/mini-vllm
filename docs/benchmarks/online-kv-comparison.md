# KV comparison / KV 对照实验

Platform: `macOS-14.8.9-arm64-arm-64bit`. Device: cpu; dtype: f32.

Warmup excluded from timing/allocations. RSS includes model loading and warmup; it is not VRAM. Allocation counts include persistent KV tensors only. Small samples are smoke measurements, not capacity claims.

| Mode | Trial | Success | tok/s | TTFT median ms | Peak RSS MiB | KV allocations | Computed prompt tokens | Prefix hits |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| contiguous | 0 | 2/2 | 13.62 | 329.27 | 1980.86 | 96 | 46 | 0 |
| paged | 0 | 2/2 | 13.40 | 336.61 | 2039.14 | 384 | 46 | 0 |
| prefix | 0 | 2/2 | 15.46 | 180.57 | 1989.92 | 192 | 14 | 32 |
| paged | 1 | 2/2 | 13.78 | 333.34 | 2112.84 | 384 | 46 | 0 |
| prefix | 1 | 2/2 | 20.73 | 139.16 | 1992.67 | 192 | 14 | 32 |
| contiguous | 1 | 2/2 | 13.04 | 322.05 | 1995.34 | 96 | 46 | 0 |
| prefix | 2 | 2/2 | 14.10 | 188.85 | 2153.38 | 192 | 14 | 32 |
| contiguous | 2 | 2/2 | 14.01 | 326.94 | 1995.92 | 96 | 46 | 0 |
| paged | 2 | 2/2 | 13.85 | 329.95 | 1984.78 | 384 | 46 | 0 |
