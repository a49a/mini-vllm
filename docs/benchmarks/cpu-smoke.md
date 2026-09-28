# KV comparison / KV 对照实验

Platform: `macOS-14.8.9-arm64-arm-64bit`. Device: cpu; dtype: f32.

Warmup excluded from timing/allocations. RSS includes model loading and warmup; it is not VRAM. Allocation counts include persistent KV tensors only. Small samples are smoke measurements, not capacity claims.

| Mode | Trial | Success | tok/s | TTFT median ms | Peak RSS MiB | KV allocations | Computed prompt tokens | Prefix hits |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| contiguous | 0 | 2/2 | 12.61 | 341.97 | 2518.89 | 96 | 46 | 0 |
| paged | 0 | 2/2 | 12.99 | 350.49 | 2800.45 | 384 | 46 | 0 |
| prefix | 0 | 2/2 | 18.74 | 158.11 | 2819.06 | 192 | 14 | 32 |
| paged | 1 | 2/2 | 13.01 | 351.22 | 2818.59 | 384 | 46 | 0 |
| prefix | 1 | 2/2 | 19.89 | 144.98 | 2814.12 | 192 | 14 | 32 |
| contiguous | 1 | 2/2 | 10.03 | 445.73 | 2438.66 | 96 | 46 | 0 |
| prefix | 2 | 2/2 | 15.61 | 187.24 | 2811.88 | 192 | 14 | 32 |
| contiguous | 2 | 2/2 | 9.18 | 423.12 | 2836.55 | 96 | 46 | 0 |
| paged | 2 | 2/2 | 12.36 | 378.29 | 2806.09 | 384 | 46 | 0 |
