| chunker | edit | chunks in file | incremental: embedded | incremental: median s | full: median s | speed-up |
|---|---|---:|---:|---:|---:|---:|
| character | none | 240 | 0 | 0.330 | 3.365 | 10.2x |
| character | insert | 239 | 119 (49.8%) | 2.033 | 3.439 | 1.7x |
| character | edit | 239 | 6 (2.5%) | 0.618 | 3.569 | 5.8x |
| character | delete | 241 | 172 (71.4%) | 2.691 | 3.520 | 1.3x |
| character | append | 241 | 1 (0.4%) | 0.393 | 3.550 | 9.0x |
| paragraph | none | 286 | 0 | 0.312 | 4.234 | 13.6x |
| paragraph | insert | 287 | 18 (6.3%) | 0.784 | 4.236 | 5.4x |
| paragraph | edit | 287 | 1 (0.3%) | 0.439 | 4.505 | 10.3x |
| paragraph | delete | 286 | 20 (7.0%) | 0.885 | 4.364 | 4.9x |
| paragraph | append | 287 | 1 (0.3%) | 0.387 | 4.358 | 11.3x |
| recursive | none | 281 | 0 | 0.296 | 3.789 | 12.8x |
| recursive | insert | 280 | 21 (7.5%) | 0.856 | 3.821 | 4.5x |
| recursive | edit | 280 | 1 (0.4%) | 0.390 | 3.868 | 9.9x |
| recursive | delete | 281 | 28 (10.0%) | 0.810 | 3.903 | 4.8x |
| recursive | append | 281 | 1 (0.4%) | 0.384 | 3.932 | 10.2x |
| semantic | none | 53 | 0 | 7.414 | 7.601 | 1.0x |
| semantic | insert | 54 | 24 (44.4%) | 7.407 | 7.482 | 1.0x |
| semantic | edit | 54 | 1 (1.9%) | 6.623 | 7.448 | 1.1x |
| semantic | delete | 53 | 34 (64.2%) | 7.240 | 7.487 | 1.0x |
| semantic | append | 53 | 1 (1.9%) | 7.166 | 7.858 | 1.1x |
