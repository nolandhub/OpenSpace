# asr_streaming - latency mỗi chunk theo CCU

| CCU | p50 ms | p90 ms | p95 ms | p99 ms | max ms | RTF p95 | first-chunk ms | RTF stream | queue ms | compute ms | batch avg | samples |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 18.85 | 23.02 | 23.68 | 26.32 | 43.6 | 0.118 | 2.62 | 0.067 | 0.05 | 13.32 | 1.0 | 299 |
| 2 | 20.68 | 43.45 | 45.28 | 50.03 | 99.79 | 0.226 | 4.79 | 0.068 | 6.91 | 13.65 | 1.01 | 598 |
| 3 | 21.92 | 65.89 | 68.38 | 73.49 | 111.01 | 0.342 | 7.87 | 0.068 | 9.12 | 22.68 | 1.51 | 897 |
| 4 | 23.97 | 89.37 | 92.77 | 106.42 | 164.55 | 0.464 | 9.51 | 0.074 | 11.47 | 32.73 | 1.87 | 1196 |

chunk = 200ms · RTF p95 = p95 ms / 200
RTF stream = thời gian server bận (batch_stats) / độ dài audio
compute ms = ghi công mỗi request (inference_stats), nên nhân lên theo batch avg
