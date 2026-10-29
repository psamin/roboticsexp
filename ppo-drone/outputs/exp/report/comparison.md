Held-out evaluation, 1,024 episodes per suite (test seed 2000). Stability columns average the last 2 s of episodes that did not fail.

### hover

| policy | success | crash | final dist | time to hover | body rate, last 2 s | yaw rate, last 2 s | wobble, last 2 s | action jitter, last 2 s | max tilt |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 100.0% | 0.0% | 1.0 cm | 0.09 s | 2.03 rad/s | 2.03 rad/s | 0.0 rad/s² | 0.0000 | 8.0° |
| stability | 100.0% | 0.0% | 0.5 cm | 0.08 s | 2.98 rad/s | 2.98 rad/s | 0.0 rad/s² | 0.0000 | 8.8° |
| shaped_nospin | 100.0% | 0.0% | 0.2 cm | 0.12 s | 0.50 rad/s | 0.50 rad/s | 0.0 rad/s² | 0.0000 | 5.0° |
| l2f | 100.0% | 0.0% | 1.9 cm | 0.10 s | 13.08 rad/s | 13.06 rad/s | 1.1 rad/s² | 0.0021 | 6.6° |
| l2f_full | 100.0% | 0.0% | 0.2 cm | 0.13 s | 100.98 rad/s | 100.97 rad/s | 0.4 rad/s² | 0.0008 | 9.5° |
| l2f_full_nospin | 100.0% | 0.0% | 0.1 cm | 0.09 s | 1.19 rad/s | 1.19 rad/s | 0.0 rad/s² | 0.0000 | 9.6° |
| PD controller | 100.0% | 0.0% | 0.0 cm | 0.15 s | 0.00 rad/s | 0.00 rad/s | 0.0 rad/s² | 0.0000 | 4.7° |
| random actions | 0.0% | 100.0% | 20.0 cm | n/a | n/a | n/a | n/a | n/a | 99.8° |

### reach

| policy | success | crash | final dist | time to hover | body rate, last 2 s | yaw rate, last 2 s | wobble, last 2 s | action jitter, last 2 s | max tilt |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 100.0% | 0.0% | 1.0 cm | 0.50 s | 2.03 rad/s | 2.03 rad/s | 0.0 rad/s² | 0.0000 | 32.3° |
| stability | 99.8% | 0.2% | 0.6 cm | 0.52 s | 2.98 rad/s | 2.98 rad/s | 0.0 rad/s² | 0.0000 | 38.1° |
| shaped_nospin | 100.0% | 0.0% | 0.2 cm | 0.74 s | 0.50 rad/s | 0.50 rad/s | 0.0 rad/s² | 0.0000 | 16.1° |
| l2f | 100.0% | 0.0% | 1.9 cm | 0.50 s | 13.08 rad/s | 13.06 rad/s | 1.1 rad/s² | 0.0021 | 26.8° |
| l2f_full | 100.0% | 0.0% | 0.2 cm | 0.62 s | 100.98 rad/s | 100.97 rad/s | 0.6 rad/s² | 0.0021 | 20.3° |
| l2f_full_nospin | 100.0% | 0.0% | 0.1 cm | 0.41 s | 1.19 rad/s | 1.19 rad/s | 0.0 rad/s² | 0.0000 | 37.0° |
| PD controller | 100.0% | 0.0% | 0.0 cm | 0.78 s | 0.00 rad/s | 0.00 rad/s | 0.0 rad/s² | 0.0000 | 15.4° |
| random actions | 0.0% | 100.0% | 49.1 cm | n/a | n/a | n/a | n/a | n/a | 99.6° |

### robust

| policy | success | crash | final dist | time to hover | body rate, last 2 s | yaw rate, last 2 s | wobble, last 2 s | action jitter, last 2 s | max tilt |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 99.0% | 1.0% | 2.7 cm | 0.73 s | 2.02 rad/s | 2.02 rad/s | 0.2 rad/s² | 0.0001 | 52.8° |
| stability | 84.3% | 15.7% | 18.0 cm | 0.75 s | 2.98 rad/s | 2.98 rad/s | 0.3 rad/s² | 0.0001 | 64.4° |
| shaped_nospin | 100.0% | 0.0% | 1.2 cm | 1.05 s | 0.44 rad/s | 0.44 rad/s | 0.0 rad/s² | 0.0000 | 29.0° |
| l2f | 100.0% | 0.0% | 2.5 cm | 0.65 s | 13.07 rad/s | 13.05 rad/s | 4.3 rad/s² | 0.0051 | 45.3° |
| l2f_full | 100.0% | 0.0% | 0.3 cm | 1.25 s | 100.98 rad/s | 100.96 rad/s | 20.2 rad/s² | 0.0942 | 36.8° |
| l2f_full_nospin | 96.4% | 3.6% | 4.5 cm | 0.64 s | 1.19 rad/s | 1.19 rad/s | 0.0 rad/s² | 0.0000 | 58.2° |
| PD controller | 100.0% | 0.0% | 2.6 cm | 1.09 s | 0.00 rad/s | 0.00 rad/s | 0.0 rad/s² | 0.0000 | 25.9° |
| random actions | 0.0% | 100.0% | 90.6 cm | n/a | n/a | n/a | n/a | n/a | 99.6° |

### recovery

| policy | success | crash | final dist | time to hover | body rate, last 2 s | yaw rate, last 2 s | wobble, last 2 s | action jitter, last 2 s | max tilt |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 100.0% | 0.0% | 1.0 cm | 0.37 s | 2.03 rad/s | 2.03 rad/s | 0.0 rad/s² | 0.0000 | 38.9° |
| stability | 99.7% | 0.3% | 0.6 cm | 0.36 s | 2.98 rad/s | 2.98 rad/s | 0.0 rad/s² | 0.0000 | 43.9° |
| shaped_nospin | 100.0% | 0.0% | 0.2 cm | 0.65 s | 0.50 rad/s | 0.50 rad/s | 0.0 rad/s² | 0.0000 | 35.1° |
| l2f | 100.0% | 0.0% | 1.9 cm | 0.45 s | 13.08 rad/s | 13.06 rad/s | 1.1 rad/s² | 0.0021 | 34.1° |
| l2f_full | 100.0% | 0.0% | 0.2 cm | 0.56 s | 100.98 rad/s | 100.97 rad/s | 0.6 rad/s² | 0.0020 | 34.5° |
| l2f_full_nospin | 99.7% | 0.3% | 0.2 cm | 0.48 s | 1.19 rad/s | 1.19 rad/s | 0.0 rad/s² | 0.0000 | 44.4° |
| PD controller | 100.0% | 0.0% | 0.0 cm | 0.80 s | 0.00 rad/s | 0.00 rad/s | 0.0 rad/s² | 0.0000 | 34.1° |
| random actions | 0.0% | 99.9% | 35.3 cm | n/a | n/a | n/a | n/a | n/a | 99.4° |
