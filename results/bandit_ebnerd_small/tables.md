### Tuned hyper-parameters (selected on the tuning period; frozen)

| policy (update mode) | frozen hyper-parameters | tuning-period CTR |
|---|---|---:|
| popularity_greedy (per-impression) | `{"window_h": 24, "m": 5}` | 0.2408 |
| popularity_greedy (hourly) | `{"window_h": 6, "m": 20}` | 0.1860 |
| eps_greedy_lgbm | `{"epsilon": 0.003}` | 0.3283 |
| linucb (per-impression) | `{"alpha": 0.0, "lam": 100}` | 0.3004 |
| linucb (hourly) | `{"alpha": 0.03, "lam": 10}` | 0.2872 |
| lints (per-impression) | `{"v": 0.003, "lam": 10}` | 0.3001 |
| lints (hourly) | `{"v": 0.03, "lam": 10}` | 0.2877 |
| hybrid_linucb (per-impression) | `{"alpha": 0.003, "lam": 100, "w": 0.0}` | 0.3478 |
| hybrid_linucb (hourly) | `{"alpha": 0.0, "lam": 100, "w": 1.0}` | 0.3420 |
| hybrid_lints (per-impression) | `{"v": 0.1, "lam": 100, "w": 0.0}` | 0.3472 |
| hybrid_lints (hourly) | `{"v": 0.1, "lam": 100, "w": 1.0}` | 0.3421 |

References on the tuning period: random 0.1194, LightGBM-greedy 0.3289.

### Held-out primary results (days 3-6; 147492 impressions, 13261 users; user-bootstrap 95% CI, n=2000)

| policy | top-1 CTR | mean regret / imp | cumulative regret | final-window (last 20%) CTR | paired Δ CTR vs LightGBM-greedy | verdict |
|---|---|---|---:|---|---|---|
| random | 0.1157 [0.1148, 0.1166] | 0.8843 | 130,426 | 0.1173 [0.1156, 0.1191] | -0.1932 [-0.1964, -0.1899] | LOSS |
| popularity_greedy (per-impression) | 0.2303 [0.2283, 0.2324] | 0.7697 | 113,521 | 0.2316 [0.2275, 0.2358] | -0.0785 [-0.0814, -0.0756] | LOSS |
| popularity_greedy (hourly) | 0.1808 [0.1791, 0.1825] | 0.8192 | 120,818 | 0.1943 [0.1906, 0.1979] | -0.1280 [-0.1309, -0.1250] | LOSS |
| lgbm_greedy | 0.3089 [0.3058, 0.3120] | 0.6911 | 101,936 | 0.3169 [0.3106, 0.3232] | - | reference |
| eps_greedy_lgbm | 0.3084 [0.3053, 0.3115] | 0.6916 | 102,009 | 0.3163 [0.3100, 0.3226] | -0.0005 [-0.0006, -0.0004] | LOSS |
| linucb (per-impression) | 0.2813 [0.2784, 0.2843] | 0.7187 | 106,002 | 0.2895 [0.2833, 0.2959] | -0.0276 [-0.0301, -0.0249] | LOSS |
| linucb (hourly) | 0.2662 [0.2635, 0.2692] | 0.7338 | 108,223 | 0.2829 [0.2768, 0.2892] | -0.0426 [-0.0453, -0.0400] | LOSS |
| lints (per-impression) | 0.2810 [0.2781, 0.2839] | 0.7190 | 106,053 | 0.2894 [0.2833, 0.2959] | -0.0279 [-0.0304, -0.0254] | LOSS |
| lints (hourly) | 0.2683 [0.2655, 0.2712] | 0.7317 | 107,920 | 0.2836 [0.2776, 0.2898] | -0.0406 [-0.0430, -0.0380] | LOSS |
| hybrid_linucb (per-impression) | 0.3195 [0.3164, 0.3227] | 0.6805 | 100,370 | 0.3193 [0.3131, 0.3258] | +0.0106 [+0.0090, +0.0123] | win |
| hybrid_linucb (hourly) | 0.3143 [0.3110, 0.3175] | 0.6857 | 101,139 | 0.3171 [0.3109, 0.3235] | +0.0054 [+0.0038, +0.0071] | win |
| hybrid_lints (per-impression) | 0.3195 [0.3164, 0.3227] | 0.6805 | 100,372 | 0.3191 [0.3129, 0.3256] | +0.0106 [+0.0091, +0.0123] | win |
| hybrid_lints (hourly) | 0.3146 [0.3115, 0.3178] | 0.6854 | 101,085 | 0.3176 [0.3114, 0.3239] | +0.0058 [+0.0043, +0.0074] | win |

Seed-to-seed std of CTR (5 seeds, stochastic policies): random 0.0009, popularity_greedy (per-impression) 0.0017, popularity_greedy (hourly) 0.0038, eps_greedy_lgbm 0.0000, lints (per-impression) 0.0004, lints (hourly) 0.0008, hybrid_lints (per-impression) 0.0003, hybrid_lints (hourly) 0.0001.

### Learning check: CTR by held-out day, and paired Δ vs LightGBM-greedy in the last 20%

| policy | day 3 | day 4 | day 5 | day 6 | Δ vs LightGBM, final window |
|---|---:|---:|---:|---:|---|
| random | 0.1134 | 0.1146 | 0.1165 | 0.1181 | -0.1996 [-0.2062, -0.1930] |
| popularity_greedy (per-impression) | 0.2225 | 0.2302 | 0.2386 | 0.2293 | -0.0853 [-0.0918, -0.0791] |
| popularity_greedy (hourly) | 0.1691 | 0.1770 | 0.1870 | 0.1894 | -0.1226 [-0.1290, -0.1162] |
| lgbm_greedy | 0.3028 | 0.3029 | 0.3140 | 0.3152 | - |
| eps_greedy_lgbm | 0.3023 | 0.3024 | 0.3135 | 0.3147 | -0.0006 [-0.0008, -0.0004] |
| linucb (per-impression) | 0.2710 | 0.2777 | 0.2879 | 0.2878 | -0.0274 [-0.0334, -0.0215] |
| linucb (hourly) | 0.2391 | 0.2584 | 0.2817 | 0.2837 | -0.0340 [-0.0399, -0.0280] |
| lints (per-impression) | 0.2716 | 0.2756 | 0.2883 | 0.2876 | -0.0275 [-0.0335, -0.0216] |
| lints (hourly) | 0.2452 | 0.2622 | 0.2813 | 0.2828 | -0.0333 [-0.0392, -0.0275] |
| hybrid_linucb (per-impression) | 0.3184 | 0.3198 | 0.3203 | 0.3193 | +0.0024 [-0.0004, +0.0056] |
| hybrid_linucb (hourly) | 0.3048 | 0.3149 | 0.3199 | 0.3167 | +0.0002 [-0.0027, +0.0034] |
| hybrid_lints (per-impression) | 0.3183 | 0.3199 | 0.3205 | 0.3190 | +0.0022 [-0.0007, +0.0053] |
| hybrid_lints (hourly) | 0.3056 | 0.3153 | 0.3195 | 0.3175 | +0.0007 [-0.0020, +0.0037] |

### Slices (CTR [CI]; Δ vs LightGBM-greedy [CI]) for the policies that matter

| slice (n impressions) | lgbm_greedy | linucb (per-impression) | hybrid_linucb (per-impression) | hybrid_linucb (hourly) | hybrid_lints (hourly) |
|---|---|---|---|---|---|
| cold_article (clicked article <100 prior exposures) (n=3,892) | 0.2109 [0.1975, 0.2242] | 0.1973 [0.1841, 0.2101]<br>Δ -0.0136 [-0.0274, -0.0008] LOSS | 0.2258 [0.2118, 0.2397]<br>Δ +0.0149 [+0.0064, +0.0240] win | 0.2194 [0.2056, 0.2333]<br>Δ +0.0085 [+0.0008, +0.0167] win | 0.2190 [0.2052, 0.2323]<br>Δ +0.0080 [+0.0006, +0.0160] win |
| warm_article (>= 100) (n=143,600) | 0.3115 [0.3084, 0.3145] | 0.2836 [0.2804, 0.2868]<br>Δ -0.0279 [-0.0307, -0.0253] LOSS | 0.3220 [0.3188, 0.3252]<br>Δ +0.0105 [+0.0088, +0.0123] win | 0.3168 [0.3137, 0.3201]<br>Δ +0.0053 [+0.0038, +0.0070] win | 0.3172 [0.3141, 0.3205]<br>Δ +0.0057 [+0.0041, +0.0074] win |
| cold_user (history < 5) (n=0) | empty | empty | empty | empty | empty |
| low_history_user (history < 20) (n=7,288) | 0.2859 [0.2749, 0.2967] | 0.2409 [0.2303, 0.2520]<br>Δ -0.0450 [-0.0555, -0.0343] LOSS | 0.2961 [0.2849, 0.3081]<br>Δ +0.0102 [+0.0022, +0.0178] win | 0.2917 [0.2803, 0.3035]<br>Δ +0.0058 [-0.0016, +0.0132] n.s. | 0.2914 [0.2801, 0.3033]<br>Δ +0.0055 [-0.0014, +0.0128] n.s. |
| higher_history_user (history >= 20) (n=140,204) | 0.3101 [0.3070, 0.3133] | 0.2834 [0.2801, 0.2865]<br>Δ -0.0267 [-0.0294, -0.0241] LOSS | 0.3207 [0.3173, 0.3240]<br>Δ +0.0106 [+0.0090, +0.0124] win | 0.3154 [0.3120, 0.3186]<br>Δ +0.0054 [+0.0038, +0.0070] win | 0.3158 [0.3124, 0.3190]<br>Δ +0.0058 [+0.0042, +0.0074] win |
| first_day (n=35,041) | 0.3028 [0.2975, 0.3083] | 0.2710 [0.2662, 0.2762]<br>Δ -0.0318 [-0.0377, -0.0262] LOSS | 0.3184 [0.3131, 0.3239]<br>Δ +0.0156 [+0.0117, +0.0197] win | 0.3048 [0.2994, 0.3101]<br>Δ +0.0020 [-0.0017, +0.0054] n.s. | 0.3056 [0.3002, 0.3108]<br>Δ +0.0027 [-0.0007, +0.0060] n.s. |
| last_day (n=37,257) | 0.3152 [0.3098, 0.3207] | 0.2878 [0.2821, 0.2936]<br>Δ -0.0274 [-0.0323, -0.0227] LOSS | 0.3193 [0.3138, 0.3247]<br>Δ +0.0041 [+0.0012, +0.0069] win | 0.3167 [0.3111, 0.3221]<br>Δ +0.0016 [-0.0013, +0.0044] n.s. | 0.3175 [0.3119, 0.3228]<br>Δ +0.0023 [-0.0004, +0.0051] n.s. |
| small_candidate_set (<= 9) (n=81,258) | 0.3722 [0.3683, 0.3762] | 0.3537 [0.3496, 0.3578]<br>Δ -0.0186 [-0.0223, -0.0150] LOSS | 0.3786 [0.3747, 0.3827]<br>Δ +0.0064 [+0.0040, +0.0088] win | 0.3742 [0.3703, 0.3781]<br>Δ +0.0020 [-0.0002, +0.0042] n.s. | 0.3744 [0.3704, 0.3783]<br>Δ +0.0022 [-0.0001, +0.0043] n.s. |
| large_candidate_set (> 9) (n=66,234) | 0.2311 [0.2274, 0.2351] | 0.1925 [0.1891, 0.1960]<br>Δ -0.0386 [-0.0427, -0.0349] LOSS | 0.2470 [0.2430, 0.2507]<br>Δ +0.0158 [+0.0133, +0.0183] win | 0.2407 [0.2366, 0.2446]<br>Δ +0.0096 [+0.0070, +0.0120] win | 0.2413 [0.2373, 0.2451]<br>Δ +0.0102 [+0.0077, +0.0125] win |

### Exploration cost: where the pick differs from the LightGBM top-1

| policy | picks differing from LightGBM top-1 | CTR of the differing picks | CTR of LightGBM's pick, same impressions | paired difference | verdict |
|---|---:|---|---|---|---|
| random | 88.4% | 0.0856 [0.0847, 0.0866] | 0.3041 [0.3010, 0.3073] | -0.2185 [-0.2221, -0.2148] | LOSS |
| popularity_greedy (per-impression) | 80.0% | 0.1769 [0.1751, 0.1788] | 0.2751 [0.2723, 0.2779] | -0.0982 [-0.1019, -0.0944] | LOSS |
| popularity_greedy (hourly) | 84.5% | 0.1369 [0.1355, 0.1384] | 0.2884 [0.2855, 0.2914] | -0.1515 [-0.1548, -0.1480] | LOSS |
| eps_greedy_lgbm | 0.3% | 0.0944 [0.0815, 0.1067] | 0.2826 [0.2629, 0.3030] | -0.1882 [-0.2156, -0.1630] | LOSS |
| linucb (per-impression) | 59.0% | 0.1910 [0.1883, 0.1938] | 0.2377 [0.2348, 0.2408] | -0.0467 [-0.0513, -0.0424] | LOSS |
| linucb (hourly) | 61.0% | 0.1761 [0.1734, 0.1790] | 0.2461 [0.2430, 0.2490] | -0.0699 [-0.0742, -0.0653] | LOSS |
| lints (per-impression) | 59.2% | 0.1912 [0.1886, 0.1938] | 0.2384 [0.2355, 0.2413] | -0.0472 [-0.0514, -0.0430] | LOSS |
| lints (hourly) | 60.7% | 0.1786 [0.1762, 0.1811] | 0.2455 [0.2426, 0.2485] | -0.0669 [-0.0708, -0.0626] | LOSS |
| hybrid_linucb (per-impression) | 25.7% | 0.2367 [0.2320, 0.2409] | 0.1953 [0.1913, 0.1998] | +0.0413 [+0.0341, +0.0478] | win |
| hybrid_linucb (hourly) | 24.1% | 0.2197 [0.2153, 0.2240] | 0.1973 [0.1931, 0.2017] | +0.0224 [+0.0155, +0.0291] | win |
| hybrid_lints (per-impression) | 25.7% | 0.2370 [0.2328, 0.2411] | 0.1958 [0.1918, 0.1996] | +0.0412 [+0.0350, +0.0478] | win |
| hybrid_lints (hourly) | 24.2% | 0.2197 [0.2156, 0.2238] | 0.1958 [0.1918, 0.1998] | +0.0239 [+0.0174, +0.0307] | win |

### SIMULATED click model (K_cls probabilities normalised per impression; Bernoulli rewards; 5 seeds; held-out stream) -- NOT log-replay

Simulator oracle expected CTR (always pick the max-probability arm): 0.3347.

| policy | cumulative expected regret (mean ± std) | simulated CTR (mean ± std) |
|---|---:|---:|
| random | 32,319 ± 40 | 0.1155 ± 0.0005 |
| popularity_greedy (per-impression) | 18,506 ± 594 | 0.2089 ± 0.0050 |
| popularity_greedy (hourly) | 23,024 ± 325 | 0.1786 ± 0.0029 |
| lgbm_greedy | 2,346 ± 0 | 0.3181 ± 0.0009 |
| eps_greedy_lgbm | 2,432 ± 4 | 0.3175 ± 0.0009 |
| linucb (per-impression) | 9,833 ± 320 | 0.2674 ± 0.0020 |
| linucb (hourly) | 10,669 ± 220 | 0.2617 ± 0.0016 |
| lints (per-impression) | 9,849 ± 372 | 0.2674 ± 0.0024 |
| lints (hourly) | 10,566 ± 287 | 0.2625 ± 0.0021 |
| hybrid_linucb (per-impression) | 2,295 ± 23 | 0.3184 ± 0.0009 |
| hybrid_linucb (hourly) | 2,353 ± 51 | 0.3180 ± 0.0008 |
| hybrid_lints (per-impression) | 2,308 ± 34 | 0.3183 ± 0.0009 |
| hybrid_lints (hourly) | 2,363 ± 45 | 0.3180 ± 0.0008 |
