### Retrained GBDT: training summary (all validation days; the models never saw validation)

| D | best iteration | stream AUC (all 7 days) | top-1 (all 7 days) | share of gain from the 6 trailing features |
|---|---:|---:|---:|---:|
| 0 min | 950 | 0.8173 | 0.4077 | 0.379 |
| 15 min | 513 | 0.8028 | 0.3859 | 0.303 |
| 30 min | 426 | 0.7943 | 0.3724 | 0.248 |
| 60 min | 462 | 0.7827 | 0.3536 | 0.183 |

Original ranker on the same stream: AUC 0.7588, top-1 0.3168.

### D = 0 min — held-out days 3-6 (n = 147,492); selections made on days 0-2

Blend chosen on days 0-2: 2h window, beta = 2.0. Forgetting hybrid re-tuned under this delay: gamma = 0.999, lambda = 1.

| policy | P@1 [CI] | Δ vs original [CI] | Δ vs forgetting hybrid [CI] | final-20% P@1 | final-20% Δ vs original | final-20% Δ vs hybrid |
|---|---|---|---|---|---|---|
| LightGBM-greedy (original) | 0.3089 [0.3058, 0.3120] | - | -0.0469 [-0.0492, -0.0448] | 0.3169 | - | -0.0457 [-0.0507, -0.0408] |
| LightGBM + trailing CTR (retrained) | 0.3995 [0.3961, 0.4029] | +0.0906 [+0.0880, +0.0932] | +0.0437 [+0.0410, +0.0464] | 0.4166 | +0.0997 [+0.0943, +0.1052] | +0.0539 [+0.0486, +0.0595] |
| popularity blend (no learning) | 0.3415 [0.3384, 0.3448] | +0.0326 [+0.0306, +0.0345] | -0.0143 [-0.0167, -0.0120] | 0.3402 | +0.0233 [+0.0196, +0.0270] | -0.0224 [-0.0273, -0.0176] |
| forgetting hybrid LinUCB | 0.3558 [0.3527, 0.3590] | +0.0469 [+0.0448, +0.0492] | - | 0.3626 | +0.0457 [+0.0408, +0.0507] | - |

Stream AUC on days 3-6: original 0.7560 [0.7540, 0.7579], retrained 0.8167 [0.8149, 0.8185], paired Δ +0.0608 [+0.0597, +0.0618]. Per-day P@1 (days 3-6): LightGBM-greedy 0.303/0.303/0.314/0.315; LightGBM + trailing CTR 0.390/0.390/0.404/0.414; popularity blend 0.337/0.332/0.352/0.344; forgetting hybrid LinUCB 0.348/0.353/0.360/0.361.

### D = 15 min — held-out days 3-6 (n = 147,492); selections made on days 0-2

Blend chosen on days 0-2: 2h window, beta = 2.0. Forgetting hybrid re-tuned under this delay: gamma = 0.9995, lambda = 10.

| policy | P@1 [CI] | Δ vs original [CI] | Δ vs forgetting hybrid [CI] | final-20% P@1 | final-20% Δ vs original | final-20% Δ vs hybrid |
|---|---|---|---|---|---|---|
| LightGBM-greedy (original) | 0.3089 [0.3058, 0.3120] | - | -0.0305 [-0.0327, -0.0285] | 0.3169 | - | -0.0314 [-0.0363, -0.0265] |
| LightGBM + trailing CTR (retrained) | 0.3786 [0.3754, 0.3820] | +0.0698 [+0.0675, +0.0722] | +0.0393 [+0.0368, +0.0415] | 0.3867 | +0.0698 [+0.0648, +0.0750] | +0.0384 [+0.0333, +0.0435] |
| popularity blend (no learning) | 0.3357 [0.3326, 0.3390] | +0.0268 [+0.0249, +0.0287] | -0.0037 [-0.0060, -0.0015] | 0.3367 | +0.0198 [+0.0159, +0.0236] | -0.0116 [-0.0166, -0.0064] |
| forgetting hybrid LinUCB | 0.3394 [0.3365, 0.3426] | +0.0305 [+0.0285, +0.0327] | - | 0.3483 | +0.0314 [+0.0265, +0.0363] | - |

Stream AUC on days 3-6: original 0.7560 [0.7540, 0.7579], retrained 0.8021 [0.8002, 0.8039], paired Δ +0.0461 [+0.0451, +0.0471]. Per-day P@1 (days 3-6): LightGBM-greedy 0.303/0.303/0.314/0.315; LightGBM + trailing CTR 0.373/0.371/0.385/0.385; popularity blend 0.332/0.327/0.345/0.339; forgetting hybrid LinUCB 0.326/0.341/0.342/0.347.

### D = 30 min — held-out days 3-6 (n = 147,492); selections made on days 0-2

Blend chosen on days 0-2: 2h window, beta = 2.0. Forgetting hybrid re-tuned under this delay: gamma = 0.9995, lambda = 10.

| policy | P@1 [CI] | Δ vs original [CI] | Δ vs forgetting hybrid [CI] | final-20% P@1 | final-20% Δ vs original | final-20% Δ vs hybrid |
|---|---|---|---|---|---|---|
| LightGBM-greedy (original) | 0.3089 [0.3058, 0.3120] | - | -0.0197 [-0.0219, -0.0175] | 0.3169 | - | -0.0249 [-0.0300, -0.0200] |
| LightGBM + trailing CTR (retrained) | 0.3659 [0.3627, 0.3691] | +0.0570 [+0.0548, +0.0591] | +0.0373 [+0.0349, +0.0397] | 0.3736 | +0.0567 [+0.0520, +0.0615] | +0.0318 [+0.0264, +0.0370] |
| popularity blend (no learning) | 0.3317 [0.3286, 0.3351] | +0.0229 [+0.0209, +0.0247] | +0.0032 [+0.0008, +0.0055] | 0.3343 | +0.0174 [+0.0136, +0.0212] | -0.0075 [-0.0124, -0.0027] |
| forgetting hybrid LinUCB | 0.3286 [0.3255, 0.3316] | +0.0197 [+0.0175, +0.0219] | - | 0.3418 | +0.0249 [+0.0200, +0.0300] | - |

Stream AUC on days 3-6: original 0.7560 [0.7540, 0.7579], retrained 0.7933 [0.7914, 0.7951], paired Δ +0.0374 [+0.0364, +0.0383]. Per-day P@1 (days 3-6): LightGBM-greedy 0.303/0.303/0.314/0.315; LightGBM + trailing CTR 0.360/0.362/0.371/0.370; popularity blend 0.329/0.323/0.340/0.334; forgetting hybrid LinUCB 0.312/0.328/0.332/0.341.

### D = 60 min — held-out days 3-6 (n = 147,492); selections made on days 0-2

Blend chosen on days 0-2: 2h window, beta = 1.0. Forgetting hybrid re-tuned under this delay: gamma = 0.9999, lambda = 10.

| policy | P@1 [CI] | Δ vs original [CI] | Δ vs forgetting hybrid [CI] | final-20% P@1 | final-20% Δ vs original | final-20% Δ vs hybrid |
|---|---|---|---|---|---|---|
| LightGBM-greedy (original) | 0.3089 [0.3058, 0.3120] | - | -0.0049 [-0.0069, -0.0028] | 0.3169 | - | -0.0132 [-0.0176, -0.0087] |
| LightGBM + trailing CTR (retrained) | 0.3485 [0.3453, 0.3519] | +0.0397 [+0.0377, +0.0417] | +0.0348 [+0.0325, +0.0372] | 0.3556 | +0.0387 [+0.0345, +0.0432] | +0.0255 [+0.0202, +0.0305] |
| popularity blend (no learning) | 0.3261 [0.3231, 0.3294] | +0.0172 [+0.0158, +0.0186] | +0.0124 [+0.0103, +0.0146] | 0.3299 | +0.0130 [+0.0103, +0.0158] | -0.0002 [-0.0049, +0.0044] |
| forgetting hybrid LinUCB | 0.3137 [0.3105, 0.3168] | +0.0049 [+0.0028, +0.0069] | - | 0.3301 | +0.0132 [+0.0087, +0.0176] | - |

Stream AUC on days 3-6: original 0.7560 [0.7540, 0.7579], retrained 0.7815 [0.7795, 0.7834], paired Δ +0.0255 [+0.0247, +0.0263]. Per-day P@1 (days 3-6): LightGBM-greedy 0.303/0.303/0.314/0.315; LightGBM + trailing CTR 0.344/0.340/0.357/0.352; popularity blend 0.322/0.317/0.335/0.329; forgetting hybrid LinUCB 0.289/0.321/0.315/0.328.

### Verdict per D (retrained GBDT vs forgetting hybrid, paired, overall / final 20%)

| D | overall | final 20% |
|---|---|---|
| 0 min | **beats** the hybrid (+0.0437 [+0.0410, +0.0464]) | **beats** the hybrid (+0.0539 [+0.0486, +0.0595]) |
| 15 min | **beats** the hybrid (+0.0393 [+0.0368, +0.0415]) | **beats** the hybrid (+0.0384 [+0.0333, +0.0435]) |
| 30 min | **beats** the hybrid (+0.0373 [+0.0349, +0.0397]) | **beats** the hybrid (+0.0318 [+0.0264, +0.0370]) |
| 60 min | **beats** the hybrid (+0.0348 [+0.0325, +0.0372]) | **beats** the hybrid (+0.0255 [+0.0202, +0.0305]) |

### Causality audit of the trained models (end to end)

| D | rows with t ≤ T0+D checked | max \|score change\| on them | later rows whose score changed | placebo top-1 (permuted-label features) | real top-1 | original ranker top-1 |
|---|---:|---:|---:|---:|---:|---:|
| D=0 | 1,555,133 | 0.0e+00 | 0.994 | 0.3092 | 0.4077 | 0.3168 |
| D=900 | 1,564,487 | 0.0e+00 | 0.989 | 0.3113 | 0.3859 | 0.3168 |
| D=3600 | 1,588,275 | 0.0e+00 | 0.917 | 0.3194 | 0.3536 | 0.3168 |
