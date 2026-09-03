eval set: ../evaluation/newdata_eval/eval_sets/test_agentic_clean.txt (204 shapes)  |  recipe: encoder mean, fp32, strict load, per-shape then mean; MAE in [0,1], L1 in [-1,1], PSNR [0,1] cap 40 dB; lit = GT max channel > 1/255
runs this invocation: {'pretrained': {'ok': 612, 'oom': 0, 'error': 0}, 'pbr2emission': {'ok': 204, 'oom': 0, 'error': 0}, 'albedo2emission_v2': {'ok': 204, 'oom': 0, 'error': 0}, 'albedo2emission_v3': {'ok': 204, 'oom': 0, 'error': 0}}

Marginal means on the 204 shapes every arm scored (mean / median). MAE is in [0,1] emission space (the paper's convention); L1 is the same error in the [-1,1] training space (= 2x MAE unless the decoder overshoots); PSNR in [0,1], cap 40 dB; 'lit' columns are GT-mask quantities (GT max channel > 1/255), like the evaluator's `_gton` columns.

| arm | n | failed | MAE all | L1 [-1,1] all | PSNR all | MAE lit | L1 [-1,1] lit | PSNR lit | IoU | pred/GT mean (lit) |
|---|---|---|---|---|---|---|---|---|---|---|
| pretrained_pad | 204 | 0 | 0.0024 / 0.0009 | 0.0051 / 0.0018 | 37.60 / 40.00 | 0.0440 / 0.0328 | 0.0906 / 0.0670 | 24.66 / 23.98 | 0.683 / 0.711 | 0.261 / 0.279 |
| ours_pbr2emission | 204 | 0 | 0.0017 / 0.0003 | 0.0037 / 0.0009 | 38.14 / 40.00 | 0.0307 / 0.0260 | 0.0623 / 0.0533 | 27.29 / 26.48 | 0.745 / 0.776 | 0.270 / 0.279 |

Paired vs `pretrained_pad` (same shapes; delta = arm - ref; 95% bootstrap CI over shapes; win = fraction of shapes where the arm is strictly better):

| arm | metric | n | mean delta [95% CI] | median delta | win |
|---|---|---|---|---|---|
| ours_pbr2emission | mae01_all | 204 | -0.0007 [-0.0009, -0.0006] | -0.0005 | 100% |
| ours_pbr2emission | mae01_lit | 204 | -0.0134 [-0.0170, -0.0103] | -0.0055 | 94% |
| ours_pbr2emission | psnr_lit | 204 | +2.6301 [+2.2936, +2.9867] | +2.0381 | 88% |
| ours_pbr2emission | iou | 204 | +0.0616 [+0.0515, +0.0721] | +0.0497 | 85% |
