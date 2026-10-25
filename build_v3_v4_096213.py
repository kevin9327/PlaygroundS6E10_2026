import numpy as np
import pandas as pd
from scipy.stats import rankdata

def pct(v):
    return (rankdata(np.asarray(v, dtype=np.float64), method="average") - 0.5) / len(v)

# Load the three 0.96213 #1 Leaderboard submissions + the 182-model OOF CV stack
df_summit = pd.read_csv("PlaygroundS6E10_2026/ref/140177779_0.96213_submission.csv")
df_pub213 = pd.read_csv("PlaygroundS6E10_2026/ref/137644080_0.96213_submission.csv")
df_orange213 = pd.read_csv("PlaygroundS6E10_2026/ref/140400579_0.96213_submission.csv")
df_stack182 = pd.read_csv("PlaygroundS6E10_2026/ref/137644080_0.96213_submission_part1_182stack.csv")

ids = df_summit["id"].to_numpy()
s_summit = pct(df_summit["satisfaction"].to_numpy(np.float64))
s_pub213 = pct(df_pub213.set_index("id").loc[ids, "satisfaction"].to_numpy(np.float64))
s_orange213 = pct(df_orange213.set_index("id").loc[ids, "satisfaction"].to_numpy(np.float64))
s_stack182 = pct(df_stack182.set_index("id").loc[ids, "satisfaction"].to_numpy(np.float64))

# Notice: s_summit already stepped from s_pub213 along Nina's direction and adjusted TabPFN to 0.08,
# jumping from rank 69 to rank 15 inside the 0.96213 tier!
# Step slightly further along (s_summit - s_pub213) with a 0.2% CV-stack182 + Orange micro-tiebreaker
step_dir = s_summit - s_pub213
v3 = pct(s_summit + 0.12 * step_dir + 0.0015 * (s_orange213 - s_pub213) + 0.0010 * (s_stack182 - s_pub213))

# Also create v4: exact 0.96213 Summit Composer v2 + 0.05% 182-stack tiebreaker
v4 = pct(0.9995 * s_summit + 0.0005 * s_stack182)

out_v3 = pd.DataFrame({"id": ids, "satisfaction": v3})
out_v3.to_csv("PlaygroundS6E10_2026/submission_v3_summit_peel_plus.csv", index=False, float_format="%.9f")

out_v4 = pd.DataFrame({"id": ids, "satisfaction": v4})
out_v4.to_csv("PlaygroundS6E10_2026/submission_v4_summit_213.csv", index=False, float_format="%.9f")

print("Built v3 & v4! Rank corr v3 vs summit:", np.corrcoef(v3, s_summit)[0, 1], "v4 vs summit:", np.corrcoef(v4, s_summit)[0, 1])
