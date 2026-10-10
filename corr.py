import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# Load data
clinical = pd.read_csv("data/CFB-GBM_clinical_data_v03_20260619.tsv", sep='\t')
treatment = pd.read_csv("data/CFB-GBM_treatment_data_v02_20260129.tsv", sep='\t')
rano = pd.read_csv("data/CFB-GBM_rano_criteria_v04_20260909.tsv", sep='\t')
features = pd.read_csv("data/CFB_GBM_features_extraction_pyradiomics_v03_20260619.tsv", sep='\t')

survival_col = clinical.columns[1]  # "survival (weeks)"
survival = clinical[["id_patient", survival_col]]

features = features.rename(columns={"Patient": "id_patient"})

results = []

for label, df in [("treatment", treatment), ("rano", rano), ("features", features)]:
    merged = pd.merge(survival, df, on="id_patient", how="inner")
    cols = [c for c in df.columns if c != "id_patient"]
    for col in cols:
        series = pd.to_numeric(merged[col], errors="coerce")
        valid = merged[[survival_col]].join(series.rename("_val")).dropna()
        if len(valid) >= 3:
            pearson_r = np.corrcoef(valid[survival_col], valid["_val"])[0, 1]
            spearman_r, spearman_p = spearmanr(valid[survival_col], valid["_val"])
        else:
            pearson_r = np.nan
            spearman_r = np.nan
            spearman_p = np.nan
        results.append({"source_file": label, "column": col,
                         "pearson_r": pearson_r, "spearman_r": spearman_r,
                         "spearman_p": spearman_p, "n": len(valid)})

res = pd.DataFrame(results)
res["abs_r"] = res["pearson_r"].abs()
res = res.sort_values(["source_file", "abs_r"], ascending=[True, False])

# Print table
for src in ["treatment", "rano", "features"]:
    sub = res[res["source_file"] == src]
    print(f"\n{'='*70}")
    print(f"  {src.upper()} — correlation with {survival_col}")
    print(f"{'='*70}")
    print(f"  {'Column':<40} {'Pearson':>8} {'Spearman':>9} {'p-val':>9}  {'n':>5}")
    print(f"  {'-'*40} {'-'*8} {'-'*9} {'-'*9}  {'-'*5}")
    for _, row in sub.iterrows():
        p_str = f"{row['pearson_r']:.4f}" if not np.isnan(row['pearson_r']) else "    N/A"
        s_str = f"{row['spearman_r']:.4f}" if not np.isnan(row['spearman_r']) else "     N/A"
        pv_str = f"{row['spearman_p']:.2e}" if not np.isnan(row['spearman_p']) else "     N/A"
        print(f"  {row['column']:<40} {p_str:>8} {s_str:>9} {pv_str:>9}  {int(row['n']):>5}")

# Save to CSV
res.to_csv("survival_correlations.csv", index=False)
print(f"\nResults saved to survival_correlations.csv")

# --- Figure: top 20 strongest correlations (Pearson & Spearman side by side) ---
res["abs_spearman"] = res["spearman_r"].abs()
top = res.dropna(subset=["pearson_r", "spearman_r"]).nlargest(20, "abs_spearman")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
colors = {"treatment": "#4C72B0", "rano": "#DD8452", "features": "#55A868"}
y_pos = range(len(top))
clrs = [colors[s] for s in top["source_file"]]

ax1.barh(y_pos, top["pearson_r"].values, color=clrs)
ax1.set_xlabel("Pearson r")
ax1.set_title("Pearson")
ax1.invert_yaxis()
ax1.axvline(0, color="black", linewidth=0.5)
ax1.set_yticks(y_pos)
ax1.set_yticklabels([row["column"] for _, row in top.iterrows()], fontsize=9)

ax2.barh(y_pos, top["spearman_r"].values, color=clrs)
ax2.set_xlabel("Spearman ρ")
ax2.set_title("Spearman")
ax2.axvline(0, color="black", linewidth=0.5)

from matplotlib.patches import Patch
ax2.legend(handles=[Patch(color=c, label=l) for l, c in colors.items()],
           loc="lower right", fontsize=9)

fig.suptitle(f"Top 20 Correlations with {survival_col}", fontsize=13)
plt.tight_layout()
plt.savefig("survival_correlations.png", dpi=150)
print("Figure saved to survival_correlations.png")