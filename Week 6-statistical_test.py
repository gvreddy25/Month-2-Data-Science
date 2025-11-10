import pandas as pd
from scipy import stats

df = pd.read_csv("../sales.csv")

north = df[df["Region"] == "North"]["Sales"]
south = df[df["Region"] == "South"]["Sales"]

print("North sample size:", north.size, "South sample size:", south.size)

# Perform independent t-test (Welch's t-test)
t_stat, p_val = stats.ttest_ind(north, south, equal_var=False, nan_policy='omit')
print(f"T-statistic: {t_stat:.4f}, P-value: {p_val:.4f}")

alpha = 0.05
if p_val < alpha:
    print("Reject Null Hypothesis: Sales are significantly different between North and South.")
else:
    print("Fail to Reject Null Hypothesis: No significant difference found.")
