import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("sales.csv")

# 1. Basic Info
print(df.info())
print(df.describe())

# 2. Missing values
print("\nMissing values per column:\n", df.isnull().sum())

# 3. Correlation heatmap
corr = df.select_dtypes(include=['number']).corr()
print("\nCorrelation matrix:\n", corr)
sns.heatmap(corr, annot=True)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("week5_correlation_heatmap.png")
plt.close()

# 4. Region vs Sales barplot
plt.figure(figsize=(8,5))
sns.barplot(x="Region", y="Sales", data=df, estimator=sum)
plt.title("Total Sales by Region")
plt.tight_layout()
plt.savefig("week5_sales_by_region.png")
plt.close()

print("EDA plots saved: week5_correlation_heatmap.png, week5_sales_by_region.png")
