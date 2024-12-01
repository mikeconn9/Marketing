import pandas as pd
import numpy as np

# Simulated sales data for three campaigns
np.random.seed(42)
campaign_a = np.random.normal(loc=200, scale=20, size=30)  # Mean = 200, Std Dev = 20
campaign_b = np.random.normal(loc=210, scale=25, size=30)  # Mean = 210, Std Dev = 25
campaign_c = np.random.normal(loc=190, scale=15, size=30)  # Mean = 190, Std Dev = 15

# Combine data into a DataFrame
data = pd.DataFrame({
    "Sales": np.concatenate([campaign_a, campaign_b, campaign_c]),
    "Campaign": ["A"] * 30 + ["B"] * 30 + ["C"] * 30
})

import seaborn as sns
import matplotlib.pyplot as plt

# Boxplot to visualize the sales distribution across campaigns
sns.boxplot(x="Campaign", y="Sales", data=data, palette="Set2")
plt.title("Sales Distribution by Campaign")
plt.show()

from scipy.stats import f_oneway

# Separate sales data by campaign
sales_a = data[data["Campaign"] == "A"]["Sales"]
sales_b = data[data["Campaign"] == "B"]["Sales"]
sales_c = data[data["Campaign"] == "C"]["Sales"]

# Perform one-way ANOVA
f_stat, p_value = f_oneway(sales_a, sales_b, sales_c)

print(f"F-statistic: {f_stat:.2f}")
print(f"p-value: {p_value:.4f}")

from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Perform Tukey's HSD test
tukey = pairwise_tukeyhsd(endog=data["Sales"], groups=data["Campaign"], alpha=0.05)
print(tukey)

# Plot the results
tukey.plot_simultaneous()
plt.title("Tukey's HSD Test Results")
plt.show()