import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt

# Observed data
observed = np.array([[1550, 29450], [1400, 29600]])

# Perform Chi-Square Test
chi2, p, dof, expected = chi2_contingency(observed)

print(f"Chi-square Statistic: {chi2}")
print(f"P-Value: {p}")

# Visualization
groups = ['Control', 'Test']
sign_up_rates = [1550 / 31000, 1400 / 31000] # Calculate rates
plt.bar(groups, sign_up_rates, color=['blue', 'orange'])
plt.title("Sign-Up Rate by Group")
plt.ylabel("Sign-Up Rate")
plt.show()