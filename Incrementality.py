# Python Implementation Example: A/B Testing for Incrementality
# Results:
#Incremental Lift: 5.00%
#Z-Statistic: 4.52, P-Value: 0.0000
#The results are statistically significant. The campaign caused the lift.

import scipy.stats as stats
from statsmodels.stats.proportion import proportions_ztest

# Conversion data
test_group = {'views': 1000, 'conversions': 150}
control_group = {'views': 4000, 'conversions': 400}

# Conversion rates
test_rate = test_group['conversions'] / test_group['views']
control_rate = control_group['conversions'] / control_group['views']

# Incremental lift
incremental_lift = test_rate - control_rate
print(f"Incremental Lift: {incremental_lift:.2%}")

# Hypothesis testing (two-proportion z-test)
successes = [test_group['conversions'], control_group['conversions']]
samples = [test_group['views'], control_group['views']]
z_stat, p_value = proportions_ztest(successes, samples)

print(f"Z-Statistic: {z_stat:.2f}, P-Value: {p_value:.4f}")

# Conclusion
if p_value < 0.05:
    print("The results are statistically significant. The campaign caused the lift.")
else:
    print("The results are not statistically significant. The lift may be due to chance.")
