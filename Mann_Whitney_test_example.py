from scipy.stats import mannwhitneyu

# Data for Control and Test Groups
control_group = [25, 30, 28, 29, 27]
test_group = [20, 22, 18, 21, 19]

# Perform Mann-Whitney U Test
u_stat, p_value = mannwhitneyu(control_group, test_group, alternative='two-sided')

print(f"Mann-Whitney U Statistic: {u_stat}")
print(f"P-Value: {p_value}")

# Conclusion
if p_value < 0.05:
    print("Reject Null Hypothesis: Significant difference between the two groups.")
else:
    print("Fail to Reject Null Hypothesis: No significant difference between the groups.")
