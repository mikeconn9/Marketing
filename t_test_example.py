from scipy.stats import ttest_ind

# Data for Control and Test Groups
control_group = [25, 30, 28, 29, 27, 26]
test_group = [20, 22, 18, 21, 19, 17]

# Perform t-test
t_stat, p_value = ttest_ind(control_group, test_group)

print(f"T-Statistic: {t_stat}")
print(f"P-Value: {p_value}")

# Conclusion
if p_value < 0.05:
    print("Reject Null Hypothesis: There is a significant difference between the two groups.")
else:
    print("Fail to Reject Null Hypothesis: No significant difference observed.")

# Results
#T-Statistic: 7.406560798180412
#P-Value: 2.29759532685201e-05
#Reject Null Hypothesis: There is a significant difference between the two groups.
