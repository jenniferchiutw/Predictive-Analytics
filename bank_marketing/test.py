

import matplotlib.pyplot as plt

features = [
    "duration", "housing", "contact", "month", "balance", "day",
    "age", "job", "campaign", "loan", "pdays", "poutcome",
    "education", "marital", "previous", "default"
]

importance = [
    0.32367333722759123, 0.11613892436094676, 0.07822421275708293,
    0.07387938902337987, 0.06773280446640402, 0.06038880610391017,
    0.05238310966023013, 0.04421818798448991, 0.038267889302401395,
    0.034526686672226206, 0.03178043342176862, 0.028476833764724284,
    0.020852161501590593, 0.015191171685438532, 0.013088260615700604,
    0.0011777914521146906
]

plt.figure(figsize=(10, 6))
plt.barh(features, importance, color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance for Random Forest')
plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature on top
plt.show()
