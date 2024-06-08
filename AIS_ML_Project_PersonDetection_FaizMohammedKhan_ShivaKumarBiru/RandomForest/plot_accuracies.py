import matplotlib.pyplot as plt

# Plotting the Accuracy data
categories = ['Constant Leaned', 'Constant Normal', 'Moving Leaned', 'Moving Normal']
accuracy_with_threshold_100 = [89.88, 90.88, 90.12, 89.99]
accuracy_with_threshold_110 = [90.12, 90.43, 90.43, 90.10]
accuracy_without_threshold_100 = [88.88, 88.99, 89.54, 88.88]
accuracy_without_threshold_110 = [89.24, 89.65, 89.65, 89.88]

# Creating subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Accuracy Comparison')


colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']


for ax, category, acc_100, acc_110, acc_100_no, acc_110_no in zip(
        axs.flatten(),
        categories,
        accuracy_with_threshold_100,
        accuracy_with_threshold_110,
        accuracy_without_threshold_100,
        accuracy_without_threshold_110
):
    ax.bar(['WithThreshold/100', 'WithThreshold/110', 'WithoutThreshold/100', 'WithoutThreshold/110'],
           [acc_100, acc_110, acc_100_no, acc_110_no], color=colors)
    ax.set_title(category)
    for i, acc in enumerate([acc_100, acc_110, acc_100_no, acc_110_no]):
        ax.text(i, acc + 0.5, f'{acc:.2f}%', ha='center', va='bottom', color='black')

    max_value = max(max(accuracy_with_threshold_100), max(accuracy_with_threshold_110),
                    max(accuracy_without_threshold_100), max(accuracy_without_threshold_110))
    ax.set_ylim(80, max_value + 2)  # Adding 2 for better visualization

plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.show()
