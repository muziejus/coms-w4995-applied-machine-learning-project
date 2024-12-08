import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pickle

file_paths = ['lstm_results.pkl', 'random_forest_results.pkl']
models = ['LSTM', 'Random Forest']
companies = ['dltr', 'lulu']
titles = ['Dollar Tree, LSTM', 'Dollar Tree, Random Forest', 'Lululemon, LSTM', 'Lululemon, Random Forest']

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
axes = axes.flatten()

plot_index = 0
for file_path, model_name in zip(file_paths, models):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    
    for company in companies:
        # Extract data
        company_data = next(item for item in data if item['company'] == company)
        y_test = company_data['y_test']
        probs = company_data['probs']

        # Calculate the ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, probs)

        # Calculate the AUC
        roc_auc = auc(fpr, tpr)

        # Plot
        ax = axes[plot_index]
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_title(f'ROC Curve for {titles[plot_index]}')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc='lower right')

        plot_index += 1

plt.tight_layout()
plt.show()
