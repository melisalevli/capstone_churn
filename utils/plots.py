import matplotlib.pyplot as plt
import os
import re

def plot_curves(roc_data, pr_data, model_name="Model"):
    fpr, tpr = roc_data
    recall, precision = pr_data

    plt.figure(figsize=(12, 5))

    
    plt.subplot(1, 2, 1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label=f'{model_name}')
    plt.title(f'ROC Curve - {model_name}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()

    
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, label=f'{model_name}')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()

    plt.tight_layout()

    
    os.makedirs("figures", exist_ok=True)
    safe_name = re.sub(r'[^A-Za-z0-9_]', '_', model_name)
    save_path = f"figures/rocpr_{safe_name}.png"
    plt.savefig(save_path)
    print(f" ROC+PR graph recorded: {save_path}")

    
    plt.show()
