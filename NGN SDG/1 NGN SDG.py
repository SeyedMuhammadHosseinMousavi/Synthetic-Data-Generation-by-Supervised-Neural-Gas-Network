%reset -f
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time
import seaborn as sns
from matplotlib.font_manager import FontProperties

# Start time for the overall process
start_time = time.time()

class SupervisedNeuralGas:
    def __init__(self, n_units_per_class=10, max_iter=200, eta_start=0.5, eta_end=0.01, lambda_start=30, lambda_end=0.1):
        self.n_units_per_class = n_units_per_class
        self.max_iter = max_iter
        self.eta_start = eta_start
        self.eta_end = eta_end
        self.lambda_start = lambda_start
        self.lambda_end = lambda_end
        self.units = {}
        self.unit_labels = {}
        self.loss_history = {}  # Added to track loss per class

    def _update_learning_rate(self, i):
        return self.eta_start * (self.eta_end / self.eta_start) ** (i / self.max_iter)

    def _update_neighborhood_range(self, i):
        return self.lambda_start * (self.lambda_end / self.lambda_start) ** (i / self.max_iter)

    def train(self, data, labels):
        unique_labels = np.unique(labels)
        n_samples, n_features = data.shape

        for label in unique_labels:
            self.units[label] = np.random.rand(self.n_units_per_class, n_features)
            indices = np.where(labels == label)[0]
            self.loss_history[label] = []  # Initialize loss list for this label

            for i in range(self.max_iter):
                eta = self._update_learning_rate(i)
                lambd = self._update_neighborhood_range(i)
                loss = 0

                np.random.shuffle(indices)
                for index in indices:
                    x = data[index]
                    dists = np.linalg.norm(self.units[label] - x, axis=1)
                    ranking = np.argsort(dists)
                    loss += dists[ranking[0]]  # Summing the smallest distance for loss
                    for rank, idx in enumerate(ranking):
                        influence = np.exp(-rank / lambd)
                        self.units[label][idx] += eta * influence * (x - self.units[label][idx])

                self.loss_history[label].append(loss / len(indices))  # Average loss per iteration
                if (i + 1) % 10 == 0:
                    print(f"Iteration {i+1}/{self.max_iter} for class {label}")

    def generate_synthetic_data(self, n_samples=100, noise_level=0.1):
        synthetic_data = []
        synthetic_labels = []
        for label in self.units:
            n_units, n_features = self.units[label].shape
            samples_per_unit = n_samples // n_units

            for unit in self.units[label]:
                for _ in range(samples_per_unit):
                    noise = np.random.randn(n_features) * noise_level
                    synthetic_data.append(unit + noise)
                    synthetic_labels.append(label)

        return np.array(synthetic_data), np.array(synthetic_labels)

# Load CSV data
data_path = 'tr.csv'
data_df = pd.read_csv(data_path)
data = data_df.iloc[:, :-1].values
target = data_df.iloc[:, -1].values

# Normalize data
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data)

# Train Supervised Neural Gas
sng = SupervisedNeuralGas(n_units_per_class=9, max_iter=100)
sng.train(data_normalized, target)

# Generate synthetic data
synthetic_data, synthetic_labels = sng.generate_synthetic_data(n_samples=500, noise_level=0.01)


# # Plotting ngn training loss over iterations for each class----------------------
# plt.figure(figsize=(6, 5 * len(sng.loss_history)))
# for i, (label, losses) in enumerate(sng.loss_history.items(), 1):
#     plt.subplot(len(sng.loss_history), 1, i)
#     plt.plot(losses, label=f'Training Loss for Class {label}', linewidth=2)
#     plt.title(f'Class {label} Training Loss', fontsize=14, fontweight='bold')
#     plt.xlabel('Iteration', fontsize=12, fontweight='bold')
#     plt.ylabel('Average Distance Loss', fontsize=12, fontweight='bold')
#     plt.legend(fontsize=12)
#     plt.grid(True)
#     plt.xticks(fontsize=10, fontweight='bold')   
#     plt.yticks(fontsize=10, fontweight='bold')   
# plt.tight_layout()
# plt.show()


# Save synthetic dataset as csv--------------------------
import pandas as pd
# Combine synthetic data and labels
synthetic_data_with_labels = np.hstack((synthetic_data, synthetic_labels.reshape(-1, 1)))
# Create a DataFrame
columns = [f'feature_{i}' for i in range(synthetic_data.shape[1])] + ['label']
synthetic_df = pd.DataFrame(synthetic_data_with_labels, columns=columns)
# Save the DataFrame to a CSV file
synthetic_df.to_csv('Synthetic_data_NGN.csv', index=False)


# TSNE-----------------
# from sklearn.manifold import TSNE
# import numpy as np
# import matplotlib.pyplot as plt
# # Determine number of unique classes for better color differentiation
# unique_labels = np.unique(synthetic_labels)
# n_classes = len(unique_labels)
# # t-SNE Analysis with adjusted parameters
# tsne = TSNE(n_components=2, perplexity=max(5, n_classes * 3), early_exaggeration=12, 
#             learning_rate=200, n_iter=500, random_state=42, init='pca', angle=0.5)

# synthetic_data_tsne = tsne.fit_transform(synthetic_data)
# plt.figure(figsize=(7, 5))
# plt.scatter(synthetic_data_tsne[:, 0], synthetic_data_tsne[:, 1], 
#             c=synthetic_labels, cmap='viridis', alpha=0.5, edgecolor='k',s=100)
# # Enhance the plot
# plt.title('Brain Wave EEG', fontsize=16, fontweight='bold')
# plt.xlabel('t-SNE Feature 1', fontsize=14, fontweight='bold')
# plt.ylabel('t-SNE Feature 2', fontsize=14, fontweight='bold')
# plt.xticks(fontsize=12, fontweight='bold')
# plt.yticks(fontsize=12, fontweight='bold')
# plt.grid(True)
# plt.show()




# import matplotlib.pyplot as plt
# import numpy as np
# plt.figure(figsize=(14, 3))
# # Feature indices to plot
# f1 = 3
# f2 = 4
# # Plot original data
# plt.subplot(1, 2, 1)
# for label in np.unique(target):
#     plt.scatter(data_normalized[target == label, f1], data_normalized[target == label, f2], label=f'Class {label}')
# plt.title('Original Data', fontsize=16, fontweight='bold')
# plt.xlabel('Feature 1', fontsize=14, fontweight='bold')
# plt.ylabel('Feature 2', fontsize=14, fontweight='bold')
# plt.legend(fontsize=12)
# plt.xticks(fontsize=12, fontweight='bold')
# plt.yticks(fontsize=12, fontweight='bold')
# plt.grid(True)
# # Plot synthetic data
# plt.subplot(1, 2, 2)
# for label in np.unique(synthetic_labels):
#     plt.scatter(synthetic_data[synthetic_labels == label, f1], synthetic_data[synthetic_labels == label, f2], label=f'Class {label}')
# plt.title('Synthetic Data', fontsize=16, fontweight='bold')
# plt.xlabel('Feature 1', fontsize=14, fontweight='bold')
# plt.ylabel('Feature 2', fontsize=14, fontweight='bold')
# plt.legend(fontsize=12)
# plt.xticks(fontsize=12, fontweight='bold')
# plt.yticks(fontsize=12, fontweight='bold')
# plt.grid(True)
# plt.show()


# Print the runtime
end_time = time.time()
runtime_seconds = end_time - start_time
runtime_minutes = runtime_seconds / 60
print(f"\nTotal Runtime: {runtime_seconds:.2f} seconds ({runtime_minutes:.2f} minutes)")



# XGB Classification--------------------------------
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import xgboost as xgb

def run_experiments(data, labels):
    labels -= np.min(labels)  # Normalize label indices to start at 0
    
    accuracies = []
    all_confusion_matrices = []  # To store confusion matrices from each run
    all_y_true = []  # To store all true labels
    all_y_pred = []  # To store all predicted labels
    
    for _ in range(5):  # Running the experiment 5 times
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=None)
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        cm = confusion_matrix(y_test, y_pred)
        all_confusion_matrices.append(cm)
    
    average_confusion_matrix = np.mean(all_confusion_matrices, axis=0)
    classification_rep = classification_report(all_y_true, all_y_pred)
    average_accuracy = np.mean(accuracies)
    std_deviation = np.std(accuracies)
    
    return accuracies, average_accuracy, std_deviation, average_confusion_matrix, classification_rep

original_accuracies, original_avg_accuracy, original_std, original_avg_cm, original_classification_rep = run_experiments(data_normalized, target)
synthetic_accuracies, synthetic_avg_accuracy, synthetic_std, synthetic_avg_cm, synthetic_classification_rep = run_experiments(synthetic_data, synthetic_labels)
combined_data = np.vstack((data_normalized, synthetic_data))
combined_labels = np.concatenate((target, synthetic_labels))
combined_accuracies, combined_avg_accuracy, combined_std, combined_avg_cm, combined_classification_rep = run_experiments(combined_data, combined_labels)

# Plotting violin plots for all three datasets in one figure
# Set up bold font properties
font_bold = FontProperties()
font_bold.set_weight('bold')
font_bold.set_size(12)

# Set up the figure
plt.figure(figsize=(18, 4))

# Plot for Original Data Accuracies
plt.subplot(1, 3, 1)
sns.violinplot(data=original_accuracies, color='violet', linewidth=2)
plt.title('Original Data Accuracy', fontsize=16, fontweight='bold')
plt.xlabel('Experiment', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy', fontsize=14, fontweight='bold')
plt.xticks(fontproperties=font_bold)
plt.yticks(fontproperties=font_bold)

# Plot for Synthetic Data Accuracies
plt.subplot(1, 3, 2)
sns.violinplot(data=synthetic_accuracies, color='Purple', linewidth=2)
plt.title('Synthetic Data Accuracy', fontsize=16, fontweight='bold')
plt.xlabel('Experiment', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy', fontsize=14, fontweight='bold')
plt.xticks(fontproperties=font_bold)
plt.yticks(fontproperties=font_bold)

# Plot for Combined Data Accuracies
plt.subplot(1, 3, 3)
sns.violinplot(data=combined_accuracies, color='Indigo', linewidth=2)
plt.title('Combined Data Accuracy', fontsize=16, fontweight='bold')
plt.xlabel('Experiment', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy', fontsize=14, fontweight='bold')
plt.xticks(fontproperties=font_bold)
plt.yticks(fontproperties=font_bold)

# Adjust layout and display the plot
plt.tight_layout()
plt.show()


# Print results
print("Original Data Accuracies:", original_accuracies)
print("Average Accuracy for Original Data:", original_avg_accuracy)
print("Standard Deviation for Original Data:", original_std)
print("Average Confusion Matrix for Original Data:\n", original_avg_cm)
print("Classification Report for Original Data:\n", original_classification_rep)

print("Synthetic Data Accuracies:", synthetic_accuracies)
print("Average Accuracy for Synthetic Data:", synthetic_avg_accuracy)
print("Standard Deviation for Synthetic Data:", synthetic_std)
print("Average Confusion Matrix for Synthetic Data:\n", synthetic_avg_cm)
print("Classification Report for Synthetic Data:\n", synthetic_classification_rep)

print("Combined Data Accuracies:", combined_accuracies)
print("Average Accuracy for Combined Data:", combined_avg_accuracy)
print("Standard Deviation for Combined Data:", combined_std)
print("Average Confusion Matrix for Combined Data:\n", combined_avg_cm)
print("Classification Report for Combined Data:\n", combined_classification_rep)



# # Signal comparison plot--------------------------------
# def plot_normalized_signal_comparison(original_data, synthetic_data, original_labels, synthetic_labels, class_label):
#     # Find indices for the specified class in both datasets
#     original_indices = np.where(original_labels == class_label)[0]
#     synthetic_indices = np.where(synthetic_labels == class_label)[0]
#     if original_indices.size > 0 and synthetic_indices.size > 0:
#         # Select the first sample from the original and synthetic data for the specified class
#         original_signal = original_data[original_indices[8]]
#         synthetic_signal = synthetic_data[synthetic_indices[8]]
#         # Normalize both signals for a fair comparison
#         original_signal = (original_signal - np.min(original_signal)) / (np.max(original_signal) - np.min(original_signal))
#         synthetic_signal = (synthetic_signal - np.min(synthetic_signal)) / (np.max(synthetic_signal) - np.min(synthetic_signal))
#         # Plotting the signals side by side
#         fig, axs = plt.subplots(1, 2, figsize=(15, 3))
#         axs[0].plot(original_signal, label='Original Signal', marker='o', linestyle='-', color='blue')
#         axs[0].set_title('  Original Signal', fontsize=14, fontweight='bold')
#         axs[0].set_xlabel('Feature Index', fontsize=12, fontweight='bold')
#         axs[0].set_ylabel('  Amplitude', fontsize=12, fontweight='bold')
#         axs[0].legend(fontsize=10)
#         axs[0].grid(True)
#         axs[1].plot(synthetic_signal, label='Synthetic Signal', marker='x', linestyle='--', color='red')
#         axs[1].set_title('  Synthetic Signal', fontsize=14, fontweight='bold')
#         axs[1].set_xlabel('Feature Index', fontsize=12, fontweight='bold')
#         axs[1].legend(fontsize=10)
#         axs[1].grid(True)
#         # Ensure tick parameters for bold and size increase are applied
#         plt.rc('xtick', labelsize=12)  # Set x-tick labels size
#         plt.rc('xtick')  # Set x-tick labels to bold
#         plt.rc('ytick', labelsize=12)  # Set y-tick labels size
#         plt.rc('ytick')  # Set y-tick labels to bold
#         plt.suptitle(f'  Signal Comparison for Class {class_label}', fontsize=16, fontweight='bold')
#         plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the suptitle
#         plt.show()
#     else:
#         print(f"No signals found for class {class_label}")

# # Comparing normalized signals for class label 0
# plot_normalized_signal_comparison(data_normalized, synthetic_data, target, synthetic_labels, class_label=0)



# MSE-------------------------------------------------
from sklearn.preprocessing import MinMaxScaler
def normalize_data(data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(data)
def calculate_average_mse(original_data, synthetic_data):
    # Check and adjust the shape of datasets if necessary
    min_samples = min(original_data.shape[0], synthetic_data.shape[0])
    original_data = original_data[:min_samples]
    synthetic_data = synthetic_data[:min_samples]
    if original_data.shape[1] != synthetic_data.shape[1]:
        raise ValueError("The number of features in original and synthetic data must be the same to calculate MSE.")
    # Normalize both datasets to the range [0, 1]
    original_data_normalized = normalize_data(original_data)
    synthetic_data_normalized = normalize_data(synthetic_data)
    # Calculate MSE
    mse_values = np.mean((original_data_normalized - synthetic_data_normalized) ** 2, axis=1)
    average_mse = np.mean(mse_values)
    return average_mse
average_mse = calculate_average_mse(data_normalized, synthetic_data)
print("Average MSE between normalized original and synthetic datasets:", average_mse)

