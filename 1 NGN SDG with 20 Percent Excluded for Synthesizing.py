%reset -f
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import time
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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
            remainder = n_samples % n_units  # Calculate the remainder to ensure total samples
    
            for i, unit in enumerate(self.units[label]):
                current_samples = samples_per_unit + (1 if i < remainder else 0)  # Distribute remainder
                for _ in range(current_samples):
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

# Adjust labels to start from 0
target -= np.min(target)

# Separate 20% of data from each class
unique_labels = np.unique(target)
test_data = []
test_labels = []
train_data = []
train_labels = []

for label in unique_labels:
    label_data = data_normalized[target == label]
    label_target = target[target == label]
    train_x, test_x, train_y, test_y = train_test_split(label_data, label_target, test_size=0.2, random_state=42)
    test_data.append(test_x)
    test_labels.append(test_y)
    train_data.append(train_x)
    train_labels.append(train_y)

test_data = np.vstack(test_data)
test_labels = np.concatenate(test_labels)
train_data = np.vstack(train_data)
train_labels = np.concatenate(train_labels)

# Train Supervised Neural Gas on the 20% data
sng = SupervisedNeuralGas(n_units_per_class=9, max_iter=100)
sng.train(test_data, test_labels)

# Generate synthetic data
synthetic_data, synthetic_labels = sng.generate_synthetic_data(n_samples=500, noise_level=0.01)

# Adjust synthetic labels to start from 0
synthetic_labels -= np.min(synthetic_labels)

# Print the runtime
end_time = time.time()
runtime_seconds = end_time - start_time
runtime_minutes = runtime_seconds / 60
print(f"\nTotal Runtime: {runtime_seconds:.2f} seconds ({runtime_minutes:.2f} minutes)")

# Save synthetic dataset as CSV
# Combine synthetic data and labels
synthetic_data_with_labels = np.hstack((synthetic_data, synthetic_labels.reshape(-1, 1)))
# Create a DataFrame
columns = [f'Feature_{i}' for i in range(synthetic_data.shape[1])] + ['Label']
synthetic_df = pd.DataFrame(synthetic_data_with_labels, columns=columns)
# Save the DataFrame to a CSV file
synthetic_df.to_csv('Synthetic_data_NGN.csv', index=False)


# Classification function
def run_experiments(data, labels, n_runs=7):
    accuracies = []
    all_confusion_matrices = []
    
    for run in range(n_runs):
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=None)
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
        
        # Calculate confusion matrix for this run
        cm = confusion_matrix(y_test, y_pred)
        all_confusion_matrices.append(cm)
        
        # Store last run's y_test and y_pred for final reporting
        if run == n_runs - 1:
            final_y_test = y_test
            final_y_pred = y_pred

    # Average confusion matrix
    average_confusion_matrix = np.mean(all_confusion_matrices, axis=0)
    average_confusion_matrix = np.round(average_confusion_matrix).astype(int)  # Convert to integer

    target_names = [str(int(label)) for label in np.unique(labels)]

    # Generate classification report for the last run
    classification_rep = classification_report(final_y_test, final_y_pred, target_names=target_names)
    
    average_accuracy = np.mean(accuracies)
    std_deviation = np.std(accuracies)
    
    return accuracies, average_accuracy, std_deviation, average_confusion_matrix, classification_rep

# Baseline classification on original data
original_accuracies, original_avg_accuracy, original_std, original_avg_cm, original_classification_rep = run_experiments(data_normalized, target)

# Classification on synthetic data
synthetic_accuracies, synthetic_avg_accuracy, synthetic_std, synthetic_avg_cm, synthetic_classification_rep = run_experiments(synthetic_data, synthetic_labels)

# Classification on combination of synthetic data and remaining original data
combined_data = np.vstack((train_data, synthetic_data))
combined_labels = np.concatenate((train_labels, synthetic_labels))
combined_accuracies, combined_avg_accuracy, combined_std, combined_avg_cm, combined_classification_rep = run_experiments(combined_data, combined_labels)

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

# Print the runtime
end_time = time.time()
runtime_seconds = end_time - start_time
runtime_minutes = runtime_seconds / 60
print(f"\nTotal Runtime: {runtime_seconds:.2f} seconds ({runtime_minutes:.2f} minutes)")

# MSE Calculation
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

# Calculate and print the average MSE
average_mse = calculate_average_mse(data_normalized, synthetic_data)
print("Average MSE between normalized original and synthetic datasets:", average_mse)
