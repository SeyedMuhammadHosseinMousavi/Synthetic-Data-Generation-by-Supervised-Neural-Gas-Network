%reset -f
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, LayerNormalization, GaussianNoise
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import time
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.utils import to_categorical

# Start time for the overall process
start_time = time.time()
# Load data
data = pd.read_csv('tr.csv')
features = data.iloc[:, :-1].values
labels = data.iloc[:, -1].values

# Normalize features
scaler = MinMaxScaler()
features_normalized = scaler.fit_transform(features)

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
num_classes = len(np.unique(labels_encoded))

# Split data
train_features, test_features, train_labels, test_labels = train_test_split(features_normalized, labels_encoded, test_size=0.2, random_state=42)

# Parameters
input_dim = train_features.shape[1]
noise_level = 0.12 
epochs = 100
batch_size = 32

# Build the diffusion model
def build_diffusion_model(input_dim, noise_std=0.1):
    input_layer = Input(shape=(input_dim,))
    label_input = Input(shape=(1,))   

    # Add Gaussian noise based on the diffusion step
    noise_input = GaussianNoise(stddev=noise_std)(input_layer)

    # Concatenate features with labels
    concat_layer = Concatenate()([noise_input, label_input])

    # Dense layers for processing
    x = Dense(128, activation='relu')(concat_layer)
    x = LayerNormalization()(x)
    x = Dense(64, activation='relu')(x)
    x = LayerNormalization()(x)
    output_layer = Dense(input_dim, activation='linear')(x)  # Predicting denoised features

    model = Model(inputs=[input_layer, label_input], outputs=output_layer)
    model.compile(optimizer='adam', loss='mse')
    return model

model = build_diffusion_model(input_dim)

# Train the model
def train_model(model, data, labels, epochs, batch_size):
    # Simulate the noise addition and reversal process
    for epoch in range(epochs):
        noise = np.random.normal(0, noise_level, data.shape)
        noisy_data = data + noise
        model.fit([noisy_data, labels], data, batch_size=batch_size, epochs=1, verbose=1)

train_model(model, train_features, train_labels, epochs, batch_size)

# Generate synthetic data
def generate_synthetic_data(model, data, labels, num_samples):
    sampled_features = np.repeat(data, np.ceil(num_samples / len(data)), axis=0)[:num_samples]
    sampled_labels = np.repeat(labels, np.ceil(num_samples / len(labels)), axis=0)[:num_samples]
    initial_noise = np.random.normal(0, noise_level, sampled_features.shape)
    synthetic_data = model.predict([sampled_features + initial_noise, sampled_labels])
    return synthetic_data

# SDG----------------------
synthetic_data = generate_synthetic_data(model, test_features, test_labels, 2000)
print("Synthetic Data Shape:", synthetic_data.shape)


# Save CSV--------------------
import numpy as np
import pandas as pd
# Function to generate synthetic data, ensuring matching feature and label counts
def generate_synthetic_data(model, data, labels, num_samples):
    # Sample indices with replacement to match the desired number of samples
    indices = np.random.choice(len(data), num_samples, replace=True)
    sampled_features = data[indices]
    sampled_labels = labels[indices]

    initial_noise = np.random.normal(0, noise_level, sampled_features.shape)
    synthetic_features = model.predict([sampled_features + initial_noise, sampled_labels])
    
    return synthetic_features, sampled_labels
# Generate the synthetic data and labels
synthetic_features, synthetic_labels = generate_synthetic_data(model, test_features, test_labels, 1000)
# Since 'synthetic_labels' may still be in encoded form, decode them if necessary
if synthetic_labels.ndim > 1 and synthetic_labels.shape[1] > 1:  # Checks if labels are one-hot encoded
    synthetic_labels_decoded = np.argmax(synthetic_labels, axis=1)
    synthetic_labels = label_encoder.inverse_transform(synthetic_labels_decoded)
else:
    synthetic_labels = label_encoder.inverse_transform(synthetic_labels)
# Combine synthetic features and labels
synthetic_data_with_labels = np.hstack((synthetic_features, synthetic_labels.reshape(-1, 1)))
# Define column names for the CSV file
feature_names = ['Feature_' + str(i) for i in range(synthetic_features.shape[1])]
column_names = feature_names + ['Label']
# Create a DataFrame from the synthetic data
synthetic_df = pd.DataFrame(synthetic_data_with_labels, columns=column_names)
# Save the DataFrame to a CSV file
synthetic_df.to_csv('Synthetic_Data-Diff.csv', index=False)


# Print the runtime
end_time = time.time()
runtime_seconds = end_time - start_time
runtime_minutes = runtime_seconds / 60
print(f"\nTotal Runtime: {runtime_seconds:.2f} seconds ({runtime_minutes:.2f} minutes)")


# XGB Classification-------------------------
# Load data
features = data.iloc[:, :-1].values
labels = data.iloc[:, -1].values

# Normalize features
scaler = MinMaxScaler()
features_normalized = scaler.fit_transform(features)

# Encode labels, ensuring they start from 0
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
print("Unique labels after encoding:", np.unique(labels_encoded))  # Check to ensure starting from 0

# Convert labels to categorical (one-hot encoding)
labels_categorical = to_categorical(labels_encoded)

def run_experiments(data, labels, n_runs=5):
    accuracies = []
    all_y_true = []
    all_y_pred = []
    for _ in range(n_runs):
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=None)
        if y_train.ndim > 1 and y_train.shape[1] > 1:
            y_train = np.argmax(y_train, axis=1)
        if y_test.ndim > 1 and y_test.shape[1] > 1:
            y_test = np.argmax(y_test, axis=1)
        model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
    print(classification_report(all_y_true, all_y_pred))
    return np.mean(accuracies), np.std(accuracies)

# Running experiments on original data
original_accuracy, original_std = run_experiments(features_normalized, labels_categorical)
print(f"Original Data - Average Accuracy: {original_accuracy * 100:.2f}%, Std Dev: {original_std:.4f}")

# Prepare synthetic data
if synthetic_labels.ndim == 1 or (synthetic_labels.ndim > 1 and synthetic_labels.shape[1] == 1):
    synthetic_labels_categorical = to_categorical(label_encoder.transform(synthetic_labels))
else:
    synthetic_labels_categorical = synthetic_labels

# Running experiments on synthetic data
synthetic_accuracy, synthetic_std = run_experiments(synthetic_features, synthetic_labels_categorical)
print(f"Synthetic Data - Average Accuracy: {synthetic_accuracy * 100:.2f}%, Std Dev: {synthetic_std:.4f}")

# Combining original and synthetic data for combined experiments
combined_features = np.vstack((features_normalized, synthetic_features))
combined_labels = np.concatenate((labels_categorical, synthetic_labels_categorical))
combined_accuracy, combined_std = run_experiments(combined_features, combined_labels)
print(f"Combined Data - Average Accuracy: {combined_accuracy * 100:.2f}%, Std Dev: {combined_std:.4f}")


# MSE ------------------------
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
# Ensure synthetic data is scaled (if it wasn't already during generation)
if 'synthetic_features' not in locals():
    scaler = MinMaxScaler()
    synthetic_scaled = scaler.fit_transform(synthetic_features) 
else:
    synthetic_scaled = synthetic_features
# Determine the minimum size to use for MSE calculation to ensure dimensions match
min_size = min(features_normalized.shape[0], synthetic_scaled.shape[0])
# Truncate both datasets to the minimum size
original_truncated = features_normalized[:min_size]
synthetic_truncated = synthetic_scaled[:min_size]
# Calculate the MSE between the truncated datasets
mse_value = mean_squared_error(original_truncated, synthetic_truncated)
print(f"Mean Squared Error between Original and Synthetic Data: {mse_value:.4f}")




