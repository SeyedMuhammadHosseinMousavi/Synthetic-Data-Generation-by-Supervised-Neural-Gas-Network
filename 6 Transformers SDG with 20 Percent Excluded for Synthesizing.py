%reset -f
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, LayerNormalization, MultiHeadAttention, Dropout, Add, Reshape, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import time
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

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

# Separate 20% of data from each class
train_features = []
train_labels = []
remaining_features = []
remaining_labels = []

unique_labels = np.unique(labels_encoded)
min_samples_per_class = min(len(features_normalized[labels_encoded == label]) for label in unique_labels)

for label in unique_labels:
    label_data = features_normalized[labels_encoded == label]
    label_targets = labels_encoded[labels_encoded == label]
    
    # Ensure equal number of samples per class for balanced dataset
    label_data = label_data[:min_samples_per_class]
    label_targets = label_targets[:min_samples_per_class]
    
    train_x, test_x, train_y, test_y = train_test_split(label_data, label_targets, test_size=0.2, random_state=42)
    train_features.append(train_x)
    train_labels.append(train_y)
    remaining_features.append(test_x)
    remaining_labels.append(test_y)

train_features = np.vstack(train_features)
train_labels = np.concatenate(train_labels)
remaining_features = np.vstack(remaining_features)
remaining_labels = np.concatenate(remaining_labels)

# Parameters
input_dim = train_features.shape[1]
epochs = 10
batch_size = 32
num_heads = 4
dff = 128  # Dimension of the feed-forward network within the Transformer

# Positional Encoding Layer
def positional_encoding(inputs, position_dim):
    position_indices = np.arange(inputs.shape[1])[:, np.newaxis]
    angle_rates = 1 / np.power(10000, (2 * (position_indices // 2)) / np.float32(position_dim))
    angle_rads = position_indices * angle_rates
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return inputs + angle_rads

# Transformer Encoder Model
def build_transformer_model(input_dim, num_heads, dff):
    inputs = Input(shape=(input_dim,))
    label_input = Input(shape=(1,))

    # Reshape the input to be 3D for MultiHeadAttention
    reshaped_inputs = Reshape((input_dim, 1))(inputs)

    # Positional encoding
    encoded_inputs = positional_encoding(reshaped_inputs, input_dim)

    # Multi-head attention and normalization
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=1)(encoded_inputs, encoded_inputs)
    attention_output = Dropout(0.1)(attention_output)
    attention_output = Add()([encoded_inputs, attention_output])
    attention_output = LayerNormalization(epsilon=1e-6)(attention_output)

    # Feed-forward network and normalization
    ffn_output = Dense(dff, activation='relu')(attention_output)
    ffn_output = Dense(1)(ffn_output)
    ffn_output = Dropout(0.1)(ffn_output)
    ffn_output = Add()([attention_output, ffn_output])
    ffn_output = LayerNormalization(epsilon=1e-6)(ffn_output)

    # Flatten the output
    flattened_output = Flatten()(ffn_output)

    # Concatenate with labels and final output layer
    concat_layer = Concatenate()([flattened_output, label_input])
    output = Dense(input_dim, activation='linear')(concat_layer)

    model = Model(inputs=[inputs, label_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

model = build_transformer_model(input_dim, num_heads, dff)

# Train the model on the 80% data
def train_model(model, data, labels, epochs, batch_size):
    for epoch in range(epochs):
        model.fit([data, labels], data, batch_size=batch_size, epochs=1, verbose=1)

train_model(model, remaining_features, remaining_labels, epochs, batch_size)

# Generate synthetic data from the remaining 20%
def generate_synthetic_data(model, data, labels, num_samples_per_class):
    synthetic_data = []
    synthetic_labels = []
    
    unique_labels = np.unique(labels)
    for label in unique_labels:
        label_data = data[labels == label]
        sampled_features = np.repeat(label_data, np.ceil(num_samples_per_class / len(label_data)), axis=0)[:num_samples_per_class]
        generated_data = model.predict([sampled_features, np.full((num_samples_per_class, 1), label)])
        synthetic_data.append(generated_data)
        synthetic_labels.append(np.full(num_samples_per_class, label))
    
    synthetic_data = np.vstack(synthetic_data)
    synthetic_labels = np.concatenate(synthetic_labels)
    
    return synthetic_data, synthetic_labels

# Usage
synthetic_features, synthetic_labels = generate_synthetic_data(model, remaining_features, remaining_labels, 500)  # 500 samples per class

# Ensure synthetic labels are within the original label range
synthetic_labels = np.clip(synthetic_labels, labels_encoded.min(), labels_encoded.max())

# Save synthetic data to CSV
synthetic_labels_decoded = label_encoder.inverse_transform(synthetic_labels.astype(int))
synthetic_data_with_labels = np.hstack((synthetic_features, synthetic_labels_decoded.reshape(-1, 1)))
feature_names = ['Feature_' + str(i) for i in range(synthetic_features.shape[1])]
column_names = feature_names + ['Label']
synthetic_df = pd.DataFrame(synthetic_data_with_labels, columns=column_names)
synthetic_df.to_csv('Synthetic_Data_Transformer.csv', index=False)

# Print the runtime
end_time = time.time()
runtime_seconds = end_time - start_time
runtime_minutes = runtime_seconds / 60
print(f"\nTotal Runtime: {runtime_seconds:.2f} seconds ({runtime_minutes:.2f} minutes)")

# XGB Classification
# XGB Classification
def run_experiments(data, labels, n_runs=5):
    accuracies = []
    all_confusion_matrices = []
    for run in range(n_runs):
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=None)
        model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
        
        # Generate confusion matrix and store
        cm = confusion_matrix(y_test, y_pred)
        all_confusion_matrices.append(cm)
        
        # Store the final run's y_test and y_pred for reporting
        if run == n_runs - 1:
            final_y_test = y_test
            final_y_pred = y_pred
    
    # Calculate the average confusion matrix
    average_confusion_matrix = np.mean(all_confusion_matrices, axis=0)
    average_confusion_matrix = np.round(average_confusion_matrix).astype(int)
    
    # Generate classification report for the last run
    target_names = label_encoder.inverse_transform(np.unique(labels)).astype(str)
    classification_rep = classification_report(final_y_test, final_y_pred, target_names=target_names)

    average_accuracy = np.mean(accuracies)
    std_deviation = np.std(accuracies)
    
    print("Average Confusion Matrix:\n", average_confusion_matrix)
    print("\nClassification Report:\n", classification_rep)
    
    return average_accuracy, std_deviation


# Running experiments on original data
original_accuracy, original_std = run_experiments(train_features, train_labels, n_runs=7)
print(f"Original Data - Average Accuracy: {original_accuracy * 100:.2f}%, Std Dev: {original_std:.4f}")

# Running experiments on synthetic data
synthetic_accuracy, synthetic_std = run_experiments(synthetic_features, synthetic_labels.astype(int), n_runs=7)
print(f"Synthetic Data - Average Accuracy: {synthetic_accuracy * 100:.2f}%, Std Dev: {synthetic_std:.4f}")

# Running experiments on combined data (original + synthetic)
combined_features = np.vstack((train_features, synthetic_features))
combined_labels = np.concatenate((train_labels, synthetic_labels.astype(int)))
combined_accuracy, combined_std = run_experiments(combined_features, combined_labels, n_runs=7)
print(f"Combined Data - Average Accuracy: {combined_accuracy * 100:.2f}%, Std Dev: {combined_std:.4f}")

# MSE Calculation
min_size = min(train_features.shape[0], synthetic_features.shape[0])
original_truncated = train_features[:min_size]
synthetic_truncated = synthetic_features[:min_size]
mse_value = mean_squared_error(original_truncated, synthetic_truncated)
print(f"Mean Squared Error between Original and Synthetic Data: {mse_value:.4f}")
