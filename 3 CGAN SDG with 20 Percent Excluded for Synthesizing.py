%reset -f

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import time

# Start time for the overall process
start_time = time.time()

# Load data
data = pd.read_csv('tr.csv')
features = data.iloc[:, :-1].values
labels = data.iloc[:, -1].values

# Normalize features
scaler = MinMaxScaler()
features_normalized = scaler.fit_transform(features)

# Prepare labels for training
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
num_classes = len(np.unique(labels_encoded))
labels_categorical = to_categorical(labels_encoded, num_classes)

# Split data into 20% for cGAN training and 80% for later use
cgan_train_data = []
cgan_train_labels = []
remaining_data = []
remaining_labels = []

unique_labels = np.unique(labels_encoded)
for label in unique_labels:
    label_data = features_normalized[labels_encoded == label]
    label_target = labels_categorical[labels_encoded == label]
    cgan_x, remaining_x, cgan_y, remaining_y = train_test_split(label_data, label_target, test_size=0.8, random_state=42)
    cgan_train_data.append(cgan_x)
    cgan_train_labels.append(cgan_y)
    remaining_data.append(remaining_x)
    remaining_labels.append(remaining_y)

cgan_train_data = np.vstack(cgan_train_data)
cgan_train_labels = np.vstack(cgan_train_labels)
remaining_data = np.vstack(remaining_data)
remaining_labels = np.vstack(remaining_labels)

# Generator
def build_generator(latent_dim, num_classes, output_dim):
    label_input = Input(shape=(num_classes,))
    noise_input = Input(shape=(latent_dim,))
    merged_input = Concatenate()([noise_input, label_input])

    x = Dense(128)(merged_input)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Dense(256)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Dense(output_dim, activation='tanh')(x)
    model = Model([noise_input, label_input], x)
    return model

# Discriminator
def build_discriminator(input_dim, num_classes):
    feature_input = Input(shape=(input_dim,))
    label_input = Input(shape=(num_classes,))
    merged_input = Concatenate()([feature_input, label_input])

    x = Dense(512)(merged_input)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(256)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model([feature_input, label_input], x)
    model.compile(loss='binary_crossentropy', optimizer=Adam(0.002, 0.5), metrics=['accuracy'])
    return model

# Model settings
latent_dim = 100
generator = build_generator(latent_dim, num_classes, cgan_train_data.shape[1])
discriminator = build_discriminator(cgan_train_data.shape[1], num_classes)
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.002, 0.5), metrics=['accuracy'])

# cGAN
discriminator.trainable = False
noise_input = Input(shape=(latent_dim,))
label_input = Input(shape=(num_classes,))
generated_features = generator([noise_input, label_input])
validity = discriminator([generated_features, label_input])
combined = Model([noise_input, label_input], validity)
combined.compile(loss='binary_crossentropy', optimizer=Adam(0.002, 0.5))

# Training function
def train_cgan(epochs, batch_size):
    for epoch in range(epochs):
        # Training discriminator
        idx = np.random.randint(0, cgan_train_data.shape[0], batch_size)
        real_features = cgan_train_data[idx]
        real_labels = cgan_train_labels[idx]

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        synthetic_features = generator.predict([noise, real_labels])
        synthetic_labels = real_labels

        d_loss_real = discriminator.train_on_batch([real_features, real_labels], np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch([synthetic_features, synthetic_labels], np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Training generator
        g_loss = combined.train_on_batch([noise, real_labels], np.ones((batch_size, 1)))

        if epoch % 100 == 0:
            print(f"Epoch {epoch} [D loss: {d_loss}, acc.: {100*d_loss[1]:.2f}] [G loss: {g_loss:.4f}]")

# Start training
train_cgan(epochs=2000, batch_size=32)

# Generate synthetic data
num_samples = 2000
noise = np.random.normal(0, 1, (num_samples, latent_dim))
synthetic_labels = to_categorical(np.random.randint(0, num_classes, num_samples), num_classes)
synthetic_samples = generator.predict([noise, synthetic_labels])

# Save synthetic data
synthetic_labels_decoded = np.argmax(synthetic_labels, axis=1)
synthetic_labels = label_encoder.inverse_transform(synthetic_labels_decoded)
synthetic_data_with_labels = np.hstack((synthetic_samples, synthetic_labels.reshape(-1, 1)))
synthetic_df = pd.DataFrame(synthetic_data_with_labels, columns=[f'Feature_{i}' for i in range(synthetic_samples.shape[1])] + ['Label'])
synthetic_df.to_csv('Synthetic_Data_GAN.csv', index=False)

# Print the runtime
end_time = time.time()
runtime_seconds = end_time - start_time
runtime_minutes = runtime_seconds / 60
print(f"\nTotal Runtime: {runtime_seconds:.2f} seconds ({runtime_minutes:.2f} minutes)")

# XGB Classification
def run_experiments(data, labels, n_runs=5):
    accuracies = []
    all_confusion_matrices = []
    for run in range(n_runs):
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=None)
        model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        model.fit(X_train, np.argmax(y_train, axis=1))
        y_pred = model.predict(X_test)
        accuracies.append(accuracy_score(np.argmax(y_test, axis=1), y_pred))
        
        # Store confusion matrices for each run
        cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred, labels=np.arange(num_classes))
        all_confusion_matrices.append(cm)
        
        # Store last run's y_test and y_pred for final reporting
        if run == n_runs - 1:
            final_y_test = np.argmax(y_test, axis=1)
            final_y_pred = y_pred

    mean_accuracy = np.mean(accuracies)
    std_deviation = np.std(accuracies)
    
    # Average confusion matrix
    avg_confusion_matrix = np.mean(all_confusion_matrices, axis=0)
    
    # Print the confusion matrix
    print("Average Confusion Matrix:\n", avg_confusion_matrix)

    # Ensure target names are strings
    target_names = label_encoder.classes_.astype(str)

    # Classification report for the last run
    print("\nClassification Report:\n", classification_report(final_y_test, final_y_pred, target_names=target_names))
    
    return mean_accuracy, std_deviation

# Performance on original data
original_accuracy, original_std = run_experiments(features_normalized, labels_categorical)
print(f"Original Data - Average Accuracy: {original_accuracy * 100:.2f}%, Std Dev: {original_std * 100:.2f}%")

# Performance on synthetic data
synthetic_accuracy, synthetic_std = run_experiments(synthetic_samples, to_categorical(synthetic_labels_decoded, num_classes))
print(f"Synthetic Data - Average Accuracy: {synthetic_accuracy * 100:.2f}%, Std Dev: {synthetic_std * 100:.2f}%")

# Performance on combined data (80% remaining original + synthetic)
combined_features = np.vstack((remaining_data, synthetic_samples))
combined_labels = np.concatenate((remaining_labels, to_categorical(synthetic_labels_decoded, num_classes)))

combined_accuracy, combined_std = run_experiments(combined_features, combined_labels)
print(f"Combined Data - Average Accuracy: {combined_accuracy * 100:.2f}%, Std Dev: {combined_std * 100:.2f}%")

# MSE Calculation
def calculate_average_mse(original_data, synthetic_data):
    min_samples = min(original_data.shape[0], synthetic_data.shape[0])
    original_data = original_data[:min_samples]
    synthetic_data = synthetic_data[:min_samples]
    if original_data.shape[1] != synthetic_data.shape[1]:
        raise ValueError("The number of features in original and synthetic data must be the same to calculate MSE.")
    
    original_data_normalized = scaler.fit_transform(original_data)
    synthetic_data_normalized = scaler.transform(synthetic_data)
    
    mse_values = np.mean((original_data_normalized - synthetic_data_normalized) ** 2, axis=1)
    average_mse = np.mean(mse_values)
    return average_mse

average_mse = calculate_average_mse(features_normalized, synthetic_samples)
print(f"Mean Squared Error between Original and Synthetic Data: {average_mse:.4f}")
