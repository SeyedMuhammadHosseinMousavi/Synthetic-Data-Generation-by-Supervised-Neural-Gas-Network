
%reset -f
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Lambda, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend as K
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
labels_categorical = to_categorical(labels_encoded)
num_classes = labels_categorical.shape[1]

# VAE Components
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def build_cvae_encoder(latent_dim, features_shape, num_classes):
    # Feature and label inputs
    feature_input = Input(shape=(features_shape,))
    label_input = Input(shape=(num_classes,))
    x = Concatenate()([feature_input, label_input])
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    z_mean = Dense(latent_dim)(x)
    z_log_var = Dense(latent_dim)(x)
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
    return Model([feature_input, label_input], [z_mean, z_log_var, z]), feature_input, label_input

def build_cvae_decoder(latent_dim, features_shape, num_classes):
    latent_input = Input(shape=(latent_dim,))
    label_input = Input(shape=(num_classes,))
    x = Concatenate()([latent_input, label_input])
    x = Dense(512, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    outputs = Dense(features_shape, activation='tanh')(x)
    return Model([latent_input, label_input], outputs)
# start models
latent_dim = 1000
encoder, feature_input, label_input = build_cvae_encoder(latent_dim, features_normalized.shape[1], num_classes)
decoder = build_cvae_decoder(latent_dim, features_normalized.shape[1], num_classes)

# Connect the encoder and decoder
z_mean, z_log_var, z = encoder([feature_input, label_input])
outputs = decoder([z, label_input])
vae = Model([feature_input, label_input], outputs)

# Define VAE loss
reconstruction_loss = binary_crossentropy(feature_input, outputs) * features_normalized.shape[1]
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1) * -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer=Adam(0.0002, 0.5))

# Training the CVAE
def train_cvae(epochs, batch_size=32):
    vae.fit([features_normalized, labels_categorical], features_normalized,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1)  # Enable verbosity for training logs

# Train the CVAE---------------------------------
train_cvae(epochs=100, batch_size=32)

import numpy as np
# Define the number of synthetic samples
num_samples = 2000

# Generate random latent noise
z_sample = np.random.normal(0, 0.1, (num_samples, latent_dim))
# Generate random labels based on the known distribution  
random_label_indices = np.random.randint(0, num_classes, num_samples)
synthetic_labels = to_categorical(random_label_indices, num_classes)
# Decode synthetic features from the latent space and labels
synthetic_samples = decoder.predict([z_sample, synthetic_labels])
# Convert labels back to original encoding if necessary for saving or further processing
label_encoder = LabelEncoder()   
labels_encoded = label_encoder.fit_transform(labels)  # Fit and transform original labels
synthetic_labels_decoded = label_encoder.inverse_transform(random_label_indices)  # Use the same encoder
# IMPORTANT: Encode synthetic labels for use with XGBoost
synthetic_labels_encoded = label_encoder.transform(synthetic_labels_decoded)
# Combine synthetic features and labels for saving or further analysis
synthetic_data_with_labels = np.hstack((synthetic_samples, synthetic_labels_encoded.reshape(-1, 1)))
# Save to CSV (optional step)
synthetic_df = pd.DataFrame(synthetic_data_with_labels, columns=[f'Feature_{i}' for i in range(synthetic_samples.shape[1])] + ['Label'])
synthetic_df.to_csv('Synthetic_Data_CVAE.csv', index=False)


# Print the runtime
end_time = time.time()
runtime_seconds = end_time - start_time
runtime_minutes = runtime_seconds / 60
print(f"\nTotal Runtime: {runtime_seconds:.2f} seconds ({runtime_minutes:.2f} minutes)")


# XGB Classification--------------------------------
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

def run_experiments(data, labels, n_runs=5):
    accuracies = []
    all_y_true = []
    all_y_pred = []
    for _ in range(n_runs):
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=None)
        # Initialize and train the XGBoost model
        model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        model.fit(X_train, y_train)
        # Predict on the test set
        y_pred = model.predict(X_test)
        # Accumulate predictions
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        # Calculate accuracy
        accuracies.append(accuracy_score(y_test, y_pred))
    # Calculate mean accuracy and standard deviation across runs
    mean_accuracy = np.mean(accuracies)
    std_deviation = np.std(accuracies)
    # Print aggregated classification report
    print(classification_report(all_y_true, all_y_pred))
    return mean_accuracy, std_deviation

# Calculate and display performance metrics for original data
original_mean_acc, original_std_dev = run_experiments(features_normalized, labels_encoded)
print(f"Original Data - Average Accuracy: {original_mean_acc * 100:.2f}%, Std Dev: {original_std_dev:.2f}")

# Calculate and display performance metrics for synthetic data
synthetic_mean_acc, synthetic_std_dev = run_experiments(synthetic_samples, synthetic_labels_encoded)
print(f"Synthetic Data - Average Accuracy: {synthetic_mean_acc * 100:.2f}%, Std Dev: {synthetic_std_dev:.2f}")

# Combine the original and synthetic data for further evaluation
combined_features = np.vstack((features_normalized, synthetic_samples))
combined_labels = np.concatenate((labels_encoded, synthetic_labels_encoded))

# Calculate and display performance metrics for combined data
combined_mean_acc, combined_std_dev = run_experiments(combined_features, combined_labels)
print(f"Combined Data - Average Accuracy: {combined_mean_acc * 100:.2f}%, Std Dev: {combined_std_dev:.2f}")



# MSE ------------------
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
# Normalize both datasets to ensure they are within the range [0, 1] if not already done so.
scaler = MinMaxScaler()
# Normalize original features
features_normalized_scaled = scaler.fit_transform(features_normalized)
# Apply the same transformation to synthetic samples  
synthetic_samples_scaled = scaler.transform(synthetic_samples)
# Determine the minimum size to use for MSE calculation
min_size = min(features_normalized_scaled.shape[0], synthetic_samples_scaled.shape[0])
# Truncate both datasets to the minimum size
original_truncated = features_normalized_scaled[:min_size]
synthetic_truncated = synthetic_samples_scaled[:min_size]
# Calculate the MSE between the truncated datasets
mse_value = mean_squared_error(original_truncated, synthetic_truncated)
print("Mean Squared Error between Original and Synthetic Data:", mse_value)


