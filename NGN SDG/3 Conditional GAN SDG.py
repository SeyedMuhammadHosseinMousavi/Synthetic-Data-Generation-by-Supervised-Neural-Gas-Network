
%reset -f
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Input, Embedding, Flatten, Concatenate, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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

# Generator
def build_generator(latent_dim, num_classes, output_dim):
    # Label input
    label_input = Input(shape=(num_classes,))
    # Noise input
    noise_input = Input(shape=(latent_dim,))
    # Concatenate label embedding and noise
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
    # Feature input
    feature_input = Input(shape=(input_dim,))
    # Label input
    label_input = Input(shape=(num_classes,))
    # Concatenate feature and label
    merged_input = Concatenate()([feature_input, label_input])

    x = Dense(512)(merged_input)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(256)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model([feature_input, label_input], x)
    model.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])
    return model

# Model settings
latent_dim = 100
generator = build_generator(latent_dim, num_classes, features_normalized.shape[1])
discriminator = build_discriminator(features_normalized.shape[1], num_classes)
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

# cGAN
discriminator.trainable = False
noise_input = Input(shape=(latent_dim,))
label_input = Input(shape=(num_classes,))
generated_features = generator([noise_input, label_input])
validity = discriminator([generated_features, label_input])
combined = Model([noise_input, label_input], validity)
combined.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

# Training function
def train_cgan(epochs, batch_size):
    for epoch in range(epochs):
        # Training discriminator
        idx = np.random.randint(0, features_normalized.shape[0], batch_size)
        real_features = features_normalized[idx]
        real_labels = labels_categorical[idx]

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        synthetic_features = generator.predict([noise, real_labels])
        synthetic_labels = real_labels

        d_loss_real = discriminator.train_on_batch([real_features, real_labels], np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch([synthetic_features, synthetic_labels], np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Training generator
        g_loss = combined.train_on_batch([noise, real_labels], np.ones((batch_size, 1)))

        if epoch % 100 == 0:
            print(f"Epoch {epoch} [D loss: {d_loss}, acc.: {100*d_loss[1]}] [G loss: {g_loss}]")


# Start training--------------------------------
train_cgan(epochs=100, batch_size=32)

# Generate synthetic data for testing
test_noise = np.random.normal(0, 0.2, (2000, latent_dim))
test_labels = to_categorical(np.random.randint(0, num_classes, 2000), num_classes)
test_synthetic_features = generator.predict([test_noise, test_labels])


# Save CSV------------------------------------------
import pandas as pd
# Decode one-hot encoded labels to original format if necessary
synthetic_labels_decoded = np.argmax(test_labels, axis=1)
# Convert labels back to original label encoding
synthetic_labels = label_encoder.inverse_transform(synthetic_labels_decoded)
# Combine synthetic features and labels
synthetic_data_with_labels = np.hstack((test_synthetic_features, synthetic_labels.reshape(-1, 1)))
# Define column names for the CSV file
feature_names = ['Feature_' + str(i) for i in range(test_synthetic_features.shape[1])]
column_names = feature_names + ['Label']
# Create a DataFrame
synthetic_df = pd.DataFrame(synthetic_data_with_labels, columns=column_names)
# Save the DataFrame to a CSV file
synthetic_df.to_csv('Synthetic_Data_GAN.csv', index=False)

# Print the runtime
end_time = time.time()
runtime_seconds = end_time - start_time
runtime_minutes = runtime_seconds / 60
print(f"\nTotal Runtime: {runtime_seconds:.2f} seconds ({runtime_minutes:.2f} minutes)")


# XGB Classification------------------------------------------
import numpy as np
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
        # Train the model
        model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        model.fit(X_train, np.argmax(y_train, axis=1))
        # Predict and evaluate
        y_pred = model.predict(X_test)
        all_y_true.extend(np.argmax(y_test, axis=1))
        all_y_pred.extend(y_pred)
        accuracy = accuracy_score(np.argmax(y_test, axis=1), y_pred)
        accuracies.append(accuracy)
    # Generate classification report after all runs
    print(classification_report(all_y_true, all_y_pred))
    # Return mean accuracy and standard deviation
    return np.mean(accuracies), np.std(accuracies)


# Running experiments on original data
original_accuracy, original_std = run_experiments(features_normalized, labels_categorical)
print(f"Original Data - Average Accuracy: {original_accuracy * 100:.2f}%, Std Dev: {original_std * 100:.2f}%")

# Running experiments on synthetic data
synthetic_accuracy, synthetic_std = run_experiments(test_synthetic_features, test_labels)
print(f"Synthetic Data - Average Accuracy: {synthetic_accuracy * 100:.2f}%, Std Dev: {synthetic_std * 100:.2f}%")

# Combining original and synthetic data for combined experiments
combined_features = np.vstack((features_normalized, test_synthetic_features))
combined_labels = np.concatenate((labels_categorical, test_labels))
combined_accuracy, combined_std = run_experiments(combined_features, combined_labels)
print(f"Combined Data - Average Accuracy: {combined_accuracy * 100:.2f}%, Std Dev: {combined_std * 100:.2f}%")



# MSE----------------------------------------
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# If 'test_synthetic_features' wasn't scaled during generation, scale it:
synthetic_scaled = scaler.fit_transform(test_synthetic_features)
# Determine the minimum size to use for MSE calculation
min_size = min(features_normalized.shape[0], synthetic_scaled.shape[0])
# Truncate both datasets to the minimum size
original_truncated = features_normalized[:min_size]
synthetic_truncated = synthetic_scaled[:min_size]
# Calculate the MSE between the truncated datasets
mse_value = mean_squared_error(original_truncated, synthetic_truncated)
print(f"Mean Squared Error between Original and Synthetic Data: {mse_value:.4f}")



