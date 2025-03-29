import os
import numpy as np
import gdown
from cosmopower import cosmopower_NN

# --- Google Drive file IDs ---
file_ids = {
    "cmb_tt_low_training_params.npz": "1KhEjYA7EqkJvPf31yif2JvpYPyWSTOw4",
    "cmb_tt_high_training_log_spectra-001.npz": "1ny6SgHsSPK_IqxjCqeyfk2sTvY09WrhC",
}

# --- Function to download from Google Drive ---
def download_from_drive(file_id, destination):
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, destination, quiet=False)

# --- Download necessary training files ---
for filename, file_id in file_ids.items():
    download_from_drive(file_id, filename)

# --- Load training data ---
training_params = np.load("cmb_tt_low_training_params.npz")['params']
training_features = np.load("cmb_tt_high_training_log_spectra-001.npz")['features']
modes = np.load("cmb_tt_high_training_log_spectra-001.npz")['modes']

# --- Configure and Train the Neural Network ---
cp_nn = cosmopower_NN(
    parameters=['H0', 'tau', 'ombh2', 'ns', 'lnAs', 'omch2'],
    modes=modes,
    n_hidden=[512, 512, 512],  # Adjust layers as needed
    verbose=True
)

# Train neural network model
cp_nn.train(training_params, training_features, 
            filename_saved_model='TT_cp_NN_trained_high',  # Choose your model name
            validation_split=0.2,  # Adjust validation split if needed
            learning_rate=1e-4,
            batch_size=128,
            epochs=50)

print("Model training completed and saved as 'TT_cp_NN_trained_high'")

