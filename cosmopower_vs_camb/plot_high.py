# First, ensure gdown is installed (uncomment if needed)
# !pip install gdown

from cosmopower import cosmopower_NN
import numpy as np
import matplotlib.pyplot as plt
import time
import gdown

# Google Drive file IDs
files_to_download = {
    "cmb_tt_high_training_params.npz": "12RnCJoyIHr8C8EzChuasH1qK-ESUC3Uo",
    "cmb_tt_high_training_log_spectra-001.npz": "1ny6SgHsSPK_IqxjCqeyfk2sTvY09WrhC",
    "cmb_tt_high_test_params.npz": "1kUhSq_PEvTR3SKy8Nx4kF6x8LwZHGTBt",
    "cmb_tt_high_test_log_spectra.npz": "1cJ6jroa8t8d3o_6b89PwcLUGzzWHkO9j"
}

# Download function
def download_from_drive(file_id, destination):
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, destination, quiet=False)

# Download only necessary files
for filename, file_id in files_to_download.items():
    download_from_drive(file_id, filename)

# Load training parameters (LOW)
training_parameters = np.load("cmb_tt_high_training_params.npz")
print("Training parameters loaded:", training_parameters.files)

# Load training features (HIGH)
training_features = np.load("cmb_tt_high_training_log_spectra-001.npz")
print("Training features loaded:", training_features.files)

# Load testing parameters (HIGH)
testing_params = np.load("cmb_tt_high_test_params.npz")['params']

# Load testing features (HIGH)
CAMB = 10. ** np.load("cmb_tt_high_test_log_spectra.npz")['features']

# Define model parameters
model_parameters = ['H0', 'tau', 'ombh2', 'ns', 'lnAs', 'omch2']
ell_range = training_features['modes']

# Load neural network model
cp_nn = cosmopower_NN(restore=True, restore_filename='TT_cp_NN_example_low')

# Predict spectra using the NN
start_time = time.time()
predicted_testing_spectra = cp_nn.ten_to_predictions_np(testing_params)
execution_time = time.time() - start_time

# Plot results
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(50, 10))
for i in range(3):
    pred = predicted_testing_spectra[i]
    true = CAMB[i]
    ax[i].semilogx(ell_range, true, 'blue', label='Original')
    ax[i].semilogx(ell_range, pred, 'red', linestyle='--', label='NN reconstructed')
    ax[i].set_xlabel('$\ell$', fontsize='x-large')
    ax[i].set_ylabel(r'$\frac{\ell(\ell+1)}{2\pi} C_\ell$', fontsize='x-large')
    ax[i].legend(fontsize=15)

plt.savefig('examples_reconstruction_TT_high-2.pdf')

print(f"Execution time: {execution_time:.3f} seconds")