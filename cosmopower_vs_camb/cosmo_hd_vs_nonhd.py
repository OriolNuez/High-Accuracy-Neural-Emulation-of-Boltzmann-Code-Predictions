import os
import gdown
from cosmopower import cosmopower_NN
import numpy as np
import matplotlib.pyplot as plt

# --- Google Drive file IDs ---
file_ids = {
    "cmb_tt_low_training_params.npz": "1KhEjYA7EqkJvPf31yif2JvpYPyWSTOw4",
    "cmb_tt_high_training_log_spectra-001.npz": "1ny6SgHsSPK_IqxjCqeyfk2sTvY09WrhC",
    "cmb_tt_high_test_params.npz": "1kUhSq_PEvTR3SKy8Nx4kF6x8LwZHGTBt",
}

# --- Function to download from Google Drive ---
def download_from_drive(file_id, destination):
    if not os.path.exists(destination):
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, destination, quiet=False)

# --- Download required files ---
for filename, file_id in file_ids.items():
    download_from_drive(file_id, filename)

# --- Load training parameters ---
training_parameters = np.load("cmb_tt_low_training_params.npz")
print("Training parameters loaded:", training_parameters.files)

# --- Load training features ---
training_features = np.load("cmb_tt_high_training_log_spectra-001.npz")
print("Training features loaded:", training_features.files)

# --- Load testing parameters ---
testing_params = np.load("cmb_tt_high_test_params.npz")['params']

# --- Define model parameters ---
model_parameters = ['H0', 'tau', 'ombh2', 'ns', 'lnAs', 'omch2']
ell_range = training_features['modes']

# --- Load neural network models ---
cp_nn_low = cosmopower_NN(restore=True, restore_filename='TT_cp_NN_example_low')
predicted_testing_spectra_low = cp_nn_low.ten_to_predictions_np(testing_params)

cp_nn_high = cosmopower_NN(restore=True, restore_filename='TT_cp_NN_example_high')
predicted_testing_spectra_high = cp_nn_high.ten_to_predictions_np(testing_params)

# --- Plot predictions ---
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(50, 10))
for i in range(3):
    pred_high = predicted_testing_spectra_high[i]
    pred_low = predicted_testing_spectra_low[i]
    ax[i].semilogx(ell_range, pred_low, 'blue', label='NN Low Accuracy')
    ax[i].semilogx(ell_range, pred_high, 'red', linestyle='--', label='NN High Accuracy')
    ax[i].set_xlabel('$\ell$', fontsize='x-large')
    ax[i].set_ylabel(r'$\frac{\ell(\ell+1)}{2 \pi} C_\ell$', fontsize='x-large')
    ax[i].legend(fontsize=15)

plt.tight_layout()
plt.savefig('HD_v_nonHD_test.pdf')

print("Plot Saved as 'HD_v_nonHD_test.pdf'")
