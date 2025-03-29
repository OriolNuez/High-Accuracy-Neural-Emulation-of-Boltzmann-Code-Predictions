import sys
import os
import gdown
sys.path.append("C:/Users/Oriol/Desktop/Cosmo_Power/so_noise_models")  # Adjust path if needed
from cosmopower import cosmopower_NN
import so_noise_models
import matplotlib.pyplot as plt
import numpy as np

# --- Google Drive file IDs ---
file_ids = {
    "cmb_tt_low_training_params.npz": "1KhEjYA7EqkJvPf31yif2JvpYPyWSTOw4",
    "cmb_tt_high_training_log_spectra-001.npz": "1ny6SgHsSPK_IqxjCqeyfk2sTvY09WrhC",
    "cmb_tt_high_test_params.npz": "1kUhSq_PEvTR3SKy8Nx4kF6x8LwZHGTBt",
    "cmb_tt_low_test_log_spectra.npz": "1IIa1RBCHmV8NKyhKINAomYY1KtTL02AD",
    "cmb_tt_high_test_log_spectra.npz": "1cJ6jroa8t8d3o_6b89PwcLUGzzWHkO9j"
}

# --- Function to download from Google Drive ---
def download_from_drive(file_id, destination):
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, destination, quiet=False)

# --- Download necessary files ---
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

# --- Load testing features ---
low_spectra = 10.**np.load("cmb_tt_low_test_log_spectra.npz")['features']
high_spectra = 10.**np.load("cmb_tt_high_test_log_spectra.npz")['features']

# --- Model parameters ---
ell_range = training_features['modes']

# --- Neural Network Predictions ---
cp_nn_low = cosmopower_NN(restore=True, restore_filename='TT_cp_NN_example_low')
predicted_testing_spectra_low = cp_nn_low.ten_to_predictions_np(testing_params)

cp_nn_high = cosmopower_NN(restore=True, restore_filename='TT_cp_NN_example_high')
predicted_testing_spectra_high = cp_nn_high.ten_to_predictions_np(testing_params)

# --- Load noise levels locally ---
noise_levels_load = np.loadtxt("so_noise_models/LAT_comp_sep_noise/v3.1.0/SO_LAT_Nell_T_atmv1_goal_fsky0p4_ILC_CMB.txt")
conv_factor = (2.7255e6)**2

ells = noise_levels_load[:, 0]
SO_TT_noise = noise_levels_load[:, 1] / conv_factor
SO_TT_noise = SO_TT_noise.reshape(1, -1)

# --- Calculate denominator ---
f_sky = 0.4
prefac = np.sqrt(2 / (f_sky * (2 * ells + 1)))
low_spectra_cut = predicted_testing_spectra_low[:9980, 38:7978]
high_spectra_cut = predicted_testing_spectra_high[:9980, 38:7978]

denominator = prefac * (low_spectra_cut + SO_TT_noise)
diff = np.abs((high_spectra_cut - low_spectra_cut) / denominator)

# --- Compute percentiles ---
percentiles = np.zeros((4, diff.shape[1]))
percentiles[0] = np.percentile(diff, 68, axis=0)
percentiles[1] = np.percentile(diff, 95, axis=0)
percentiles[2] = np.percentile(diff, 99, axis=0)
percentiles[3] = np.percentile(diff, 99.9, axis=0)

# --- Plot results ---
plt.figure(figsize=(12, 9))
plt.fill_between(ells, 0, percentiles[2, :], color='lightskyblue', label='99%', alpha=0.8)
plt.fill_between(ells, 0, percentiles[1, :], color='royalblue', label='95%', alpha=0.7)
plt.fill_between(ells, 0, percentiles[0, :], color='darkblue', label='68%', alpha=1)

plt.legend(frameon=False, fontsize=30, loc='upper left')
plt.ylabel(r'$\frac{| C_{\ell, \rm{HD}}^{\rm{TT}} - C_{\ell, \rm{NONHD}}^{\rm{TT}}|} {\sigma_{\ell, \rm{CMB}}^{\rm{TT}}}$', fontsize=50)
plt.xlabel(r'$\ell$', fontsize=50)
plt.ylim(0, 20)

ax = plt.gca()
ax.xaxis.set_major_locator(plt.MaxNLocator(10))
ax.yaxis.set_major_locator(plt.MaxNLocator(5))

plt.setp(ax.get_xticklabels(), fontsize=25)
plt.setp(ax.get_yticklabels(), fontsize=25)
plt.tight_layout()
plt.savefig('./accuracy_emu_TT_wide_HD_v_NONHD.pdf')

print("Plot Saved!")
