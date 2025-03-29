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
    "cmb_tt_high_test_log_spectra.npz": "1cJ6jroa8t8d3o_6b89PwcLUGzzWHkO9j",
    "cmb_tt_high_test_params.npz": "1kUhSq_PEvTR3SKy8Nx4kF6x8LwZHGTBt"
}

# --- Function to download from Google Drive ---
def download_from_drive(file_id, destination):
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, destination, quiet=False)

# --- Download required files ---
for filename, file_id in file_ids.items():
    download_from_drive(file_id, filename)

# --- Load testing features ---
testing_spectra = 10**np.load("cmb_tt_high_test_log_spectra.npz")['features']

# --- Load testing parameters ---
testing_params = np.load("cmb_tt_high_test_params.npz")['params']

# --- Load neural network model ---
cp_nn = cosmopower_NN(restore=True, restore_filename='TT_cp_NN_example_high')
predicted_testing_spectra = cp_nn.ten_to_predictions_np(testing_params)

# --- Load noise levels locally (as requested) ---
noise_levels_load = np.loadtxt("so_noise_models/LAT_comp_sep_noise/v3.1.0/SO_LAT_Nell_T_atmv1_goal_fsky0p4_ILC_CMB.txt")
conv_factor = (2.7255e6)**2

ells = noise_levels_load[:, 0]
SO_TT_noise = noise_levels_load[:, 1] / conv_factor
SO_TT_noise = SO_TT_noise.reshape(1, -1)

f_sky = 0.4
prefac = np.sqrt(2 / (f_sky * (2 * ells + 1)))

# Ensure indexing matches your data range correctly
testing_spectra_cut = testing_spectra[:, 38:7978]
predicted_spectra_cut = predicted_testing_spectra[:, 38:7978]

denominator = prefac * (testing_spectra_cut + SO_TT_noise)
diff = np.abs((predicted_spectra_cut - testing_spectra_cut) / denominator)

# --- Compute percentiles ---
percentiles = np.zeros((4, diff.shape[1]))
percentiles[0] = np.percentile(diff, 68, axis=0)
percentiles[1] = np.percentile(diff, 95, axis=0)
percentiles[2] = np.percentile(diff, 99, axis=0)
percentiles[3] = np.percentile(diff, 99.9, axis=0)

# --- Plot results ---
plt.figure(figsize=(12, 9))
plt.fill_between(ells, 0, percentiles[2, :], color='salmon', label='99%', alpha=0.8)
plt.fill_between(ells, 0, percentiles[1, :], color='red', label='95%', alpha=0.7)
plt.fill_between(ells, 0, percentiles[0, :], color='darkred', label='68%', alpha=1)

plt.ylim(0, 0.2)
plt.legend(frameon=False, fontsize=30, loc='upper left')

plt.ylabel(r'$\frac{| C_{\ell, \rm{emulated}}^{\rm{TT}} - C_{\ell, \rm{true}}^{\rm{TT}}|} {\sigma_{\ell, \rm{CMB}}^{\rm{TT}}}$', fontsize=50)
plt.xlabel(r'$\ell$', fontsize=50)

ax = plt.gca()
ax.xaxis.set_major_locator(plt.MaxNLocator(10))
ax.yaxis.set_major_locator(plt.MaxNLocator(5))

plt.setp(ax.get_xticklabels(), fontsize=25)
plt.setp(ax.get_yticklabels(), fontsize=25)
plt.tight_layout()
plt.savefig('./accuracy_emu_TT_wide_low.pdf')

print("Plot Saved!")
