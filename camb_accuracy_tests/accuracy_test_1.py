import matplotlib.pyplot as plt
import camb
import numpy as np
import time

noise_levels_load = np.loadtxt("C:/Users/Oriol/Desktop/Cosmo_Power/training/so_noise_models/LAT_comp_sep_noise/v3.1.0/SO_LAT_Nell_T_atmv1_goal_fsky0p4_ILC_CMB.txt")
conv_factor = (2.7255e6)**2
ells = noise_levels_load[:, 0]
ells = ells[2:7930] #ell multipoles



def cl_power_spectra(lsamp, lacc, acc):
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.32, ombh2=0.022383, omch2=0.12011, mnu=0.06, omk=0, tau=0.0543) #ACDM Params
    pars.InitPower.set_params(As = 2.1e-9, ns = 0.965) #ACDM Params
    pars.set_for_lmax(7930, lens_potential_accuracy=8)
    pars.set_accuracy(lSampleBoost=lsamp, lAccuracyBoost=lacc, AccuracyBoost=acc) #accuracy settings
    results = camb.get_results(pars)
    cmb_spectra = results.get_cmb_power_spectra(pars)
    
    Cl = cmb_spectra['total'][:, 0] #Only TT spectra
    Cl = Cl[2:7930]  #from 2-7930
    Cl_ells = np.column_stack((ells, Cl))
    
    return Cl_ells

def chi_squared(c_low, c_high):
    SO_TT_noise = noise_levels_load[:, 1] / conv_factor
    SO_TT_noise = SO_TT_noise[2:7930]  #from 2-7930
    f_sky = 0.4
    prefac = np.sqrt(2/(f_sky*(2*ells+1)))
    denominator = prefac*(c_high[:, 1]+ SO_TT_noise)
    diff = np.abs((c_low[:, 1] - c_high[:, 1])/(denominator))
    mean = np.mean(diff)
    
    return mean

cl_ref_lacc = cl_power_spectra(1,8,1)
print("lacc_ref completed")
cl_ref_lsamp = cl_power_spectra(49,1,1)
print("lsamp_ref completed")
cl_ref_lsamp1 = cl_power_spectra(50,1,1)
print("lsamp1_ref completed")
cl_ref_acc = cl_power_spectra(1,1,5)
print("acc_ref completed")
#cl_ref_all = cl_power_spectra(6.9,1,8)


acc_array_dif = np.zeros((4,3)) #2D Array (accuracy, mean)
#accuracy spectra
for i in range(1,5):
    start_time = time.time()
    cl = cl_power_spectra(1,1,i)
    acc_array_dif[i-1, 0] = i
    data = chi_squared(cl,cl_ref_acc)
    acc_array_dif[i-1, 1] = data
    end_time = time.time()
    acc_time = end_time-start_time
    acc_array_dif[i-1,2] = acc_time
    print(f"acc completed: {i}, time: {acc_time:.4f} sec, mean: {acc_array_dif[i-1, 1]:.4g}")

#laccuracy spectra
lacc_array_dif = np.zeros((6,3)) #2D Array (accuracy, mean)
for i in range(1,7):
    start_time = time.time()
    cl = cl_power_spectra(1,i,1)
    lacc_array_dif[i-1, 0] = i
    data = chi_squared(cl,cl_ref_lacc)
    lacc_array_dif[i-1,1] = data
    end_time = time.time()
    lacc_time = end_time-start_time
    lacc_array_dif[i-1,2] = lacc_time
    print(f"acc completed: {i}, time: {lacc_time:.4f} sec, mean: {lacc_array_dif[i-1, 1]:.4g}")



#lsample spectra
lsamp_array_dif = np.zeros((49,3)) #2D Array (accuracy, mean)
for i in range(1,50):
    start_time = time.time()
    cl = cl_power_spectra(i,1,1)
    lsamp_array_dif[i-1, 0] = i
    data = chi_squared(cl,cl_ref_lsamp)
    lsamp_array_dif[i-1, 1] = data
    end_time = time.time()
    lsamp_time = end_time-start_time
    lsamp_array_dif[i-1,2] = lsamp_time
    print(f"acc completed: {i}, time: {lsamp_time:.4f} sec, mean: {lsamp_array_dif[i-1, 1]:.4g}")

#lsample1 spectra
lsamp1_array_dif = np.zeros((50,3)) #2D Array (accuracy, mean)
for i in range(1,51):
    start_time = time.time()
    cl = cl_power_spectra(i,1,1)
    lsamp1_array_dif[i-1, 0] = i
    data = chi_squared(cl,cl_ref_lsamp1)
    lsamp1_array_dif[i-1, 1] = data
    end_time = time.time()
    lsamp1_time = end_time-start_time
    lsamp1_array_dif[i-1,2] = lsamp1_time
    print(f"acc completed: {i}, time: {lsamp1_time:.4f} sec, mean: {lsamp1_array_dif[i-1, 1]:.4g}")


def plotter(Accuracy, param, accuracy_array, lim, log_scale = False):
    plt.figure(figsize=(6, 4))
    if param == "Execution Time (seconds)":
        y_data = accuracy_array[:, 2]
    else:
        y_data = accuracy_array[:, 1]
    plt.plot(accuracy_array[:, 0], y_data, linestyle='-')
    if log_scale:  # Apply log scale only if requested
        plt.yscale("log")  # Log scale to enhance small differences
    plt.xlabel(f"{Accuracy}")
    plt.ylabel(f"{param}")
    plt.title(f"Effect of {Accuracy} on {param}")
    plt.xlim(1, lim)  # Keep x-axis focused on relevant range
    plt.savefig(f"{param.replace(' ', '_').replace('(', '').replace(')', '')}_v_{Accuracy}_interval(1-{lim}).pdf")
    plt.show()


plotter("lAccuracyBoost", "Chi-squared Mean Difference", lacc_array_dif, lim=6, log_scale=True)
plotter("lSampleBoost(ref49)", "Chi-squared Mean Difference", lsamp_array_dif, lim=49, log_scale=True)
plotter("lSampleBoost(ref49)", "Chi-squared Mean Difference", lsamp_array_dif, lim=6)
plotter("lSampleBoost(ref50)", "Chi-squared Mean Difference", lsamp1_array_dif, lim=50, log_scale=True)
plotter("AccuracyBoost", "Chi-squared Mean Difference", acc_array_dif, lim=4, log_scale=True)
plotter("lAccuracyBoost", "Execution Time (seconds)", lacc_array_dif, lim=6)
plotter("lSampleBoost(ref49)", "Execution Time (seconds)", lsamp_array_dif, lim=49)
plotter("AccuracyBoost", "Execution Time (seconds)", acc_array_dif, lim=4)
plotter("lSampleBoost(ref50)", "Execution Time (seconds)", lsamp1_array_dif, lim=50)
plotter("lSampleBoost(ref49)", "Execution Time (seconds)", lsamp1_array_dif, lim=50)