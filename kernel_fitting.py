# kernel_fitting.py

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from tqdm import tqdm
from scipy.signal import convolve


def euler_integration(X, Y, interval):
    """Calculate the integral of two signals X*Y over a given interval."""
    mask = ~np.isnan(X) & ~np.isnan(Y)
    product = X[mask] * Y[mask]
    scale = (interval[-1] - interval[0]) / (len(interval) - 1)
    return scale * np.sum(product)


def conv(X, Y, interval):
    """Calculate the convolution of two signals X and Y over a given interval."""
    return convolve(X, Y, mode='same') * (interval[1] - interval[0])


def exp_kernel_func(t, *params):
    """Exponentials combinations with two parameters each: amplitude, decay constant."""
    exponential_combination = np.zeros(len(t))
    for p_idx in range(0, len(params), 3):
        exponential_combination += params[p_idx] * np.exp(-params[p_idx+1] * t)
    return exponential_combination * np.heaviside(t, 1.0)


def fit_exp_kernel(t, x, y):
    """Fits an exponential kernel to data."""
    def convolved_model(t, *params):
        resolution = len(t)
        kernel = exp_kernel_func(t, *params)
        convolved_signal = np.zeros(resolution)
        for t_idx in range(1, resolution):
            convolved_signal[t_idx] = euler_integration(x[:t_idx], kernel[(t_idx-1)::-1], t[:t_idx])
        return convolved_signal

    initial_guess = np.random.rand(3)
    params_opt, params_cov = curve_fit(convolved_model, t, y, p0=initial_guess)
    return params_opt, params_cov


def nonlinear_equation_original(t, X, N, A):
    """Nonlinear ODE system without external stimulus."""
    dX = np.zeros_like(X)
    for i in range(N):
        sum_term = np.sum(A[i] * (1 - X[i]) * X) - A[i, i] * (1 - X[i]) * X[i]
        dX[i] = sum_term - X[i]
    return dX


def nonlinear_equation_stimuli(t, X, N, A, stim_neurons, stimuli):
    """Nonlinear ODE system with external stimulus."""
    dX = np.zeros_like(X)
    for i in range(N):
        sum_term = np.sum(A[i] * (1 - X[i]) * X) - A[i, i] * (1 - X[i]) * X[i]

        stim = 0.0
        if (i in stim_neurons) and (2 < t < 8):
            stim = stimuli

        dX[i] = sum_term - X[i] + stim
    return dX


def fit_kernels(N_trials=8, resolution=500):
    """Fits kernels to simulated network data."""
    t_eval = np.linspace(0, 20, resolution)
    DeltaX = np.random.rand(N_trials, resolution)  # Simulated data

    fitted_lin_params_list = []
    for trial_idx in range(N_trials):
        fitted_lin_params, _ = fit_exp_kernel(t_eval, DeltaX[trial_idx, :], DeltaX[trial_idx, :])
        fitted_lin_params_list.append(fitted_lin_params)

    return fitted_lin_params_list, t_eval


def plot_kernels(fitted_lin_params_list, t_eval):
    """Plots the estimated kernels."""
    plt.figure(figsize=(12, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(fitted_lin_params_list)))

    for trial_idx, params in enumerate(fitted_lin_params_list):
        kernel = exp_kernel_func(t_eval, *params)
        plt.plot(t_eval, kernel, label=f'Kernel #{trial_idx}', color=colors[trial_idx])

    plt.title("Estimated Kernels")
    plt.xlabel("Time")
    plt.ylabel("Kernel Response")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    fitted_lin_params_list, t_eval = fit_kernels()
    plot_kernels(fitted_lin_params_list, t_eval)
