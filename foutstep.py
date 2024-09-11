import numpy as np


def tanh(t, w, t0):
    return 0.5 * (np.tanh((t-t0)/w) + 1.)

def tanh_fft(f, w, t0):  # FourierParameters -> {0,-2Pi} in Mathematica
    return 0.5 * (-1j * np.pi * w) * csch(np.pi**2 * f * w) * np.exp(-2j * np.pi * f * t0)

def csch(x):
    return 1.0 / np.sinh(x)

def fft_sigmoid_regularization(h, t, dt, sigmas_away_from_boundaries = 5., tc_frac = 0.5, return_sigmoid_and_residual=False):

    t0 = t[0]
    tf = t[-1]

    h0 = h[0]
    hf = h[-1]
    
    # fitting sigmoid
    #fit_a = sigmas_away_from_boundaries / (tf - t0) 
    fit_w = 10 #1 / fit_a
    fit_tshift = 0 #tc_frac * (tf + t0) 

    tanh_t = tanh(t, fit_w, fit_tshift)
    tanh_t0 = tanh_t[0]
    tanh_tf = tanh_t[-1]

    fit_denominator = tanh_t0 - tanh_tf
    fit_amplitude = (h0 - hf) / fit_denominator
    fit_offset = (tanh_t0 * hf - tanh_tf * h0) / fit_denominator

    fitted_tanh = fit_amplitude * tanh_t + fit_offset

    # residual
    residual = h - fitted_tanh 

    # Fourier
    f = np.fft.rfftfreq(len(h), dt)

    fft_fitted_tanh = fit_amplitude * tanh_fft(f, fit_w, fit_tshift - t0)
    fft_my_residual = np.fft.rfft(residual) * dt
    fft = fft_fitted_tanh + fft_my_residual

    if return_sigmoid_and_residual:
        return f, fft, fitted_tanh, residual, fft_fitted_tanh, fft_my_residual
    else:
        return f, fft

