from foutstep.stepdft import syss_dft, fit_sigmoid, sigmoid_fd
import matplotlib.pyplot as plt
import numpy as np

dt = 0.01
timestamps = np.arange(-15, 15, dt)
td_samples = np.tanh(timestamps) + 0.5 * np.exp(
    -0.5 * ((timestamps + 1.5) / 2.0) ** 2
) * np.sin(2.0 * timestamps)

sigmoid_loc = 0.0
sigmoid_width = 0.1

fd_samples, frequencies = syss_dft(
    td_samples,
    timestamps,
    dt,
    sigmoid_loc,
    sigmoid_width,
)


regularizator = fit_sigmoid(
    td_samples,
    timestamps,
    sigmoid_loc,
    sigmoid_width,
)

residual = td_samples - regularizator


fig, axes = plt.subplots(1, 2)

axes[0].plot(timestamps, td_samples)
axes[0].plot(timestamps, regularizator, "--")
axes[0].plot(timestamps, residual, "-.")

axes[1].loglog(frequencies[frequencies > 0], np.abs(fd_samples[frequencies > 0]))
axes[1].loglog(frequencies[frequencies>0], np.abs(sigmoid_fd(frequencies[frequencies>0], sigmoid_loc, sigmoid_width)), "--")
axes[1].loglog(frequencies[frequencies>0], np.abs(dt*np.fft.fft(residual)[frequencies>0]), "-.")
axes[1].set_ylim(ymax=10, ymin=1e-11)

plt.show()
