import sys

from foutstep.stepdft import syss_dft
import matplotlib.pyplot as plt
import numpy as np


# Create some mock data
dt = 0.01
timestamps = -10 + dt * np.arange(2000)

osc_frequency = 2

td_samples = np.tanh(timestamps / 0.5) + 0.5 * np.exp(
    -0.5 * ((timestamps) / 0.5) ** 2
) * np.sin(2 * np.pi * osc_frequency * timestamps)

# Compute the FFT with regularisation!
frequencies, fd_samples = syss_dft(
    td_samples,
    timestamps,
    dt,
)

# Compute without regularisation for comparison...
bare_fd_samples = dt * np.fft.fft(td_samples)

# Plot!
pos_freqs = slice(0, len(timestamps) // 2)

fig, axes = plt.subplots(2, 1, figsize=(16, 10))
fig.suptitle("Simple application of foutstep")
for ax in axes:
    ax.grid(which="both")
axes[0].set(xlabel="Timestamps [s]", ylabel="Time-domain signal")
axes[1].set(
    xlabel="Frequencies [Hz]",
    ylabel=r"$\delta t \cdot $FFT",
    xscale="log",
    yscale="log",
)

axes[0].plot(timestamps, td_samples)

axes[1].plot(
    frequencies[pos_freqs],
    np.abs(bare_fd_samples[pos_freqs]),
    label="No regularisation",
)
axes[1].plot(frequencies[pos_freqs], np.abs(fd_samples[pos_freqs]), label="SySS")
axes[1].axvline(osc_frequency, color="red", label="Oscillation frequency", ls="--")

axes[1].legend()

# Hack to save or show depending on input.
# Don't try this at home.
try:
    plt.savefig(sys.argv[1])
except IndexError:
    plt.show()
    













