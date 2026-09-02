import numpy as np
import matplotlib.pyplot as plt

singles_rate1 = 6e6
singles_rate2 = 6e6
coincidence_rate = 100e3
coincidence_window = 2e-9
background_rate = 10e3


def SNR(transmission, integration_time):
    G2_no_loss = coincidence_rate / singles_rate1 / singles_rate2
    delta_G2_no_loss = G2_no_loss - coincidence_window
    signal_singles2 = transmission * singles_rate2
    measured_singles2 = signal_singles2 + background_rate
    delta_G2 = (
        delta_G2_no_loss * signal_singles2 / measured_singles2
    )
    return (
        np.sqrt(singles_rate1 * measured_singles2 * integration_time)
        * delta_G2
        / np.sqrt(coincidence_window + delta_G2)
    )


transmission = np.logspace(-3, 0, 128)
integration_time = np.linspace(1, 5, 128)
transmission, integration_time = np.meshgrid(
    transmission, integration_time, sparse=True
)

SNRs = SNR(transmission, integration_time)

fig, ax = plt.subplots()

contour = ax.contourf(
    transmission.flatten(),
    integration_time.flatten(),
    SNRs,
    cmap="inferno",
    levels=np.arange(11),
)
colorbar = fig.colorbar(contour)

ax.set_xscale("log")
ax.set_xlabel("Transmissivity")
ax.set_ylabel("Measurement Duration (s)")

plt.show()
