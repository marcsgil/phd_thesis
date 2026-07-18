"""Plot the aperture-size prefactor for a target clipping efficiency."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Match the manuscript's default 10 pt Computer Modern typography.  The figure
# is 0.70\textwidth wide for the geometry used in telescope_size.tex.
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    }
)


def kappa(eta: np.ndarray) -> np.ndarray:
    """Dimensionless aperture prefactor for overall efficiency eta."""
    return np.sqrt(2 / np.pi * np.log(1 / (1 - np.sqrt(eta))))


def main() -> None:
    eta = np.linspace(0.0, 0.999, 1_000)
    eta_98 = 0.98
    kappa_98 = kappa(np.array([eta_98]))[0]

    fig, ax = plt.subplots(figsize=(4.13, 2.58))
    ax.plot(eta, kappa(eta), color="C0", linewidth=2)
    ax.plot(eta_98, kappa_98, "o", color="C3")
    ax.annotate(
        rf"$\eta_0=0.98$, $\kappa={kappa_98:.2f}$",
        xy=(eta_98, kappa_98),
        xytext=(-100, 20),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "C3"},
    )
    ax.set(
        xlabel=r"Target overall clipping efficiency $\eta_0$",
        ylabel=r"Dimensionless prefactor $\kappa(\eta_0)$",
        xlim=(0.0, 1.00),
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()

    output = Path(__file__).parent / "plots" / "kappa_vs_efficiency.pdf"
    output.parent.mkdir(exist_ok=True)
    fig.savefig(output)


if __name__ == "__main__":
    main()
