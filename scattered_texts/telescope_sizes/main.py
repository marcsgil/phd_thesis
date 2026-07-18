import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path
from pint import UnitRegistry, Quantity

UREG = UnitRegistry()
PLOTS_DIR = Path(__file__).parent / "plots"


def configure_manuscript_plotting():
    """Use the manuscript's default Computer Modern typography in PDF figures."""
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )


def paraxial_proapgation(u0, dx, dy, dz, lbda):
    Ny = u0.shape[-1]
    Nx = u0.shape[-2]
    qx = np.fft.fftfreq(Nx, d=dx) * 2 * np.pi
    qy = np.fft.fftfreq(Ny, d=dy) * 2 * np.pi
    qx, qy = np.meshgrid(qx, qy, sparse=True)

    if isinstance(u0, Quantity):
        ft_u0 = np.fft.fft2(u0.magnitude)
        return u0.units * np.fft.ifft2(
            (ft_u0 * np.exp(-1j * dz * lbda * (qx**2 + qy**2) / 4 / np.pi)).magnitude
        )
    else:
        ft_u0 = np.fft.fft2(u0)
        return np.fft.ifft2(
            ft_u0 * np.exp(-1j * dz * lbda * (qx**2 + qy**2) / 4 / np.pi)
        )


@dataclass
class TelescopeObjective:
    diameter: Quantity
    focal_length: None | Quantity = None
    price: None | int = None
    name: None | str = None
    url: None | str = None

    def __post_init__(self):
        if self.name is None:
            self.name = f"D={self.diameter:~}"

    def __call__(self, u, xs, ys):
        pupil = xs**2 + ys**2 <= (self.diameter / 2) ** 2
        return u * pupil


svbony = TelescopeObjective(
    diameter=102 * UREG.millimeters,
    focal_length=660 * UREG.millimeters,
    url="https://fotonastro.com.br/produto/svbony-sv48p-telescopio-refrator-102mm-f-6-5/",
)

d154 = TelescopeObjective(
    diameter=154 * UREG.millimeters,
    focal_length=750 * UREG.millimeters,
    url="https://pt.aliexpress.com/item/1005009812692231.html?spm=a2g0o.productlist.main.58.44acBf8OBf8OGg&algo_pvid=9c8b0de5-6d1a-4707-b974-2609bf291077&algo_exp_id=9c8b0de5-6d1a-4707-b974-2609bf291077-57&pdp_ext_f=%7B%22order%22%3A%224%22%2C%22eval%22%3A%221%22%2C%22fromPage%22%3A%22search%22%7D&pdp_npi=6%40dis%21BRL%21511.12%21444.68%21%21%2192.70%2180.65%21%40210311c217842520980241543e0e65%2112000050250696315%21sea%21BR%210%21ABX%211%210%21n_tag%3A-29910%3Bd%3A24e6758f%3Bm03_new_user%3A-29895&curPageLogUid=vx0DlxHzj92H&utparam-url=scene%3Asearch%7Cquery_from%3A%7Cx_object_id%3A1005009812692231%7C_p_origin_prod%3A",
)

d52 = TelescopeObjective(
    diameter=52 * UREG.millimeters,
    focal_length=360 * UREG.millimeters,
    url="https://pt.aliexpress.com/item/1005011753771484.html?spm=a2g0n.productlist.0.0.47dauwDnuwDnk2&browser_id=f3ab41db6cea47719a4f6a2b2fb34ed7&aff_trace_key=439446e0f3894a3faf1e61ff373ce7d3-1784216492320-06335-_oFgTQeV&aff_platform=msite&m_page_id=txdqixfo7ikcab8519f6da582915206aed6147e145&gclid=&pdp_ext_f=%7B%22order%22%3A%221%22%2C%22eval%22%3A%221%22%2C%22fromPage%22%3A%22search%22%7D&pdp_npi=6%40dis%21BRL%21113.99%21113.99%21%21%21139.95%21139.95%21%40ac106e7717842509912372604d3c34%2112000056446398864%21sea%21BR%210%21ABX%211%210%21n_tag%3A-29910%3Bd%3A93e4b87f%3Bm03_new_user%3A-29895&algo_pvid=91438eec-4b3a-4e31-84cb-65f4d255e9e4&utparam-url=scene%3Asearch%7Cquery_from%3A%7Cx_object_id%3A1005011753771484%7C_p_origin_prod%3A&search_p4p_id=202607161816313685019013936520011600416_1",
)


def collimated(
    dz, lbda, xs, ys, dx, dy, w0, transmitting_objectives, receiving_objectives
):
    w0, ys, xs = np.meshgrid(w0, ys, xs, sparse=True, indexing="ij")

    u0 = np.exp(-(xs**2 + ys**2) / w0**2)

    P0 = np.sum(abs(u0) ** 2, axis=(-1, -2))

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle(f"z = {dz * 1e-6:.1f} km; lambda = {lbda * 10**6:.0f} nm")
    fig.subplots_adjust(right=0.72)

    for to in transmitting_objectives:
        u = paraxial_proapgation(to(u0, xs, ys), dx, dy, dz, lbda)
        for ro in receiving_objectives:
            P = np.sum(abs(ro(u, xs, ys)) ** 2, axis=(-1, -2))

            ax.plot(2 * w0.flatten(), P / P0, label=f"{to.name} -> {ro.name}")

    ax.set_xlabel("Diameter at UFF (collimated) [mm]")
    ax.set_ylabel("Collection efficiency at CBPF")

    ax.set_yticks(np.arange(0.0, 1.1, 0.1))

    ax.legend(bbox_to_anchor=(1.02, 0.5), loc="center left")
    plt.tight_layout()
    plt.show()


def write_clipping_table(results):
    """Write the numerical maxima as a LaTeX table fragment for the manuscript."""
    rows = [
        r"\begin{tabular}{cc|cc|cc|cc}",
        r"\hline",
        r"\shortstack{$D_t \to D_r$\\(mm)} & \shortstack{$\rho_{98}$\\\vphantom{(mm)}} & \shortstack{$\eta_{\mathrm{G}}^\star$\\\vphantom{(mm)}} & \shortstack{$\eta_{\mathrm{sim}}^\star$\\\vphantom{(mm)}} & \shortstack{$2W_{0,\mathrm{G}}^\star$\\(mm)} & \shortstack{$2W_{0,\mathrm{sim}}^\star$\\(mm)} & \shortstack{$\Delta z_{\mathrm{G}}^\star$\\($\mu$m)} & \shortstack{$\Delta z_{\mathrm{sim}}^\star$\\($\mu$m)} \\",
        r"\hline",
    ]
    for result in results:
        rows.append(
            f"{result['transmitter_diameter_mm']:.0f} $\\to$ "
            f"{result['receiver_diameter_mm']:.0f} & "
            f"{result['rho_98']:.3f} & "
            f"{result['gaussian_efficiency']:.3f} & "
            f"{result['efficiency']:.3f} & "
            f"{result['gaussian_beam_diameter_mm']:.1f} & "
            f"{result['beam_diameter_mm']:.1f} & "
            f"{result['gaussian_shift_um']:.1f} & "
            f"{result['shift_um']:.1f} \\\\"
        )
    rows.extend([r"\hline", r"\end{tabular}"])
    (PLOTS_DIR / "telescope_clipping_results.tex").write_text("\n".join(rows) + "\n")


def not_collimated(
    dz, lbda, xs, ys, dx, dy, w0_values, delta_z_values,
    transmitting_objectives, receiving_objectives,
):
    """Simulate clipped Gaussian links and generate manuscript-ready results."""
    configure_manuscript_plotting()
    PLOTS_DIR.mkdir(exist_ok=True)
    delta_z, w0, ys, xs = np.meshgrid(
        delta_z_values, w0_values, ys, xs, sparse=True, indexing="ij"
    )
    beam_diameters_mm = (2 * w0_values).to(UREG.millimeter).magnitude
    shifts_um = delta_z_values.to(UREG.micrometer).magnitude

    fig, axes = plt.subplots(
        2, 2, figsize=(5.9, 4.7), sharex=True, sharey=True, layout="constrained"
    )
    fig.set_constrained_layout_pads(w_pad=0.12, h_pad=0.06, wspace=0.03, hspace=0.03)
    levels = np.linspace(0.0, 1.0, 21)
    results = []

    for transmitter_index, to in enumerate(transmitting_objectives):
        F = to.focal_length**2 / delta_z
        useful_shift_um = (2 * to.focal_length**2 / dz).to(UREG.micrometer).magnitude
        u0 = np.exp(-(xs**2 + ys**2) * (1 / w0**2 + 1j * np.pi / (lbda * F)))
        P0 = np.sum(abs(u0) ** 2, axis=(-1, -2))
        u = paraxial_proapgation(to(u0, xs, ys), dx, dy, dz, lbda)

        for receiver_index, ro in enumerate(receiving_objectives):
            ax = axes[transmitter_index, receiver_index]
            P = np.sum(abs(ro(u, xs, ys)) ** 2, axis=(-1, -2))
            efficiency = P / P0
            contour = ax.contourf(
                beam_diameters_mm, shifts_um, efficiency, levels=levels, cmap="inferno"
            )
            maximum_index = np.unravel_index(np.argmax(efficiency), efficiency.shape)
            shift_index, beam_index = maximum_index
            maximum_efficiency = float(efficiency[maximum_index])
            gaussian_efficiency = float(
                (
                    1
                    - np.exp(
                        -np.pi
                        * (to.diameter * ro.diameter).to(UREG.meter**2).magnitude
                        / (2 * (lbda * dz).to(UREG.meter**2).magnitude)
                    )
                ) ** 2
            )
            gaussian_shift_um = (to.focal_length**2 / dz).to(UREG.micrometer).magnitude
            gaussian_beam_diameter_mm = 2 * np.sqrt(
                (
                    lbda * dz * to.diameter / (np.pi * ro.diameter)
                ).to(UREG.meter**2).magnitude
            ) * 1e3
            d98 = np.sqrt(
                2
                / np.pi
                * np.log(1 / (1 - np.sqrt(0.98)))
                * (lbda * dz).to(UREG.meter**2).magnitude
            )
            rho_98 = np.sqrt(
                (to.diameter * ro.diameter).to(UREG.meter**2).magnitude
            ) / d98
            ax.plot(
                beam_diameters_mm[beam_index], shifts_um[shift_index],
                marker="*", color="white", markeredgecolor="black", markersize=8,
            )
            ax.axhline(0, color="white", linestyle="--", linewidth=1.0)
            ax.axhline(useful_shift_um, color="white", linestyle="--", linewidth=1.0)
            ax.set_title(
                rf"$D_t={to.diameter.to(UREG.millimeter).magnitude:.0f}\,\mathrm{{mm}}$, "
                rf"$D_r={ro.diameter.to(UREG.millimeter).magnitude:.0f}\,\mathrm{{mm}}$",
                fontsize=9,
            )
            results.append(
                {
                    "transmitter_diameter_mm": to.diameter.to(UREG.millimeter).magnitude,
                    "receiver_diameter_mm": ro.diameter.to(UREG.millimeter).magnitude,
                    "efficiency": maximum_efficiency,
                    "gaussian_efficiency": gaussian_efficiency,
                    "rho_98": rho_98,
                    "beam_diameter_mm": beam_diameters_mm[beam_index],
                    "gaussian_beam_diameter_mm": gaussian_beam_diameter_mm,
                    "shift_um": shifts_um[shift_index],
                    "gaussian_shift_um": gaussian_shift_um,
                }
            )

    for ax in axes[-1, :]:
        ax.set_xlabel(r"Input Gaussian diameter $2W_0$ (mm)")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"Telescope shift $\Delta z$ ($\mu$m)")
        ax.set_yticks([-200, -100, 0, 100, 200])
        ax.set_yticklabels([r"\mbox{-}200", r"\mbox{-}100", r"0", r"100", r"200"])
    colorbar = fig.colorbar(contour, ax=axes, shrink=0.92, pad=0.02)
    colorbar.set_label(r"End-to-end clipping efficiency $\eta_{\mathrm{tot}}$")
    fig.savefig(PLOTS_DIR / "telescope_clipping.pdf")
    plt.close(fig)
    write_clipping_table(results)
    return results


if __name__ == "__main__":
    N = 128
    xs = np.linspace(-500, 500, N) * UREG.millimeter
    ys = np.linspace(-500, 500, N) * UREG.millimeter
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]

    dz = 6800 * UREG.meter
    lbda = 810e-9 * UREG.meter
    w0 = np.linspace(5, 80, 64) * UREG.millimeter
    delta_z = np.linspace(-0.2, 0.2, 64) * UREG.millimeter

    transmitting_objectives = [svbony, d52]
    receiving_objectives = [svbony, d154]

    # collimated(dz, lbda, xs, ys, dx, dy, w0, transmitting_objectives, receiving_objectives)
    not_collimated(
        dz,
        lbda,
        xs,
        ys,
        dx,
        dy,
        w0,
        delta_z,
        transmitting_objectives,
        receiving_objectives,
    )
