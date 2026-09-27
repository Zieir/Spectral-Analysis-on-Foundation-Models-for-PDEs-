"""
EDA pour les datasets générés par generate_physbiasbench_style.py
(burgers.npz / decay.npz).

Lancement :
    python eda_physbiasbench.py physbiasbench_style/burgers.npz
    python eda_physbiasbench.py physbiasbench_style/decay.npz

Dépendances :
    pip install numpy matplotlib
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_and_inspect(npz_path: Path) -> dict:
    """Charge le .npz et affiche un résumé structuré de son contenu."""
    data = np.load(npz_path)
    keys = list(data.keys())

    print(f"\n{'='*70}")
    print(f"Fichier : {npz_path}")
    print(f"Nombre de tableaux (clés) : {len(keys)}")
    print(f"{'='*70}\n")

    # Regroupe les clés par split (train/val/test) pour un affichage clair
    splits = {}
    for k in keys:
        split, family = k.split("_", 1)
        splits.setdefault(split, []).append(family)

    for split in ["train", "val", "test"]:
        if split not in splits:
            continue
        print(f"[{split}] familles présentes : {sorted(splits[split])}")

    print()
    header = f"{'clé':<22}{'shape':<28}{'dtype':<10}{'min':>10}{'max':>10}{'mean':>10}{'has_nan':>9}"
    print(header)
    print("-" * len(header))

    summary = {}
    for k in keys:
        arr = data[k]
        has_nan = bool(np.isnan(arr).any())
        row = {
            "shape": arr.shape,
            "dtype": arr.dtype,
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "has_nan": has_nan,
        }
        summary[k] = row
        print(
            f"{k:<22}{str(arr.shape):<28}{str(arr.dtype):<10}"
            f"{row['min']:>10.3f}{row['max']:>10.3f}{row['mean']:>10.3f}{str(has_nan):>9}"
        )

    return {"data": data, "summary": summary}


def plot_sample_trajectory(data, key: str, channel: int = 0, sample_idx: int = 0,
                            n_frames: int = 8, out_path: Path | None = None):
    """Affiche n_frames d'une trajectoire échantillon pour vérifier visuellement la dynamique."""
    arr = data[key]  # (N, T, C, H, W)
    n_traj, n_t, n_c, h, w = arr.shape
    frame_idx = np.linspace(0, n_t - 1, n_frames).astype(int)

    fig, axes = plt.subplots(1, n_frames, figsize=(2.2 * n_frames, 2.6))
    vmin, vmax = arr[sample_idx, :, channel].min(), arr[sample_idx, :, channel].max()

    for ax, t in zip(axes, frame_idx):
        im = ax.imshow(arr[sample_idx, t, channel], vmin=vmin, vmax=vmax, cmap="RdBu_r")
        ax.set_title(f"t={t}")
        ax.axis("off")

    fig.suptitle(f"{key} — trajectoire #{sample_idx}, canal {channel}")
    fig.colorbar(im, ax=axes, shrink=0.8, label="valeur")

    if out_path:
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        print(f"Figure sauvegardée : {out_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_energy_over_time(data, key: str, out_path: Path | None = None):
    """Trace l'énergie moyenne (norme L2 par frame) pour toutes les trajectoires d'une famille.
    Utile pour voir la dissipation/évolution de la dynamique et repérer des trajectoires
    aberrantes (divergence numérique, NaN, etc.)."""
    arr = data[key]  # (N, T, C, H, W)
    # énergie = moyenne sur canaux/espace de la norme au carré, par trajectoire et par temps
    energy = np.mean(arr ** 2, axis=(2, 3, 4))  # (N, T)

    fig, ax = plt.subplots(figsize=(7, 4))
    for i in range(energy.shape[0]):
        ax.plot(energy[i], alpha=0.4, lw=1)
    ax.plot(energy.mean(axis=0), color="black", lw=2, label="moyenne")
    ax.set_xlabel("frame")
    ax.set_ylabel("énergie moyenne (⟨u²⟩)")
    ax.set_title(f"Évolution de l'énergie — {key}")
    ax.legend()

    if out_path:
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        print(f"Figure sauvegardée : {out_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_value_histogram(data, key: str, out_path: Path | None = None):
    """Histogramme des valeurs (tous canaux/temps/trajectoires confondus)
    pour repérer la plage typique et d'éventuelles anomalies (valeurs extrêmes)."""
    arr = data[key]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(arr.ravel(), bins=100, color="steelblue")
    ax.set_title(f"Distribution des valeurs — {key}")
    ax.set_xlabel("valeur")
    ax.set_ylabel("compte")
    ax.set_yscale("log")

    if out_path:
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        print(f"Figure sauvegardée : {out_path}")
    else:
        plt.show()
    plt.close(fig)


def compare_families_complexity(data, split: str, families: list[str], out_path: Path | None = None):
    """Compare l'énergie moyenne finale entre familles IC (simple -> complex)
    pour vérifier que la complexité croissante se reflète bien dans la dynamique."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for fam in families:
        key = f"{split}_{fam}"
        if key not in data:
            continue
        arr = data[key]
        energy = np.mean(arr ** 2, axis=(2, 3, 4))  # (N, T)
        ax.plot(energy.mean(axis=0), label=fam)
    ax.set_xlabel("frame")
    ax.set_ylabel("énergie moyenne")
    ax.set_title(f"Comparaison des familles IC — split={split}")
    ax.legend()

    if out_path:
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        print(f"Figure sauvegardée : {out_path}")
    else:
        plt.show()
    plt.close(fig)


def main():
    if len(sys.argv) < 2:
        print("Usage: python eda_physbiasbench.py <chemin_vers_fichier.npz>")
        sys.exit(1)

    npz_path = Path(sys.argv[1])
    result = load_and_inspect(npz_path)
    data = result["data"]

    out_dir = npz_path.parent / f"eda_{npz_path.stem}"
    out_dir.mkdir(exist_ok=True)

    # Choisit une clé représentative pour les visualisations détaillées
    all_keys = list(data.keys())
    example_key = next((k for k in all_keys if k.startswith("test_medium")), all_keys[0])

    plot_sample_trajectory(data, example_key, channel=0, sample_idx=0,
                            out_path=out_dir / f"{example_key}_frames.png")
    plot_energy_over_time(data, example_key,
                           out_path=out_dir / f"{example_key}_energy.png")
    plot_value_histogram(data, example_key,
                          out_path=out_dir / f"{example_key}_hist.png")

    families = ["OOD-simple", "simple", "medium", "complex", "OOD-complex"]
    compare_families_complexity(data, "test", families,
                                 out_path=out_dir / "families_energy_comparison.png")

    print(f"\nEDA terminée. Figures dans : {out_dir}/")


if __name__ == "__main__":
    main()
