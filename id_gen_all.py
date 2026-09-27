"""
Reproduit la génération de données de l'Annexe A du preprint
"Do Physics Foundation Models Learn Generalizable Physics?" (arXiv:2605.29283)
pour les 8 dynamiques : Gray-Scott (A.1), Wave (A.2), Fisher-KPP (A.3),
Burgers (A.4), Swift-Hohenberg (A.5), Decay (A.6), Kolmogorov Flow (A.7),
Kuramoto-Sivashinsky (A.8).

IMPORTANT — à lire avant de lancer :
-------------------------------------
Les steppers Burgers et NavierStokesVorticity (Decay) utilisent une API
confirmée par un run réel. Pour les 6 autres dynamiques (Gray-Scott, Wave,
Fisher-KPP, Swift-Hohenberg, Kolmogorov Flow, Kuramoto-Sivashinsky), les noms
de paramètres ci-dessous sont ma meilleure estimation basée sur la
cohérence de l'API Exponax (cf. table "Built-in Equations" de la doc
officielle : https://fkoehler.site/exponax/), mais n'ont pas été vérifiés
par exécution. Lance d'abord `python generate_all_8.py --check` : ça
affiche la signature réelle de chaque stepper installé chez toi, pour que
tu corriges en 30s les noms d'arguments avant la génération complète.

Installation :
    pip install exponax

Lancement :
    python generate_all_8.py --check          # vérifie les signatures des steppers
    python generate_all_8.py                  # génère les 8 datasets
    python generate_all_8.py --only gray_scott wave   # génère un sous-ensemble
"""

import argparse
import inspect
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import exponax as ex

OUT_DIR = Path("physbiasbench_style")
OUT_DIR.mkdir(exist_ok=True)

NUM_SPATIAL_DIMS = 2
NUM_POINTS = 64
NUM_FRAMES = 100  # frames stockées par trajectoire (CI incluse), s=0..99

FAMILIES = ["OOD-simple", "simple", "medium", "complex", "OOD-complex"]
TRAIN_FAMILIES = ["simple", "medium", "complex"]

# Le papier utilise 200/50/50 (annexe A). Réglé ici à 200/100/100 comme demandé.
N_TRAIN = 200
N_VAL = 100
N_TEST = 100

SEED_BASE = {"train": 0, "val": 10_000, "test": 20_000}


# ---------------------------------------------------------------------------
# Vérification des signatures (à lancer avec --check avant de générer)
# ---------------------------------------------------------------------------
def check_signatures():
    candidates = {
        "Burgers": ex.stepper.Burgers,
        "NavierStokesVorticity": ex.stepper.NavierStokesVorticity,
        "KolmogorovFlowVorticity": getattr(ex.stepper, "KolmogorovFlowVorticity", None),
        "Wave": getattr(ex.stepper, "Wave", None),
        "KuramotoSivashinsky": getattr(ex.stepper, "KuramotoSivashinsky", None),
        "reaction.FisherKPP": getattr(getattr(ex.stepper, "reaction", None), "FisherKPP", None),
        "reaction.GrayScott": getattr(getattr(ex.stepper, "reaction", None), "GrayScott", None),
        "reaction.SwiftHohenberg": getattr(getattr(ex.stepper, "reaction", None), "SwiftHohenberg", None),
    }
    print("Signatures réelles des steppers installés :\n")
    for name, cls in candidates.items():
        if cls is None:
            print(f"  {name} : NON TROUVÉ à ce chemin (vérifier la structure du module ex.stepper)")
            continue
        try:
            sig = inspect.signature(cls.__init__)
            print(f"  {name}:\n    {sig}\n")
        except (TypeError, ValueError):
            print(f"  {name}: signature non introspectable\n")


# ---------------------------------------------------------------------------
# Générateurs de conditions initiales
# ---------------------------------------------------------------------------
def make_fourier_ic_generator(cutoff: int, amplitude: float, num_channels: int, max_one: bool = True):
    """CI = série de Fourier tronquée. Utilisée pour Burgers, Decay, Fisher-KPP,
    Swift-Hohenberg, Kuramoto-Sivashinsky. `max_one` reproduit le comportement
    "clampé/rescalé" (Fisher-KPP -> [0,1] nécessite un post-traitement séparé,
    voir make_fisher_kpp_ic_generator)."""
    base = ex.ic.RandomTruncatedFourierSeries(NUM_SPATIAL_DIMS, cutoff=cutoff, max_one=max_one)
    scaled = ex.ic.ScaledICGenerator(base, scale=amplitude)
    if num_channels == 1:
        return scaled
    return ex.ic.RandomMultiChannelICGenerator([scaled] * num_channels)


def fisher_kpp_ic_sample(key, cutoff: int):
    """IC Fisher-KPP : Fourier tronquée, clampée puis rescalée vers [0, 1]
    (papier A.3). ScaledICGenerator seul ne fait pas ce rescale min-max, donc
    on prend la sortie brute d'un générateur Fourier et on la renormalise
    nous-mêmes."""
    base = ex.ic.RandomTruncatedFourierSeries(NUM_SPATIAL_DIMS, cutoff=cutoff, max_one=False)
    raw = base(num_points=NUM_POINTS, key=key)  # (1, H, W)
    lo, hi = raw.min(), raw.max()
    rescaled = (raw - lo) / (hi - lo + 1e-12)
    return rescaled  # dans [0, 1], shape (1, H, W)


def blob_ic_sample(key, num_blobs: int, domain_extent: float, sigma_range=(0.002, 0.05),
                    center_frac: float = 0.5, zero_mean_unit_max: bool = False,
                    background=None):
    """Génère un champ 2D fait de `num_blobs` bosses gaussiennes à centres/
    variances aléatoires, dans une fraction centrale du domaine
    (`center_frac`). Utilisé pour Gray-Scott (A.1) et Wave (A.2).

    - Si `background` est None : renvoie juste la somme des bosses (utile pour
      Wave, où le champ est ensuite renormalisé zero-mean/unit-max).
    - Si `background` est un dict {"base": v, "bump": v} : renvoie un champ qui
      vaut `base` hors des bosses et interpole vers `bump` au centre des
      bosses (utile pour Gray-Scott, où u/v ont des niveaux de fond non nuls).
    """
    k_center, k_sigma, k_amp = jax.random.split(key, 3)
    lo = 0.5 - center_frac / 2
    hi = 0.5 + center_frac / 2
    centers = jax.random.uniform(k_center, (num_blobs, 2), minval=lo, maxval=hi) * domain_extent
    sigmas = jax.random.uniform(k_sigma, (num_blobs,), minval=sigma_range[0], maxval=sigma_range[1]) * domain_extent

    xs = jnp.linspace(0, domain_extent, NUM_POINTS, endpoint=False)
    xx, yy = jnp.meshgrid(xs, xs, indexing="ij")

    def one_blob(c, s):
        d2 = (xx - c[0]) ** 2 + (yy - c[1]) ** 2
        return jnp.exp(-d2 / (2 * s ** 2))

    bumps = jax.vmap(one_blob)(centers, sigmas)  # (num_blobs, H, W)
    field = jnp.sum(bumps, axis=0)
    field = jnp.clip(field, 0.0, 1.0)

    if background is None:
        return field[None]  # (1, H, W)

    out = background["base"] + (background["bump"] - background["base"]) * field
    return out[None]  # (1, H, W)


def wave_ic_sample(key, num_blobs: int, domain_extent: float):
    """IC Wave (A.2) : hauteur = bosses gaussiennes normalisées zero-mean/
    unit-max-abs ; vitesse initiale = 0. Renvoie un état à 2 canaux (h, v)."""
    h = blob_ic_sample(key, num_blobs, domain_extent, sigma_range=(0.002, 0.008))[0]
    h = h - h.mean()
    h = h / (jnp.max(jnp.abs(h)) + 1e-12)
    v = jnp.zeros_like(h)
    return jnp.stack([h, v], axis=0)  # (2, H, W)


def gray_scott_ic_sample(key, num_blobs: int, domain_extent: float):
    """IC Gray-Scott (A.1) : fond homogène (u=1, v=0), bosses gaussiennes
    créant une perturbation locale classique (u->0.5, v->0.25 au centre des
    bosses). Ces niveaux ne sont pas donnés explicitement dans le papier ;
    ce sont les valeurs standard utilisées dans la littérature Gray-Scott
    pour amorcer la réaction. À ajuster si besoin."""
    ku, kv = jax.random.split(key)
    u = blob_ic_sample(ku, num_blobs, domain_extent, background={"base": 1.0, "bump": 0.5})[0]
    v = blob_ic_sample(kv, num_blobs, domain_extent, background={"base": 0.0, "bump": 0.25})[0]
    return jnp.stack([u, v], axis=0)  # (2, H, W)


# ---------------------------------------------------------------------------
# Rollout générique (gère aussi le rejet de transitoire pour Kolmogorov/KS)
# ---------------------------------------------------------------------------
def rollout_from_ic_array(stepper, ic_array, num_frames: int, discard_frames: int = 0):
    """Déroule `stepper` à partir d'un état initial déjà matérialisé
    (array, pas un IC generator Exponax), avec option de rejet de
    transitoire (Kolmogorov : 600 frames simulées, 500 rejetées ;
    Kuramoto-Sivashinsky : 200 simulées, 100 rejetées)."""
    total = num_frames + discard_frames
    full_traj = ex.rollout(stepper, total - 1, include_init=True)(ic_array)  # (total, C, H, W)
    return full_traj[discard_frames:]  # (num_frames, C, H, W)


def generate_split_from_sampler(stepper, sample_fn, num_samples: int, seed: int,
                                 num_frames: int = NUM_FRAMES, discard_frames: int = 0):
    """Génère un split complet à partir d'une fonction `sample_fn(key) -> ic_array`
    (utilisée pour les IC "custom" : blobs, clamp/rescale, etc.)."""
    keys = jax.random.split(jax.random.PRNGKey(seed), num_samples)
    ics = jax.vmap(sample_fn)(keys)  # (N, C, H, W)
    rollout_fn = lambda ic: rollout_from_ic_array(stepper, ic, num_frames, discard_frames)
    trj = jax.vmap(rollout_fn)(ics)
    return np.asarray(trj)


def generate_split_builtin(stepper, ic_generator, num_samples: int, seed: int):
    """Génère un split via un IC generator Exponax standard (ex.build_ic_set +
    ex.rollout), comme dans le script Burgers/Decay d'origine."""
    ic_set = ex.build_ic_set(
        ic_generator, num_points=NUM_POINTS, num_samples=num_samples,
        key=jax.random.PRNGKey(seed),
    )
    trj = jax.vmap(ex.rollout(stepper, NUM_FRAMES - 1, include_init=True))(ic_set)
    return np.asarray(trj)


def build_dataset(name, split_fn, n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST,
                   complexity_values: dict = None):
    """split_fn(family, split, seed, n) -> np.array (n, T, C, H, W)
    complexity_values : dict optionnel {family: valeur} si la complexité
    n'est pas un simple cutoff Fourier (ex: Kolmogorov -> injection mode)."""
    data = {}
    for family in FAMILIES:
        if family in TRAIN_FAMILIES and n_train > 0:
            t0 = time.time()
            data[f"train_{family}"] = split_fn(family, "train", SEED_BASE["train"], n_train)
            print(f"  {name}/train_{family}: {data[f'train_{family}'].shape}  ({time.time()-t0:.1f}s)")
        if n_val > 0:
            t0 = time.time()
            data[f"val_{family}"] = split_fn(family, "val", SEED_BASE["val"], n_val)
            print(f"  {name}/val_{family}: {data[f'val_{family}'].shape}  ({time.time()-t0:.1f}s)")
        if n_test > 0:
            t0 = time.time()
            data[f"test_{family}"] = split_fn(family, "test", SEED_BASE["test"], n_test)
            print(f"  {name}/test_{family}: {data[f'test_{family}'].shape}  ({time.time()-t0:.1f}s)")
    np.savez_compressed(OUT_DIR / f"{name}.npz", **data)
    return data


CUTOFF_OF = {"OOD-simple": 1, "simple": 2, "medium": 3, "complex": 4, "OOD-complex": 5}
BLOBS_OF = {"OOD-simple": 1, "simple": 2, "medium": 3, "complex": 4, "OOD-complex": 5}
INJECTION_MODE_OF = {"OOD-simple": 2, "simple": 3, "medium": 4, "complex": 5, "OOD-complex": 6}


# ---------------------------------------------------------------------------
# A.1 Gray-Scott — L=1.35, dt=3.5, 14 sous-pas -> espacement 49.0
# ---------------------------------------------------------------------------
def build_gray_scott():
    base = ex.stepper.reaction.GrayScott(
        num_spatial_dims=NUM_SPATIAL_DIMS, domain_extent=1.35, num_points=NUM_POINTS,
        dt=3.5, diffusivity_1=2e-5, diffusivity_2=1e-5, feed_rate=0.044, kill_rate=0.06,
    )
    stepper = ex.RepeatedStepper(base, num_sub_steps=14)

    def split_fn(family, split, seed, n):
        seed_offset = BLOBS_OF[family]
        sample_fn = lambda key: gray_scott_ic_sample(key, BLOBS_OF[family], domain_extent=1.35)
        return generate_split_from_sampler(stepper, sample_fn, n, seed + seed_offset)

    return build_dataset("gray_scott", split_fn)


# ---------------------------------------------------------------------------
# A.2 Wave — L=1.0, dt=0.01, 1 sous-pas ; 2 canaux (h, v)
# ---------------------------------------------------------------------------
def build_wave():
    stepper = ex.stepper.Wave(
        num_spatial_dims=NUM_SPATIAL_DIMS, domain_extent=1.0, num_points=NUM_POINTS,
        dt=0.01, speed_of_sound=1.0,
    )

    def split_fn(family, split, seed, n):
        seed_offset = BLOBS_OF[family]
        sample_fn = lambda key: wave_ic_sample(key, BLOBS_OF[family], domain_extent=1.0)
        return generate_split_from_sampler(stepper, sample_fn, n, seed + seed_offset)

    return build_dataset("wave", split_fn)


# ---------------------------------------------------------------------------
# A.3 Fisher-KPP — L=4.0, dt=0.001, 2 sous-pas -> espacement 0.002
# ---------------------------------------------------------------------------
def build_fisher_kpp():
    base = ex.stepper.reaction.FisherKPP(
        num_spatial_dims=NUM_SPATIAL_DIMS, domain_extent=4.0, num_points=NUM_POINTS,
        dt=0.001, diffusivity=0.005, reactivity=20.0,
    )
    stepper = ex.RepeatedStepper(base, num_sub_steps=2)

    def split_fn(family, split, seed, n):
        cutoff = CUTOFF_OF[family]
        sample_fn = lambda key: fisher_kpp_ic_sample(key, cutoff)
        return generate_split_from_sampler(stepper, sample_fn, n, seed + cutoff)

    return build_dataset("fisher_kpp", split_fn)


# ---------------------------------------------------------------------------
# A.4 Burgers — L=3.0, dt=0.02, 2 sous-pas (API confirmée)
# ---------------------------------------------------------------------------
def build_burgers():
    base = ex.stepper.Burgers(
        num_spatial_dims=NUM_SPATIAL_DIMS, domain_extent=3.0, num_points=NUM_POINTS,
        dt=0.02, diffusivity=0.01, convection_scale=2.0, single_channel=False,
        order=2, dealiasing_fraction=2 / 3,
    )
    stepper = ex.RepeatedStepper(base, num_sub_steps=2)

    def split_fn(family, split, seed, n):
        cutoff = CUTOFF_OF[family]
        ic_gen = make_fourier_ic_generator(cutoff, amplitude=0.5, num_channels=2)
        return generate_split_builtin(stepper, ic_gen, n, seed + cutoff)

    return build_dataset("burgers", split_fn)


# ---------------------------------------------------------------------------
# A.5 Swift-Hohenberg — L=30.0, dt=0.05 (espacement direct, pas de sous-pas)
# ---------------------------------------------------------------------------
def build_swift_hohenberg():
    stepper = ex.stepper.reaction.SwiftHohenberg(
        num_spatial_dims=NUM_SPATIAL_DIMS, domain_extent=30.0, num_points=NUM_POINTS,
        dt=0.05, reactivity=0.7, critical_number=1.0,
        order=2, dealiasing_fraction=1 / 2,
    )

    def split_fn(family, split, seed, n):
        cutoff = CUTOFF_OF[family]
        ic_gen = make_fourier_ic_generator(cutoff, amplitude=0.5, num_channels=1)
        return generate_split_builtin(stepper, ic_gen, n, seed + cutoff)

    return build_dataset("swift_hohenberg", split_fn)


# ---------------------------------------------------------------------------
# A.6 Decay — L=5.0, dt=0.06, 5 sous-pas (API confirmée)
# ---------------------------------------------------------------------------
def build_decay():
    base = ex.stepper.NavierStokesVorticity(
        num_spatial_dims=NUM_SPATIAL_DIMS, domain_extent=5.0, num_points=NUM_POINTS,
        dt=0.06, diffusivity=9e-4, vorticity_convection_scale=1.0, drag=0.0,
        order=2, dealiasing_fraction=2 / 3,
    )
    stepper = ex.RepeatedStepper(base, num_sub_steps=5)

    def split_fn(family, split, seed, n):
        cutoff = CUTOFF_OF[family]
        ic_gen = make_fourier_ic_generator(cutoff, amplitude=1.0, num_channels=1)
        return generate_split_builtin(stepper, ic_gen, n, seed + cutoff)

    return build_dataset("decay", split_fn)


# ---------------------------------------------------------------------------
# A.7 Kolmogorov Flow — L=6.0, dt=0.01, 2 sous-pas ; complexité = mode
# d'injection (PAS la CI). 600 frames simulées, 500 rejetées.
# ---------------------------------------------------------------------------
def build_kolmogorov():
    def make_stepper(injection_mode):
        base = ex.stepper.KolmogorovFlowVorticity(
            num_spatial_dims=NUM_SPATIAL_DIMS, domain_extent=6.0, num_points=NUM_POINTS,
            dt=0.01, diffusivity=0.01, convection_scale=1.2, drag=-0.1,
            injection_mode=injection_mode, injection_scale=1.2,
        )
        return ex.RepeatedStepper(base, num_sub_steps=2)

    # La CI elle-même n'est pas l'axe de complexité ici (cf. papier A.7) ;
    # on utilise une CI Fourier générique modérée (cutoff=3) pour toutes les
    # familles — hypothèse raisonnable car le rejet de 500 frames de
    # transitoire rend l'état initial non pertinent une fois le régime
    # forcé atteint.
    def sample_fn(key):
        base_ic = ex.ic.RandomTruncatedFourierSeries(NUM_SPATIAL_DIMS, cutoff=3, max_one=True)
        return ex.ic.ScaledICGenerator(base_ic, scale=1.0)(num_points=NUM_POINTS, key=key)

    def split_fn(family, split, seed, n):
        mode = INJECTION_MODE_OF[family]
        stepper = make_stepper(mode)
        return generate_split_from_sampler(stepper, sample_fn, n, seed + mode,
                                            num_frames=NUM_FRAMES, discard_frames=500)

    return build_dataset("kolmogorov", split_fn)


# ---------------------------------------------------------------------------
# A.8 Kuramoto-Sivashinsky — L=42.0, dt=0.1, 1 sous-pas ; 200 frames
# simulées, 100 rejetées.
# ---------------------------------------------------------------------------
def build_kuramoto_sivashinsky():
    stepper = ex.stepper.KuramotoSivashinsky(
        num_spatial_dims=NUM_SPATIAL_DIMS, domain_extent=42.0, num_points=NUM_POINTS,
        dt=0.1, gradient_norm_scale=1.0, second_order_scale=1.0,
        fourth_order_scale=1.0,
    )

    def split_fn(family, split, seed, n):
        cutoff = CUTOFF_OF[family]

        def sample_fn(key):
            base_ic = ex.ic.RandomTruncatedFourierSeries(NUM_SPATIAL_DIMS, cutoff=cutoff, max_one=True)
            return ex.ic.ScaledICGenerator(base_ic, scale=1.0)(num_points=NUM_POINTS, key=key)

        return generate_split_from_sampler(stepper, sample_fn, n, seed + cutoff,
                                            num_frames=NUM_FRAMES, discard_frames=100)

    return build_dataset("kuramoto_sivashinsky", split_fn)


# ---------------------------------------------------------------------------
BUILDERS = {
    "gray_scott": build_gray_scott,
    "wave": build_wave,
    "fisher_kpp": build_fisher_kpp,
    "burgers": build_burgers,
    "swift_hohenberg": build_swift_hohenberg,
    "decay": build_decay,
    "kolmogorov": build_kolmogorov,
    "kuramoto_sivashinsky": build_kuramoto_sivashinsky,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="affiche les signatures des steppers et quitte")
    parser.add_argument("--only", nargs="*", default=None, choices=list(BUILDERS.keys()),
                         help="ne générer que ces dynamiques")
    args = parser.parse_args()

    if args.check:
        check_signatures()
        return

    names = args.only or list(BUILDERS.keys())
    for name in names:
        print(f"\n=== Génération : {name} ===")
        BUILDERS[name]()

    print(f"\nTerminé. Fichiers dans {OUT_DIR.resolve()}/")


if __name__ == "__main__":
    main()