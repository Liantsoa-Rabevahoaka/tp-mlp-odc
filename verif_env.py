#!/usr/bin/env python3
"""
Vérification de l'environnement, à lancer à 14h00 avant le TP.

Ce script ne s'arrête pas à la première erreur : il fait tous les contrôles,
affiche une ligne ✅/❌ pour chacun, puis termine par un résumé en français.
Si tout est vert, exit code 0. Sinon, exit code 1.

Usage : python verif_env.py
"""
import os
import sys
import tempfile

erreurs = []


def verifie(nom, fonction):
    """Exécute `fonction`, affiche ✅/❌ `nom`, et note l'erreur si besoin."""
    try:
        detail = fonction()
        print(f"✅ {nom}" + (f" : {detail}" if detail else ""))
        return True
    except Exception as erreur:
        print(f"❌ {nom} : {erreur}")
        erreurs.append(nom)
        return False


def check_python():
    version = sys.version_info
    if version < (3, 9):
        raise RuntimeError(f"Python {version.major}.{version.minor} détecté, il faut >= 3.9")
    return f"Python {version.major}.{version.minor}.{version.micro}"


def check_numpy():
    import numpy
    return f"version {numpy.__version__}"


def check_matplotlib():
    import matplotlib
    return f"version {matplotlib.__version__}"


def check_torch():
    import torch
    return f"version {torch.__version__}"


def check_matplotlib_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0, 1, 4])
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        chemin = f.name
    try:
        fig.savefig(chemin)
        plt.close(fig)
    finally:
        if os.path.exists(chemin):
            os.remove(chemin)
    return "figure créée et enregistrée"


def check_torch_forward_backward():
    import torch
    from torch import nn

    couche = nn.Linear(4, 2)
    x = torch.randn(3, 4)
    y = couche(x)
    y.sum().backward()
    assert couche.weight.grad is not None, "le gradient n'a pas été calculé"
    return "forward + backward OK"


def check_modules_locaux():
    sys.path.insert(0, "src")
    import couleurs  # noqa: F401
    import mnist  # noqa: F401
    import viz  # noqa: F401
    return "mnist, viz, couleurs importés depuis src/"


def check_donnees_mnist():
    chemin_cache = os.path.join("data", "mnist.npz")
    if os.path.exists(chemin_cache):
        return "données déjà en cache (data/mnist.npz)"

    sys.path.insert(0, "src")
    import mnist
    try:
        mnist.charge_mnist()
    except Exception:
        raise RuntimeError(
            "pas de cache local et le téléchargement a échoué. "
            "copiez mnist.npz depuis la clé USB dans data/ (plan B hors-ligne)"
        )
    return "données téléchargées et mises en cache"


def check_notebooks():
    attendus = [
        "notebooks/00_tp.ipynb",
        "notebooks/01_perceptron.ipynb",
        "notebooks/02_mlp_a_la_main.ipynb",
        "notebooks/03_mlp_pytorch.ipynb",
        "notebooks/04_limites.ipynb",
    ]
    manquants = [f for f in attendus if not os.path.exists(f)]
    if manquants:
        raise RuntimeError(f"fichiers manquants : {', '.join(manquants)}")
    return f"{len(attendus)} notebooks présents"


def main():
    print("=== Vérification de l'environnement, formation Deep Learning ===\n")

    verifie("Python >= 3.9", check_python)
    verifie("Import numpy", check_numpy)
    verifie("Import matplotlib", check_matplotlib)
    verifie("Import torch", check_torch)
    verifie("matplotlib peut construire une figure", check_matplotlib_figure)
    verifie("torch peut faire forward + backward", check_torch_forward_backward)
    verifie("Modules locaux (src/mnist, viz, couleurs)", check_modules_locaux)
    verifie("Données MNIST disponibles", check_donnees_mnist)
    verifie("Les notebooks sont présents", check_notebooks)

    print()
    if not erreurs:
        print("Tout est prêt, bon TP !")
        return 0

    print("Il reste des choses à régler avant 14h15 :")
    for i, nom in enumerate(erreurs, start=1):
        print(f"  {i}. {nom}")
    print("\nEn cas de souci réseau pour les données : copiez mnist.npz depuis")
    print("la clé USB fournie dans le dossier data/ (voir README.md, section")
    print("« Plan B hors-ligne »).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
