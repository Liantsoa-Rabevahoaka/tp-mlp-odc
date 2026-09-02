"""
Chargement du jeu de données MNIST.

Ce module télécharge les 4 fichiers idx.gz classiques de MNIST une seule
fois, les parse, puis met le résultat en cache dans un fichier .npz unique
(données brutes, non normalisées) pour ne plus jamais retoucher le réseau
par la suite.

Fonction publique :

    charge_mnist(aplati=True, normalise=True, sous_ensemble=None, dossier=None)
"""

import gzip
import os
import struct
import urllib.request

import numpy as np

# Miroirs essayés dans l'ordre (yann.lecun.com n'est pas fiable en salle de
# classe, on ne l'utilise pas comme source principale).
MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
]

FICHIERS = {
    "X_train": "train-images-idx3-ubyte.gz",
    "y_train": "train-labels-idx1-ubyte.gz",
    "X_test": "t10k-images-idx3-ubyte.gz",
    "y_test": "t10k-labels-idx1-ubyte.gz",
}

TIMEOUT = 15  # secondes
USER_AGENT = "Mozilla/5.0 (compatible; formation-deep-learning/1.0)"


def _telecharge_fichier(nom_fichier, dossier):
    """Télécharge `nom_fichier` depuis le premier miroir qui répond,
    et l'enregistre dans `dossier`. Ne fait rien si le fichier existe déjà."""
    chemin = os.path.join(dossier, nom_fichier)
    if os.path.exists(chemin):
        return chemin

    derniere_erreur = None
    for mirroir in MIRRORS:
        url = mirroir + nom_fichier
        try:
            print(f"Téléchargement de MNIST depuis {url} ...")
            requete = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(requete, timeout=TIMEOUT) as reponse:
                donnees = reponse.read()
            with open(chemin, "wb") as f:
                f.write(donnees)
            return chemin
        except Exception as erreur:
            derniere_erreur = erreur
            continue

    raise RuntimeError(
        "Impossible de télécharger MNIST (tous les miroirs ont échoué : "
        f"{derniere_erreur}).\n"
        "Plan B hors-ligne : copiez le fichier 'mnist.npz' depuis la clé USB "
        f"fournie dans le dossier '{dossier}/'."
    )


def _parse_images(chemin_gz):
    """Parse un fichier idx3 (images) compressé en gzip -> tableau uint8 (n, 28, 28)."""
    with gzip.open(chemin_gz, "rb") as f:
        magique, n, lignes, colonnes = struct.unpack(">IIII", f.read(16))
        assert magique == 2051, f"Magique inattendu pour un fichier d'images : {magique}"
        donnees = np.frombuffer(f.read(), dtype=np.uint8)
        donnees = donnees.reshape(n, lignes, colonnes)
    return donnees


def _parse_labels(chemin_gz):
    """Parse un fichier idx1 (labels) compressé en gzip -> tableau uint8 (n,)."""
    with gzip.open(chemin_gz, "rb") as f:
        magique, n = struct.unpack(">II", f.read(8))
        assert magique == 2049, f"Magique inattendu pour un fichier de labels : {magique}"
        donnees = np.frombuffer(f.read(), dtype=np.uint8)
    return donnees


def _construit_cache(dossier):
    """Télécharge (si besoin) et parse les 4 fichiers MNIST, puis construit
    le cache .npz avec les données brutes (uint8, non normalisées)."""
    os.makedirs(dossier, exist_ok=True)

    chemins = {cle: _telecharge_fichier(nom, dossier) for cle, nom in FICHIERS.items()}

    X_train = _parse_images(chemins["X_train"])
    y_train = _parse_labels(chemins["y_train"])
    X_test = _parse_images(chemins["X_test"])
    y_test = _parse_labels(chemins["y_test"])

    print("Construction du cache mnist.npz ...")
    chemin_cache = os.path.join(dossier, "mnist.npz")
    np.savez_compressed(
        chemin_cache,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    return X_train, y_train, X_test, y_test


def charge_mnist(aplati=True, normalise=True, sous_ensemble=None, dossier=None):
    """
    Charge le jeu de données MNIST et renvoie (X_train, y_train, X_test, y_test).

    Au premier appel, télécharge les fichiers officiels depuis un miroir et
    construit un cache local `dossier/mnist.npz` contenant les données
    brutes (uint8). Les appels suivants relisent directement ce cache, sans
    jamais retoucher le réseau.

    Paramètres
    ----------
    aplati : bool
        Si True, les images sont aplaties en vecteurs de taille 784.
        Si False, elles gardent la forme (28, 28).
    normalise : bool
        Si True, les pixels sont ramenés dans [0, 1] (float32).
        Si False, les pixels restent dans [0, 255] mais en float32 (pas en
        uint8, pour éviter les débordements de calcul). Utile pour montrer
        en cours ce qui se passe quand on entraîne SANS normalisation.
    sous_ensemble : liste de chiffres, optionnel
        Par exemple [0, 1] pour ne garder que les 0 et les 1 (utile pour un
        perceptron binaire). Les labels gardent leur valeur d'origine (0..9)
        : si le notebook a besoin de labels 0..k-1, c'est à lui de les
        remapper.
    dossier : str ou None
        Dossier du cache mnist.npz. Par défaut, le dossier data/ du dépôt,
        résolu à partir de l'emplacement de ce fichier (donc indépendant du
        répertoire de travail).

    Renvoie
    -------
    X_train, y_train, X_test, y_test : tableaux numpy
    """
    if dossier is None:
        # On vise toujours le dossier data/ du dépôt, quel que soit le
        # répertoire courant : les notebooks vivent dans notebooks/ et les
        # corrigés dans notebooks/solutions/.
        dossier = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), os.pardir, "data"
        )
    chemin_cache = os.path.join(dossier, "mnist.npz")

    if os.path.exists(chemin_cache):
        with np.load(chemin_cache) as cache:
            X_train = cache["X_train"]
            y_train = cache["y_train"]
            X_test = cache["X_test"]
            y_test = cache["y_test"]
    else:
        X_train, y_train, X_test, y_test = _construit_cache(dossier)

    # On travaille sur une copie pour ne jamais modifier les données brutes
    # mises en cache.
    X_train = X_train.copy()
    y_train = y_train.copy()
    X_test = X_test.copy()
    y_test = y_test.copy()

    # 1. Filtrage sur un sous-ensemble de chiffres.
    if sous_ensemble is not None:
        chiffres = set(sous_ensemble)
        masque_train = np.isin(y_train, list(chiffres))
        masque_test = np.isin(y_test, list(chiffres))
        X_train, y_train = X_train[masque_train], y_train[masque_train]
        X_test, y_test = X_test[masque_test], y_test[masque_test]

    # 2. Normalisation (ou simple conversion en float32).
    if normalise:
        X_train = X_train.astype(np.float32) / 255.0
        X_test = X_test.astype(np.float32) / 255.0
    else:
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)

    # 3. Aplatissement en vecteurs de 784 valeurs.
    if aplati:
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)

    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)

    # Vérification finale de cohérence des formes.
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    if aplati:
        assert X_train.shape[1:] == (784,)
        assert X_test.shape[1:] == (784,)
    else:
        assert X_train.shape[1:] == (28, 28)
        assert X_test.shape[1:] == (28, 28)

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = charge_mnist()
    print("X_train :", X_train.shape, X_train.dtype)
    print("y_train :", y_train.shape, y_train.dtype)
    print("X_test  :", X_test.shape, X_test.dtype)
    print("y_test  :", y_test.shape, y_test.dtype)
