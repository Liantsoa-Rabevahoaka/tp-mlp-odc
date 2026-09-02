"""
Fonctions d'affichage pour les notebooks de la formation.

Ces fonctions existent pour que les cellules des notebooks parlent
d'idées (perceptron, gradient, poids appris...) plutôt que de code
matplotlib. Chaque fonction :
  - applique le style graphique de la formation (couleurs.applique_style),
  - affiche la figure avec plt.show(),
  - ne renvoie rien, sauf plot_confusion qui renvoie la matrice calculée.
"""

import matplotlib.pyplot as plt
import numpy as np

from couleurs import applique_style, BLEU, VIOLET, BLEU_CLAIR, VIOLET_CLAIR, TEXTE, CMAP_CHIFFRES, CMAP_POIDS

ROUGE_ERREUR = "#c0392b"


def _vers_images(X):
    """Convertit X (n, 784) ou (n, 28, 28) en tableau d'images (n, 28, 28)."""
    X = np.asarray(X)
    if X.ndim == 2 and X.shape[1] == 784:
        return X.reshape(-1, 28, 28)
    return X


def affiche_chiffres(X, y=None, predictions=None, n=25, titre=None):
    """
    Affiche une grille de chiffres MNIST.

    X : images, forme (n, 784) ou (n, 28, 28).
    y : labels vrais (optionnel).
    predictions : labels prédits (optionnel). Si fournis avec y, le titre de
        chaque case affiche "vrai / prédit" et devient rouge en cas d'erreur.
    n : nombre d'images à afficher (environ 5 par ligne).
    """
    applique_style()

    images = _vers_images(X)
    n = min(n, images.shape[0])

    n_colonnes = 5
    n_lignes = int(np.ceil(n / n_colonnes))

    fig, axes = plt.subplots(n_lignes, n_colonnes, figsize=(n_colonnes * 1.6, n_lignes * 1.8))
    axes = np.atleast_1d(axes).ravel()

    for i in range(len(axes)):
        ax = axes[i]
        ax.set_xticks([])
        ax.set_yticks([])
        if i >= n:
            ax.axis("off")
            continue

        ax.imshow(images[i], cmap=CMAP_CHIFFRES)

        if y is not None and predictions is not None:
            vrai, pred = y[i], predictions[i]
            couleur = ROUGE_ERREUR if vrai != pred else TEXTE
            ax.set_title(f"{vrai} / {pred}", color=couleur, fontsize=10)
        elif y is not None:
            ax.set_title(str(y[i]), color=TEXTE, fontsize=10)

    if titre:
        fig.suptitle(titre)
    fig.tight_layout()
    plt.show()


def plot_courbes(historique, titre="Courbes d'apprentissage"):
    """
    Affiche une ou plusieurs courbes d'apprentissage.

    historique : dict label -> liste de floats, ex. {"train": [...], "validation": [...]}
    """
    applique_style()

    fig, ax = plt.subplots(figsize=(6, 4))
    for label, valeurs in historique.items():
        ax.plot(valeurs, label=label)

    ax.set_xlabel("époque")
    ax.set_ylabel("loss")
    ax.set_title(titre)
    ax.legend()
    fig.tight_layout()
    plt.show()


def plot_comparaison(resultats, titre, ylabel="loss"):
    """
    Affiche plusieurs courbes sur les mêmes axes, pour comparer par exemple
    plusieurs taux d'apprentissage (eta=1e-5 / 1e-3 / 0.1 / 10).

    resultats : dict label -> liste de floats. Une série peut contenir des
    nan/inf (divergence) : dans ce cas l'axe y est quand même lisible et la
    légende indique " (diverge)" pour cette série.
    """
    applique_style()

    fig, ax = plt.subplots(figsize=(6, 4))

    toutes_valeurs_finies = []
    for label, valeurs in resultats.items():
        valeurs = np.asarray(valeurs, dtype=float)
        diverge = not np.all(np.isfinite(valeurs))
        label_legende = label + (" (diverge)" if diverge else "")
        ax.plot(valeurs, label=label_legende)
        finies = valeurs[np.isfinite(valeurs)]
        if finies.size > 0:
            toutes_valeurs_finies.append(finies)

    if toutes_valeurs_finies:
        toutes = np.concatenate(toutes_valeurs_finies)
        bas = np.nanpercentile(toutes, 1)
        haut = np.nanpercentile(toutes, 99)
        if bas == haut:
            bas -= 1
            haut += 1
        marge = (haut - bas) * 0.1
        ax.set_ylim(bas - marge, haut + marge)

    ax.set_xlabel("époque")
    ax.set_ylabel(ylabel)
    ax.set_title(titre)
    ax.legend()
    fig.tight_layout()
    plt.show()


def plot_confusion(y_vrai, y_pred, titre="Matrice de confusion"):
    """
    Calcule et affiche la matrice de confusion 10x10 (chiffres 0..9).

    Renvoie la matrice de confusion (tableau numpy 10x10).
    """
    applique_style()

    y_vrai = np.asarray(y_vrai)
    y_pred = np.asarray(y_pred)

    matrice = np.zeros((10, 10), dtype=np.int64)
    for vrai, pred in zip(y_vrai, y_pred):
        matrice[vrai, pred] += 1

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(matrice, cmap=CMAP_CHIFFRES)

    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("prédit")
    ax.set_ylabel("vrai")
    ax.set_title(titre)

    seuil = matrice.max() / 2 if matrice.max() > 0 else 0
    for i in range(10):
        for j in range(10):
            valeur = matrice[i, j]
            couleur = "white" if valeur > seuil else TEXTE
            ax.text(j, i, str(valeur), ha="center", va="center", color=couleur, fontsize=8)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    plt.show()

    return matrice


def montre_poids(W, titre="Poids appris"):
    """
    Affiche les poids appris par un ou plusieurs neurones, reformés en
    images 28x28.

    W : forme (784,) pour un seul perceptron, ou (784, k) / (k, 784) pour
    plusieurs neurones (l'orientation est détectée automatiquement).
    Au maximum 10 neurones sont affichés.
    """
    applique_style()

    W = np.asarray(W)

    if W.ndim == 1:
        neurones = W.reshape(1, -1)
    elif W.shape[0] == 784:
        # (784, k) -> une colonne par neurone
        neurones = W.T
    elif W.shape[1] == 784:
        # (k, 784) -> une ligne par neurone
        neurones = W
    else:
        raise ValueError(f"Forme de poids inattendue : {W.shape} (une dimension doit valoir 784)")

    n = min(neurones.shape[0], 10)
    vmax = np.abs(neurones[:n]).max()
    if vmax == 0:
        vmax = 1.0

    fig, axes = plt.subplots(1, n, figsize=(n * 1.6, 1.8))
    axes = np.atleast_1d(axes).ravel()

    for i in range(n):
        ax = axes[i]
        image = neurones[i].reshape(28, 28)
        im = ax.imshow(image, cmap=CMAP_POIDS, vmin=-vmax, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(titre)
    fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    plt.show()


def montre_frontiere(X, y, w, b, titre="Frontière de décision"):
    """
    Affiche la frontière de décision d'un perceptron dans le plan 2D.

    X : forme (n, 2). y : labels binaires (0/1). w : forme (2,), b : scalaire.
    La droite w[0]*x1 + w[1]*x2 + b = 0 est tracée sur la plage de x
    actuelle, avec les deux demi-plans légèrement teintés. Fonctionne aussi
    quand w[1] == 0 (droite verticale).
    """
    applique_style()

    X = np.asarray(X)
    y = np.asarray(y)
    w = np.asarray(w, dtype=float)

    fig, ax = plt.subplots(figsize=(5.5, 5))

    classe0 = X[y == 0]
    classe1 = X[y == 1]
    ax.scatter(classe0[:, 0], classe0[:, 1], color=BLEU, label="classe 0", zorder=3)
    ax.scatter(classe1[:, 0], classe1[:, 1], color=VIOLET, label="classe 1", zorder=3)

    marge = 0.5
    x_min, x_max = X[:, 0].min() - marge, X[:, 0].max() + marge
    y_min, y_max = X[:, 1].min() - marge, X[:, 1].max() + marge
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    resolution = 200
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    zz = w[0] * xx + w[1] * yy + b
    ax.contourf(xx, yy, zz, levels=[-np.inf, 0, np.inf], colors=[BLEU_CLAIR, VIOLET_CLAIR], alpha=0.5, zorder=1)

    if abs(w[1]) > 1e-12:
        x1 = np.array([x_min, x_max])
        x2 = -(w[0] * x1 + b) / w[1]
        ax.plot(x1, x2, color=TEXTE, linewidth=2, zorder=2)
    elif abs(w[0]) > 1e-12:
        # droite verticale : w[1] == 0, w[0]*x1 + b = 0 -> x1 = -b / w[0]
        x1 = -b / w[0]
        ax.axvline(x1, color=TEXTE, linewidth=2, zorder=2)
    # si w[0] == w[1] == 0, pas de frontière à tracer

    ax.set_title(titre)
    ax.legend()
    fig.tight_layout()
    plt.show()
