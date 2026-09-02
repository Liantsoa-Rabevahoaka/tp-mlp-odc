"""
Palette et style graphique communs à tous les notebooks.

Les couleurs correspondent à celles utilisées dans le diaporama du matin :
on les réutilise l'après-midi pour que les graphiques restent cohérents.

Utilisation typique dans un notebook :

    from couleurs import applique_style
    applique_style()
"""

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --- Couleurs de la charte -------------------------------------------------

BLEU = "#169dd9"
VIOLET = "#5b4294"
BLEU_CLAIR = "#e8f5fc"
VIOLET_CLAIR = "#ece8f6"
GRIS = "#e2e2e1"
TEXTE = "#5a5c82"

# Cycle de couleurs utilisé pour les courbes multiples (comparaisons, etc.)
CYCLE = [BLEU, VIOLET, "#23244d", "#1580b3"]

# Colormap pour afficher les chiffres MNIST : du blanc vers le bleu.
CMAP_CHIFFRES = LinearSegmentedColormap.from_list(
    "chiffres", ["#ffffff", BLEU]
)

# Colormap divergente pour afficher des poids signés : violet -> blanc -> bleu.
CMAP_POIDS = LinearSegmentedColormap.from_list(
    "poids", [VIOLET, "#ffffff", BLEU]
)


def applique_style():
    """
    Configure les rcParams de matplotlib pour que les graphiques
    ressemblent visuellement au diaporama.

    À appeler une fois en début de notebook (ou automatiquement,
    puisque les fonctions de viz.py l'appellent elles-mêmes).
    """
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=CYCLE)
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["figure.facecolor"] = "white"

    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.color"] = GRIS

    plt.rcParams["text.color"] = TEXTE
    plt.rcParams["axes.labelcolor"] = TEXTE
    plt.rcParams["xtick.color"] = TEXTE
    plt.rcParams["ytick.color"] = TEXTE
    plt.rcParams["axes.edgecolor"] = TEXTE

    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False

    plt.rcParams["font.size"] = 11
