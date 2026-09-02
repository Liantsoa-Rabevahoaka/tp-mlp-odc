# ===MD===
# Notebook 1 · Le perceptron à la main

Au programme : un perceptron complet, la règle de mise à jour du matin, son échec sur
XOR, sa réussite sur « 0 contre 1 », et les poids appris affichés en image.

Le code reprend la notation du matin (`x`, `w`, `b`, `z`, `y_chapeau`, `eta`).

> 5 lignes à écrire en tout. Si tu bloques, prends le corrigé et avance : l'essentiel est
> dans les graphiques.
# ===CODE===
import pathlib
import sys

# Racine du dépôt, qu'on lance ce notebook depuis notebooks/ ou notebooks/solutions/.
RACINE = next(p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]
              if (p / "src" / "mnist.py").exists())
sys.path.insert(0, str(RACINE / "src"))

import numpy as np
import matplotlib.pyplot as plt

import viz
from couleurs import applique_style, BLEU, VIOLET

applique_style()
np.random.seed(0)   # même affichage sur ton écran et au vidéoprojecteur
# ===MD===
---
# Partie A · XOR

XOR vaut 1 quand **exactement une** des deux entrées vaut 1.

| x₁ | x₂ | y |
|----|----|---|
| 0  | 0  | 0 |
| 0  | 1  | 1 |
| 1  | 0  | 1 |
| 1  | 1  | 0 |

Quatre points, aucun téléchargement. Essaie de tracer une seule droite qui sépare les
bleus des violets.
# ===CODE===
X_xor = np.array([[0.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 0.0],
                  [1.0, 1.0]])
y_xor = np.array([0, 1, 1, 0])

plt.figure(figsize=(4, 4))
for classe, couleur in [(0, VIOLET), (1, BLEU)]:
    points = X_xor[y_xor == classe]
    plt.scatter(points[:, 0], points[:, 1], c=couleur, s=200, label=f"classe {classe}")
plt.xlabel("x₁"); plt.ylabel("x₂"); plt.title("XOR")
plt.legend(); plt.show()
# ===MD===
## A.1 · La fonction seuil

$$\hat{y} = f(z) = \begin{cases} 1 & \text{si } z \geq 0 \\ 0 & \text{sinon}\end{cases}
\qquad z = w \cdot x + b$$

Deux fonctions à écrire.
# ===CODE===
def seuil(z):
    """1 si z >= 0, 0 sinon. Marche aussi sur un tableau."""
    # ===SOL=== renvoie 1 quand z >= 0 et 0 sinon (np.where t'aide)
    return np.where(z >= 0, 1, 0)
    # ===ENDSOL===


def predit(x, w, b):
    """Prédiction du perceptron pour un exemple x, ou pour un lot d'exemples."""
    # ===SOL=== calcule z = w · x + b, puis applique seuil()
    z = x @ w + b
    return seuil(z)
    # ===ENDSOL===


# Avec ces poids-là, le neurone doit calculer le OU logique.
assert list(predit(X_xor, np.array([1.0, 1.0]), -0.5)) == [0, 1, 1, 1]
print("OK : ce neurone calcule le OU logique. Pas encore XOR.")
# ===MD===
## A.2 · La règle de mise à jour

Celle du matin, à l'identique :

$$w \leftarrow w + \eta \cdot (y - \hat{y}) \cdot x
\qquad
b \leftarrow b + \eta \cdot (y - \hat{y})$$

Trois cas possibles :

| Situation | `y - y_chapeau` | Effet sur `w` |
|---|---|---|
| bonne prédiction | 0 | rien ne bouge |
| on attendait 1, on a prédit 0 | +1 | on augmente |
| on attendait 0, on a prédit 1 | −1 | on diminue |
# ===CODE===
def entraine_perceptron(X, y, eta=0.1, epoques=20, trace=False):
    """Entraîne un perceptron. Renvoie (w, b, erreurs_par_epoque)."""
    w = np.zeros(X.shape[1])
    b = 0.0
    erreurs = []

    for epoque in range(epoques):
        n_erreurs = 0
        for x, y_vrai in zip(X, y):
            y_chapeau = predit(x, w, b)

            # ===SOL=== applique les deux formules ci-dessus
            w = w + eta * (y_vrai - y_chapeau) * x
            b = b + eta * (y_vrai - y_chapeau)
            # ===ENDSOL===

            n_erreurs += int(y_chapeau != y_vrai)
        erreurs.append(n_erreurs)
        if trace:
            print(f"époque {epoque:3d} : {n_erreurs} erreur(s), w={w.round(2)} b={b:.2f}")

    return w, b, erreurs
# ===CODE===
w, b, erreurs = entraine_perceptron(X_xor, y_xor, eta=0.1, epoques=15, trace=True)
# ===CODE===
viz.plot_courbes({"erreurs sur XOR": erreurs}, titre="XOR : aucune convergence")
viz.montre_frontiere(X_xor, y_xor, w, b, titre="La meilleure droite possible, et elle rate")
# ===MD===
### Lecture des deux graphiques

Le nombre d'erreurs ne descend jamais à 0. Il oscille. `w` bouge, revient, repart.

Ni bug, ni `eta` mal réglé : aucun couple (`w`, `b`) ne satisfait les 4 points à la fois,
puisqu'un perceptron ne trace **qu'une seule droite**. C'est la limite vue ce matin.

> ✏️ **À toi** : relance avec `eta=0.5` et `epoques=500`. Le résultat change-t-il ?
# ===CODE===
w2, b2, erreurs2 = entraine_perceptron(X_xor, y_xor, eta=0.5, epoques=500)
print("erreurs sur les 10 dernières époques :", erreurs2[-10:])
# ===MD===
---
# Partie B · MNIST, 0 contre 1

Une droite ne suffit pas pour XOR. Pour distinguer un 0 d'un 1, ça devrait aller.

## B.1 · Regarder les données

La plupart des tutoriels cachent cette étape dans un `transform`. Ici chaque
transformation reste visible.
# ===CODE===
import mnist

X_train, y_train, X_test, y_test = mnist.charge_mnist(aplati=False, normalise=False)

print("X_train :", X_train.shape, X_train.dtype)
print("pixels de", X_train.min(), "à", X_train.max())
print("répartition des classes :", np.bincount(y_train))
# ===CODE===
viz.affiche_chiffres(X_train, y_train, n=25, titre="MNIST brut")
# ===MD===
### Trois décisions, trois raisons

**Aplatir en 784.** Le neurone attend un vecteur `x`, pas une grille. `28 × 28 = 784`.
Retiens cette ligne : on reviendra l'accuser au notebook 4.

**Diviser par 255.** Le pas de mise à jour vaut `eta * (y - ŷ) * x`, donc il est
proportionnel à `x`. Avec des pixels à 255, les pas sont 255 fois trop grands. On le
mesure en B.4.

**Garder un jeu de test intact.** Pour mesurer si le modèle généralise, pas s'il a
mémorisé. On s'en sert une fois, à la fin.
# ===MD===
## B.2 · Entraînement

Le perceptron ne sait faire que du binaire, donc on ne garde que les 0 et les 1.
# ===CODE===
X_train01, y_train01, X_test01, y_test01 = mnist.charge_mnist(
    aplati=True, normalise=True, sous_ensemble=[0, 1]
)

w01, b01, erreurs01 = entraine_perceptron(X_train01, y_train01, eta=0.01, epoques=5)

accuracy = (predit(X_test01, w01, b01) == y_test01).mean()
print(f"accuracy sur le jeu de test : {accuracy:.2%}")
assert accuracy > 0.99

viz.plot_courbes({"erreurs": erreurs01}, titre="0 contre 1 : cette fois ça converge")
# ===MD===
## B.3 · Regarder `w`

`w` a 784 composantes, une par pixel. Il se replie donc en une image 28×28.
# ===CODE===
viz.montre_poids(w01, titre="Les poids appris, repliés en 28×28")
# ===MD===
C'est le graphique le plus utile du notebook. Les poids ne sont pas une abstraction,
ce sont des pixels :

- **bleu** : de l'encre ici pousse vers la classe 1 (la barre du **1**) ;
- **violet** : pousse vers la classe 0 (l'anneau du **0**) ;
- **blanc** : poids nul, pixels inutiles (les bords).

`w · x` mesure la ressemblance entre l'image et ce gabarit. C'est tout ce que fait un
neurone.
# ===MD===
## B.4 · L'effet de `/255`

Même code, mêmes époques, même `eta`. Seule différence : les pixels vont de 0 à 255.
# ===CODE===
X_brut, y_brut, _, _ = mnist.charge_mnist(aplati=True, normalise=False, sous_ensemble=[0, 1])
_, _, erreurs_brut = entraine_perceptron(X_brut, y_brut, eta=0.01, epoques=5)

viz.plot_comparaison(
    {"pixels /255 (0 à 1)": erreurs01, "pixels bruts (0 à 255)": erreurs_brut},
    titre="Avec et sans normalisation",
    ylabel="erreurs par époque",
)
# ===MD===
`/255` n'est pas une incantation copiée d'un tutoriel : c'est un réglage d'échelle du pas
d'apprentissage, et on vient de le mesurer.

## B.5 · Le mur

0 contre 1, c'est facile : ces deux chiffres ne se ressemblent pas. Essayons des chiffres
qui se ressemblent.
# ===MD===
### Attention aux étiquettes

`charge_mnist` renvoie les vrais chiffres : pour `[3, 5]`, `y` contient des 3 et des 5.
Le perceptron, lui, ne répond que 0 ou 1. Sans traduction, la règle reçoit un écart de
`5 - 1 = 4` au lieu de `1 - 0 = 1` et part n'importe où.

Pour `[0, 1]` ça ne changeait rien, d'où un bug facile à ne pas voir.
# ===CODE===
def en_binaire(y, paire):
    """Traduit les étiquettes : le plus petit chiffre vers 0, l'autre vers 1."""
    return (y == max(paire)).astype(int)


for paire in ([0, 1], [3, 5], [4, 9]):
    Xa, ya, Xb, yb = mnist.charge_mnist(aplati=True, normalise=True, sous_ensemble=paire)
    wp, bp, _ = entraine_perceptron(Xa, en_binaire(ya, paire), eta=0.01, epoques=5)
    score = (predit(Xb, wp, bp) == en_binaire(yb, paire)).mean()
    print(f"{paire[0]} contre {paire[1]} : {score:.2%}")
# ===MD===
> ✏️ **À toi** : trouve la paire de chiffres que le perceptron confond le plus, puis
> affiche quelques images qu'il rate.
# ===CODE===
# ===SOL=== balaie les paires, garde la pire, affiche ses erreurs
pires = []
for a in range(10):
    for c in range(a + 1, 10):
        Xa, ya, Xb, yb = mnist.charge_mnist(aplati=True, normalise=True, sous_ensemble=[a, c])
        # 2000 exemples et 2 époques suffisent pour classer les paires entre elles.
        wp, bp, _ = entraine_perceptron(Xa[:2000], en_binaire(ya[:2000], [a, c]),
                                        eta=0.01, epoques=2)
        pires.append(((predit(Xb, wp, bp) == en_binaire(yb, [a, c])).mean(), a, c))

score, a, c = min(pires)
print(f"la pire paire : {a} contre {c}, {score:.2%}")

Xa, ya, Xb, yb = mnist.charge_mnist(aplati=True, normalise=True, sous_ensemble=[a, c])
wp, bp, _ = entraine_perceptron(Xa, en_binaire(ya, [a, c]), eta=0.01, epoques=3)
predictions = np.where(predit(Xb, wp, bp) == 1, c, a)
rates = predictions != yb
viz.affiche_chiffres(Xb[rates], yb[rates], predictions[rates], n=10, titre="Ce qu'il rate")
# ===ENDSOL===
# ===MD===
---
# À retenir

| | |
|---|---|
| Un perceptron trace une droite | `w` l'oriente, `b` la place |
| XOR est impossible | par forme du modèle, pas par manque d'entraînement |
| Les poids sont une image | `w.reshape(28, 28)` est un gabarit lisible |
| `/255` est une décision | on l'a mesurée |
| Deux chiffres proches font plafonner | une droite ne suffit plus |

**La suite.** Il faut combiner plusieurs neurones, et les empiler en couches. C'est le
notebook 2 : on construit le réseau complet à la main, en NumPy.
