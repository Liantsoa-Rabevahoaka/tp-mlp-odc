# ===MD===
# Notebook 4 · On casse le MLP

Trois expériences courtes. Chacune révèle ce que l'accuracy de 97 % du notebook 3
cachait.

Savoir où un modèle échoue vaut plus que savoir qu'il réussit.
# ===CODE===
import pathlib
import sys

# On retrouve la racine du dépôt, qu'on lance ce notebook depuis notebooks/
# ou depuis notebooks/solutions/.
RACINE = next(p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]
              if (p / "src" / "mnist.py").exists())
sys.path.insert(0, str(RACINE / "src"))

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import mnist
import viz
from couleurs import applique_style

applique_style()
torch.manual_seed(0)
np.random.seed(0)

X_train, y_train, X_test, y_test = mnist.charge_mnist(aplati=True, normalise=True)


def construit_modele():
    return nn.Sequential(
        nn.Linear(784, 128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, 10),
    )


def accuracy(modele, X, y):
    modele.eval()
    with torch.no_grad():
        pred = modele(torch.from_numpy(X).float()).argmax(dim=1).numpy()
    return (pred == y).mean()


modele = construit_modele()
modele.load_state_dict(torch.load(RACINE / "data" / "mlp.pt"))
print(f"référence, accuracy sur le test : {accuracy(modele, X_test, y_test):.2%}")
# ===MD===
---
## Expérience 1 · Décaler les chiffres de 2 pixels

Le modèle n'a jamais vu ça à l'entraînement, mais un 7 décalé de 2 pixels reste un 7
pour un humain.
# ===CODE===
def decale(X, dx=2, dy=0):
    """Décale les images de dx pixels vers la droite et dy vers le bas."""
    images = X.reshape(-1, 28, 28)
    images = np.roll(images, shift=(dy, dx), axis=(1, 2))
    return images.reshape(-1, 784)


X_decale = decale(X_test, dx=2)
viz.affiche_chiffres(X_decale, y_test, n=10, titre="Les mêmes chiffres, décalés de 2 pixels")

for dx in [0, 1, 2, 3, 4]:
    print(f"décalage de {dx} px → accuracy {accuracy(modele, decale(X_test, dx), y_test):.2%}")
# ===MD===
### Ce qu'on vient de voir

L'accuracy s'effondre. Deux pixels.

Le poids `W₁[357, k]` est attaché au pixel 357, définitivement. Le trait qui l'activait
s'est déplacé, le neurone ne le voit plus.

Le MLP n'a pas appris des formes. Il a appris des positions.
# ===MD===
---
## Expérience 2 · Permuter tous les pixels

La plus parlante des trois. On tire une permutation aléatoire des 784 pixels, appliquée
au train et au test : les images deviennent illisibles pour un humain.

On réentraîne le modèle depuis zéro dessus. Prédis avant de lancer : l'accuracy
s'effondre-t-elle ?
# ===CODE===
permutation = np.random.permutation(784)
X_train_perm = X_train[:, permutation]
X_test_perm = X_test[:, permutation]

viz.affiche_chiffres(X_test_perm, y_test, n=10, titre="Les mêmes chiffres, pixels permutés")
# ===CODE===
from torch.utils.data import TensorDataset, DataLoader


def entraine_vite(X, y, epoques=3):
    modele = construit_modele()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long()),
        batch_size=64, shuffle=True,
    )
    critere, optimizer = nn.CrossEntropyLoss(), torch.optim.SGD(modele.parameters(), lr=0.1)
    for _ in range(epoques):
        modele.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            critere(modele(xb), yb).backward()
            optimizer.step()
    return modele


# 20 000 exemples suffisent : ce qui nous intéresse ici n'est pas le score absolu,
# c'est l'ÉCART entre les deux, et il est parlant bien avant la convergence.
modele_normal = entraine_vite(X_train[:20000], y_train[:20000])
modele_permute = entraine_vite(X_train_perm[:20000], y_train[:20000])

print(f"images normales : {accuracy(modele_normal, X_test, y_test):.2%}")
print(f"pixels permutés : {accuracy(modele_permute, X_test_perm, y_test):.2%}")
# ===MD===
### Le résultat

Le score est le même, à un poil près. Le modèle apprend aussi bien sur des images que
personne ne peut lire.

La raison : `X = X.reshape(-1, 784)` au notebook 1 a jeté l'information « ce pixel est
à côté de celui-là ». Pour le MLP, une image est un sac de nombres sans relation entre
eux, permuter ce sac ne lui enlève rien.

> **Le MLP n'a jamais su que c'était une image en 2D.**

Ce n'est pas un bug. C'est la définition d'une couche dense.
# ===MD===
---
## Expérience 3 · Un chiffre écrit par toi

MNIST est centré, normalisé, blanc sur noir. Ton écriture ne coche aucune de ces cases.

Fabrique une image 28×28 ci-dessous, en dessinant dans le tableau ou en chargeant un
PNG.
# ===CODE===
# Un « 7 » grossier, dessiné à la main dans un tableau numpy.
mon_chiffre = np.zeros((28, 28), dtype=np.float32)
mon_chiffre[6, 7:20] = 1.0                          # la barre du haut
for i, ligne in enumerate(range(7, 22)):
    mon_chiffre[ligne, 19 - int(i * 0.7)] = 1.0     # la diagonale

# Pour utiliser ton propre PNG à la place :
#   from PIL import Image
#   img = Image.open("mon_chiffre.png").convert("L").resize((28, 28))
#   mon_chiffre = 1.0 - np.array(img, dtype=np.float32) / 255.0   # blanc sur noir !

plt.figure(figsize=(3, 3))
plt.imshow(mon_chiffre, cmap="gray_r"); plt.axis("off"); plt.title("mon chiffre"); plt.show()

with torch.no_grad():
    z = modele(torch.from_numpy(mon_chiffre.reshape(1, 784)).float())
    probas = torch.softmax(z, dim=1).numpy()[0]

print("prédiction :", probas.argmax(), f"(confiance {probas.max():.1%})")
print("top 3      :", [(int(i), f"{probas[i]:.1%}") for i in probas.argsort()[::-1][:3]])
# ===MD===
### Décalage de distribution, en direct

Souvent, ça rate, et avec confiance : 90 % sur une mauvaise réponse. Softmax force les
probabilités à sommer à 1, il y aura toujours un gagnant, même quand tout est mauvais.

97 % sur MNIST ne dit rien sur ta propre écriture.

> ### ✏️ À toi de jouer
>
> Redessine ton chiffre décalé vers un coin, ou plus petit. La confiance reste-t-elle
> élevée ?
# ===MD===
---
# Et maintenant ? La réponse s'appelle la convolution

| Le problème | La réponse |
|---|---|
| Positions apprises, pas des formes | Filtres locaux, identiques où qu'ils regardent |
| 100 352 poids pour la première couche | Poids partagés : un filtre 3×3, 9 paramètres réutilisés |
| Un décalage de 2 px casse tout | Pooling : on résume chaque zone, tolérant aux déplacements |

Ensemble, ces idées forment le CNN : même budget de calcul, plus de 99 % sur MNIST, et
il survit au décalage.

Pas au programme aujourd'hui, volontairement. Le notebook `04b_bonus_cnn.ipynb` est
fourni, à lire quand vous voulez.

## Et MNIST lui-même ?

Trop propre, trop petit, saturé : état de l'art à 99,8 % depuis des années. Bon terrain
d'apprentissage, mauvais indicateur de performance réelle.

Pour continuer, dans l'ordre de difficulté :

- **Fashion-MNIST** : même format, plus dur, ton code tourne dessus tel quel.
- **CIFAR-10** : 32×32 couleur. Le MLP décroche, le CNN devient indispensable.
- **Augmentation de données** : décalages et rotations ajoutés à l'entraînement.
- **Transfer learning** : repartir d'un réseau déjà entraîné, comme 90 % des projets réels.

---

# À retenir · la journée entière

| | |
|---|---|
| **Un neurone trace une frontière.** | `w` l'oriente, `b` la place. |
| **Une seule frontière ne suffit pas.** | XOR : il faut combiner. |
| **La non-linéarité fait la profondeur.** | Sans elle, N couches = 1 couche. |
| **Apprendre = ajuster θ pour faire baisser L.** | Forward, loss, backward, update. |
| **Il n'y a pas de magie.** | Tu as écrit la backprop à la main. PyTorch fait ça. |
| **Un modèle a une zone de compétence.** | Hors distribution, il se trompe avec confiance. |
| **Regarde toujours ce que ton modèle rate.** | L'accuracy seule ne dit jamais pourquoi. |

**Merci !**
