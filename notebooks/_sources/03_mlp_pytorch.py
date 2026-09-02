# ===MD===
# Notebook 3 · Le MLP en PyTorch

Au notebook 2, tu as écrit un réseau de neurones à la main. Ici on refait **exactement le
même** : même architecture, même compte de paramètres, mais avec PyTorch.

Le but n'est pas d'apprendre une API. C'est de voir que PyTorch ne fait **rien que tu ne
saches déjà faire** : il automatise le backward que tu as écrit, et c'est à peu près tout.

Au programme : traduire ton code ligne à ligne, mesurer l'effet du taux d'apprentissage, et
regarder où le modèle se trompe.
# ===CODE===
import pathlib
import sys

RACINE = next(p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]
              if (p / "src" / "mnist.py").exists())
sys.path.insert(0, str(RACINE / "src"))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import mnist
import viz
from couleurs import applique_style

applique_style()
torch.manual_seed(0)
np.random.seed(0)

print("PyTorch", torch.__version__, ": on reste sur CPU, c'est suffisant ici.")
# ===MD===
---
## 1 · La table de traduction

Rien de neuf, juste une correspondance avec ce que tu as écrit au notebook 2.

| Ton code NumPy | Ce que PyTorch écrit | Ce que ça change |
|---|---|---|
| `W.append(np.random.randn(...) * ...)` | `nn.Linear(784, 128)` | l'initialisation est faite pour toi |
| `relu(z)` | `nn.ReLU()` | rien |
| `X @ W[0] + b[0]` | `couche(x)` | rien |
| `softmax(...)` puis `cross_entropy(...)` | `nn.CrossEntropyLoss()` | **les deux d'un coup**, en plus stable |
| toute ta fonction `backward()` | `loss.backward()` | **c'est ça, l'apport de PyTorch** |
| `W[k] = W[k] - eta * dW[k]` | `optimizer.step()` | rien, pour SGD |
| rien d'équivalent | `optimizer.zero_grad()` | **nouveau**, on voit pourquoi plus bas |
| ton découpage manuel en lots | `DataLoader` | mélange, lots et performance |

La seule vraie nouveauté de la colonne du milieu, c'est `loss.backward()`. Les lignes de
dérivées que tu as écrites à la main, PyTorch les reconstruit tout seul en enregistrant les
opérations du forward.

## 2 · Les données

Toujours notre `charge_mnist`, pour que le prétraitement reste visible.

Le `Normalize((0.1307,), (0.3081,))` qu'on copie-colle dans tous les tutoriels, c'est le
`/255` du notebook 1 suivi d'un centrage. Autant le voir en clair.
# ===CODE===
X_train, y_train, X_test, y_test = mnist.charge_mnist(aplati=True, normalise=True)
X_val, y_val = X_train[:5000], y_train[:5000]
X_train, y_train = X_train[5000:], y_train[5000:]

viz.affiche_chiffres(X_train, y_train, n=10, titre="Ce qu'on donne au réseau")
print(X_train.shape, X_val.shape, X_test.shape)
# ===CODE===
def en_tenseurs(X, y):
    return TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())


train_loader = DataLoader(en_tenseurs(X_train, y_train), batch_size=64, shuffle=True)
val_loader = DataLoader(en_tenseurs(X_val, y_val), batch_size=512)
test_loader = DataLoader(en_tenseurs(X_test, y_test), batch_size=512)

# Pour comparer des réglages, un sous-ensemble suffit : on cherche la forme des courbes,
# pas le dernier dixième de pourcent.
train_rapide = DataLoader(en_tenseurs(X_train[:15000], y_train[:15000]),
                          batch_size=64, shuffle=True)

print(f"{len(train_loader)} lots de 64 par époque, {len(train_rapide)} pour les comparaisons")
# ===MD===
---
## 3 · Le réseau

La même architecture qu'au notebook 2 : **784 → 128 → 64 → 10**. `nn.Sequential` empile les
couches dans l'ordre, c'est ton forward pass écrit en déclaratif.

Un seul piège, et c'est le plus fréquent en PyTorch : **pas de softmax à la fin**. Au
notebook 2 tu appelais `softmax` puis `cross_entropy` ; ici `nn.CrossEntropyLoss` fait les
deux, de façon plus stable. Il attend donc les scores bruts `z` (les *logits*). Un softmax
ajouté avant lui fait apprendre le modèle de travers, sans message d'erreur.
# ===CODE===
def construit_modele():
    # ===SOL=== empile 784->128, ReLU, 128->64, ReLU, 64->10 (pas de softmax)
    return nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )
    # ===ENDSOL===


modele = construit_modele()
n_parametres = sum(p.numel() for p in modele.parameters())
print(modele)
print(f"\nparamètres : {n_parametres:,}".replace(",", " "))
assert n_parametres == 109_386
print("=> 109 386, le compte fait au tableau ce matin.")
# ===MD===
---
## 4 · La boucle d'entraînement

Les quatre étapes du matin, dans l'ordre : **forward**, **loss**, **backward**, **update**.

Deux lignes méritent un mot.

`optimizer.zero_grad()` : PyTorch **accumule** les gradients au lieu de les remplacer.
Sans cette ligne, le gradient du lot 2 s'ajoute à celui du lot 1, et le modèle part en
vrille sans rien signaler.

`model.train()` / `model.eval()` : certaines couches (dropout, batch-norm) se comportent
différemment à l'entraînement et à l'évaluation. Notre modèle n'en a pas encore, mais
c'est une source classique de bugs silencieux.
# ===CODE===
def evalue(modele, loader, critere):
    """Renvoie (loss moyenne, accuracy), sans toucher aux gradients."""
    modele.eval()
    perte, justes, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            z = modele(x)
            perte += critere(z, y).item() * len(y)
            justes += (z.argmax(dim=1) == y).sum().item()
            total += len(y)
    return perte / total, justes / total


def entraine(modele, train_loader, val_loader, eta=0.1, epoques=8,
             optimiseur="sgd", bavard=True):
    critere = nn.CrossEntropyLoss()
    if optimiseur == "sgd":
        optimizer = torch.optim.SGD(modele.parameters(), lr=eta)
    else:
        optimizer = torch.optim.Adam(modele.parameters(), lr=eta)

    historique = {"train": [], "validation": []}

    for epoque in range(epoques):
        modele.train()
        for x, y in train_loader:
            z = modele(x)                    # 1. forward
            loss = critere(z, y)             # 2. loss

            # ===SOL=== remise à zéro, backward, update
            optimizer.zero_grad()
            loss.backward()                  # 3. backward
            optimizer.step()                 # 4. update
            # ===ENDSOL===

        loss_train, _ = evalue(modele, train_loader, critere)
        loss_val, acc_val = evalue(modele, val_loader, critere)
        historique["train"].append(loss_train)
        historique["validation"].append(loss_val)
        if bavard:
            print(f"époque {epoque + 1:2d}/{epoques} : loss {loss_train:.4f}, "
                  f"accuracy validation {acc_val:.2%}")

    return historique
# ===MD===
`loss.backward()` est la seule vraie nouveauté de PyTorch. Il reconstruit tout seul les
dérivées de chaque couche, en enregistrant les opérations du forward. C'est exactement le
travail fait à la main au notebook 2.
# ===CODE===
modele = construit_modele()
%time historique = entraine(modele, train_loader, val_loader, eta=0.1, epoques=8)
# ===CODE===
critere = nn.CrossEntropyLoss()
_, accuracy = evalue(modele, test_loader, critere)
print(f"accuracy sur le jeu de test : {accuracy:.2%}")
assert accuracy > 0.97

viz.plot_courbes(historique, titre="Loss d'entraînement et de validation")
# ===MD===
Trois régimes possibles sur ce graphique :

| Ce qu'on voit | Nom | Ce qu'il faut faire |
|---|---|---|
| les deux courbes restent hautes | sous-apprentissage | modèle trop petit, ou pas assez d'époques |
| elles descendent ensemble | bon régime | rien, c'est notre cas |
| la validation remonte | sur-apprentissage | arrêter plus tôt, régulariser |

C'est pour repérer le troisième cas qu'on garde un jeu de validation.
# ===MD===
### Ce que la première couche a appris

Ses poids ont 784 composantes, donc ils se replient en images 28×28, comme au notebook 1.
# ===CODE===
W1 = modele[0].weight.detach().numpy()
viz.montre_poids(W1[:10].T, titre="10 des 128 neurones de la première couche")
# ===MD===
C'est moins lisible que le gabarit du perceptron, et c'est normal : aucun de ces neurones
ne détecte « un 3 ». Chacun repère un fragment, une boucle, une barre, un vide. Ce sont
les couches suivantes qui les combinent.
# ===MD===
---
# 5 · Atelier : le taux d'apprentissage

C'est l'expérience annoncée ce matin. La règle `w ← w - η · ∂L/∂w` était un dessin. On en
fait une mesure.

Chaque binôme prend **un** `eta`, lance la cellule, note ce qu'il voit. On superpose les
quatre courbes ensuite.

| `eta` | Ta prédiction | Ce que tu observes |
|---|---|---|
| `1e-5` | | |
| `1e-3` | | |
| `0.1`  | | |
| `10`   | | |

Remplis la colonne « prédiction » **avant** de lancer.
# ===CODE===
resultats = {}
for eta in [1e-5, 1e-3, 0.1, 10.0]:
    torch.manual_seed(0)                       # même point de départ pour tous
    h = entraine(construit_modele(), train_rapide, val_loader,
                 eta=eta, epoques=5, bavard=False)
    resultats[f"eta = {eta}"] = h["train"]
    fin = h["train"][-1]
    print(f"eta = {eta:<8} : {'diverge' if not np.isfinite(fin) else f'loss finale {fin:.4f}'}")
# ===CODE===
viz.plot_comparaison(resultats, titre="Effet du taux d'apprentissage")
# ===MD===
Repère la valeur **2.30** sur l'axe. C'est `log(10)`, la loss d'un modèle qui répond au
hasard entre 10 classes. Toute courbe qui y reste n'a rien appris.

| `eta` | Résultat | Interprétation |
|---|---|---|
| `1e-5` | collée à 2.30 | apprend, mais il faudrait des milliers d'époques |
| `1e-3` | décolle à peine | même problème, en moins extrême |
| `0.1` | chute nette | notre réglage |
| `10` | au-dessus de 2.30 | pire que le hasard, les pas sautent par-dessus le minimum |

L'asymétrie est utile en pratique : un `eta` trop petit fait perdre du temps, un `eta`
trop grand empêche d'arriver, quel que soit le temps qu'on y passe.
# ===MD===
## 5 bis · Et l'optimiseur ?

Même modèle, même `eta = 0.01`. On ne change que l'optimiseur.
# ===CODE===
resultats_opt = {}
for nom in ["sgd", "adam"]:
    torch.manual_seed(0)
    h = entraine(construit_modele(), train_rapide, val_loader,
                 eta=0.01, epoques=3, optimiseur=nom, bavard=False)
    resultats_opt[nom.upper()] = h["train"]
    print(f"{nom.upper():5} : loss finale {h['train'][-1]:.4f}")

viz.plot_comparaison(resultats_opt, titre="SGD contre Adam, à eta identique")
# ===MD===
À `eta` rigoureusement identique, Adam descend environ dix fois plus bas. Il **adapte un
pas par paramètre** au lieu d'appliquer le même `eta` partout.

D'où son statut de choix par défaut : il pardonne un mauvais `eta`. Ce qui ne dispense pas
de le régler, mais laisse plus de marge.
# ===MD===
---
# 6 · Où le modèle se trompe

97 %, c'est un chiffre. Il ne dit pas **où** ça casse.

### D'abord, l'accuracy peut mentir

MNIST est équilibré, donc l'accuracy y est honnête. Fabriquons un jeu déséquilibré pour
voir le piège.
# ===CODE===
masque_1 = y_test == 1
y_desequilibre = np.concatenate([y_test[masque_1], y_test[~masque_1][:60]])

print(f"accuracy du modèle « je réponds toujours 1 » : {(y_desequilibre == 1).mean():.2%}")
print("=> 95 %, sans jamais regarder l'image.")
# ===MD===
### La matrice de confusion

Elle montre quelle classe est confondue avec laquelle. La diagonale, ce sont les bonnes
réponses.
# ===CODE===
modele.eval()
with torch.no_grad():
    y_pred = modele(torch.from_numpy(X_test).float()).argmax(dim=1).numpy()

matrice = viz.plot_confusion(y_test, y_pred)

erreurs = matrice.copy()
np.fill_diagonal(erreurs, 0)
for _ in range(3):
    vrai, pred = np.unravel_index(erreurs.argmax(), erreurs.shape)
    print(f"{erreurs[vrai, pred]:4d} fois : un {vrai} pris pour un {pred}")
    erreurs[vrai, pred] = 0
# ===MD===
Tu verras surtout des **4 ↔ 9**, **3 ↔ 5**, **7 ↔ 1**. Ce sont les paires qu'un humain
confond aussi : le modèle se trompe sur ce qui est réellement ambigu.
# ===CODE===
rates = y_pred != y_test
print(f"{rates.sum()} erreurs sur {len(y_test)} images")
viz.affiche_chiffres(X_test[rates], y_test[rates], y_pred[rates], n=15,
                     titre="Les images que le modèle rate")
# ===MD===
Regarde-les honnêtement : certaines sont illisibles. Toi non plus tu ne trancherais pas.

C'est là que vit l'**erreur irréductible**. Une partie des 3 % restants tient au bruit des
étiquettes, pas au modèle. Courir après 100 % sur MNIST, c'est courir après des images que
personne ne sait lire.
# ===MD===
> ✏️ **À toi** : choisis **une** expérience, lance-la, rapporte le résultat au groupe.
>
> 1. Couche cachée de **16**, puis **1024** neurones. Plus gros = mieux ? Regarde aussi le
>    nombre de paramètres : ×8 en taille, pour quel gain ?
> 2. `nn.Sigmoid()` à la place de `nn.ReLU()`. Que devient la courbe ?
> 3. `nn.Dropout(0.2)` après chaque ReLU. Effet sur l'écart train/validation ?
> 4. Données non normalisées (`normalise=False`). Comparer.
# ===CODE===
# ===SOL=== exemple : l'effet de la taille de la couche cachée
comparaison = {}
for taille in [16, 128, 1024]:
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(784, taille), nn.ReLU(),
                      nn.Linear(taille, 64), nn.ReLU(),
                      nn.Linear(64, 10))
    h = entraine(m, train_rapide, val_loader, eta=0.1, epoques=6, bavard=False)
    _, acc = evalue(m, test_loader, nn.CrossEntropyLoss())
    comparaison[f"{taille} neurones"] = h["validation"]
    print(f"{taille:5} neurones : {acc:.2%} de test, "
          f"{sum(p.numel() for p in m.parameters()):,} paramètres".replace(",", " "))

viz.plot_comparaison(comparaison, titre="Taille de la couche cachée")
# ===ENDSOL===
# ===CODE===
torch.save(modele.state_dict(), RACINE / "data" / "mlp.pt")
print("modèle sauvegardé dans data/mlp.pt (le notebook 4 va s'en servir)")
# ===MD===
---
# À retenir

| | |
|---|---|
| PyTorch n'automatise qu'une chose | le `backward()`. Le reste garde la même forme |
| `CrossEntropyLoss` mange des logits | pas de softmax avant. Erreur n°1 |
| `zero_grad()` n'est pas décoratif | PyTorch accumule les gradients |
| `eta` se mesure | trop petit : ça rampe. Trop grand : ça n'arrive jamais |
| Adam pardonne un mauvais `eta` | d'où son statut de choix par défaut |
| L'accuracy seule ment | matrice de confusion, et regarder les images ratées |
| Une part de l'erreur est irréductible | certaines images sont illisibles |

**La suite.** 97 %, c'est bien. Mais ce modèle a une faiblesse plus grave que les 3 % qui
manquent. → **notebook 4**.
