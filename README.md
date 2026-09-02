# Formation deep learning (MNIST)

Ce TP vous permettra de gagner en intuition sur le perceptron et de comprendre les
décisions derrière les choix courants : la cross-entropy au lieu de la MSE, le `/255`,
l'aplatissement en 784, le softmax en sortie.

Formation d'une journée pour des développeurs à l'aise en Python mais rouillés en maths.

## Deux formats

### Format court, 1h15 (par défaut)

Un seul notebook, **`00_tp.ipynb`**, qui reprend l'essentiel des quatre autres.

| Horaire | Contenu |
|---|---|
| 0-5 min | `verif_env.py`, première image affichée |
| 5-70 min | `00_tp.ipynb` |
| 70-75 min | Questions, où continuer |

Le notebook contient 4 exercices (une dizaine de lignes à écrire), dont 2 en PyTorch, plus
une section « Bonus » de cellules prêtes à lancer pour ceux qui finissent en avance.

### Format long, 3h

Les quatre notebooks détaillés, **dans l'ordre**. C'est une progression : chacun reprend
là où le précédent s'est arrêté, et le notebook 3 traduit ligne à ligne le code écrit au
notebook 2.

| Horaire | Contenu | Durée |
|---|---|---|
| 14h00 | Installation, `verif_env.py`, première image affichée | 15 min |
| 14h15 | **NB1 · Le perceptron à la main** (XOR, puis MNIST 0 contre 1) | 40 min |
| 14h55 | **NB2 · Le MLP à la main** (forward et backward en NumPy) | 50 min |
| 15h45 | *Pause* | 15 min |
| 16h00 | **NB3 · Le même MLP en PyTorch, atelier η** | 45 min |
| 16h45 | **NB4 · On casse le MLP** (projeté par le formateur) | 15 min |
| 17h00 | Fin | |

> Si le temps manque sur ce format, ne sautez pas le notebook 2 : le notebook 3 s'appuie
> dessus. Basculez plutôt sur `00_tp.ipynb`, qui est le condensé autonome.

## Les notebooks

- **`00_tp.ipynb`** · *le format court, autonome.* Perceptron, XOR, MNIST, réseau PyTorch
  784 → 128 → 64 → 10, taux d'apprentissage, matrice de confusion, puis les limites du MLP.
  C'est le résumé des quatre notebooks ci-dessous, et il ne dépend d'aucun d'eux.

Le format long déroule la même histoire en quatre étapes enchaînées :

- **`01_perceptron.ipynb`** · *pratique, 5 lignes à écrire.* Un perceptron en NumPy.
  D'abord sur XOR, où il ne converge pas. Puis sur MNIST 0 contre 1, où il dépasse 99 %.
  Les poids appris, affichés en image, forment un gabarit lisible du chiffre.
- **`02_mlp_a_la_main.ipynb`** · *le cœur du parcours.* Un réseau complet en NumPy, forward
  et backward écrits à la main, 784 → 128 → 64 → 10 (109 386 paramètres assertés), plus de
  95 %, avec le gradient vérifié numériquement. Il prouve qu'il n'y a pas de magie, et il
  fournit le code que le notebook 3 va traduire.
- **`03_mlp_pytorch.ipynb`** · *le même réseau, en PyTorch.* S'ouvre sur une table de
  correspondance entre le code NumPy du notebook 2 et son équivalent PyTorch, puis
  dépasse 97 %. Contient l'atelier sur le taux d'apprentissage, la comparaison SGD/Adam,
  la matrice de confusion et les images ratées.
- **`04_limites.ipynb`** · *démo.* On déconstruit le MLP (décalage de 2 pixels, permutation des
  pixels, un chiffre écrit à la main), puis on montre pourquoi le CNN répond exactement à
  ces trois limites. On ne code pas de CNN, on motive sa nécessité.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Si vous n'utilisez pas de GPU, installez la version CPU de PyTorch :

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Puis vérifiez que tout fonctionne :

```bash
python verif_env.py
jupyter lab
```

## téléchargement de MNIST

```bash
python src/mnist.py
```

Cela télécharge et met en cache `data/mnist.npz` (~11 Mo).

## Structure du dépôt

```
notebooks/               # notebooks étudiants (sorties vidées)
  _sources/              # SOURCE UNIQUE des notebooks, c'est ici qu'on édite
  solutions/             # notebooks formateur, exécutés (figures de référence)
src/                     # mnist.py, viz.py, couleurs.py, plomberie partagée
data/                    # cache MNIST (.npz), rempli au 1er run ou depuis la clé USB
build_notebooks.py       # régénère notebooks/ et notebooks/solutions/ depuis _sources/
verifie_todos.py         # contrôle que les notebooks étudiants s'arrêtent proprement
verif_env.py             # à lancer à 14h00
ANTISECHE.md             # le pont slide ↔ ligne de code, à imprimer
claude-plan-v2.md        # le plan pédagogique détaillé
```
## Glossaire FR/EN

| Anglais | Français |
|---|---|
| weight | poids |
| bias | biais |
| layer | couche |
| hidden layer | couche cachée |
| loss (function) | fonction de coût |
| gradient | gradient |
| learning rate (η) | taux d'apprentissage |
| epoch | époque |
| batch | lot |
| forward pass | propagation avant |
| backward pass | rétropropagation |
| overfitting | sur-apprentissage |
| accuracy | taux de bonne classification |
| features | caractéristiques |
| training set | jeu d'entraînement |
| validation set | jeu de validation |
| test set | jeu de test |
| one-hot encoding | encodage one-hot |
| flatten | aplatir |
