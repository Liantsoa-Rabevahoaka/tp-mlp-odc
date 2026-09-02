#!/usr/bin/env python3
"""Construit les notebooks à partir des sources texte de notebooks/src/.

Une seule source par notebook produit deux fichiers :
  - notebooks/NN_nom.ipynb            (version étudiant : les blocs solution
                                        deviennent des TODO)
  - notebooks/solutions/NN_nom_solution.ipynb  (version formateur, complète)

Format source
-------------
    # ===MD===        ouvre une cellule markdown
    # ===CODE===      ouvre une cellule de code
    # ===SOL=== indice        dans une cellule de code : début du bloc à masquer
    # ===ENDSOL===            fin du bloc

Usage : python build_notebooks.py
"""
import json
import pathlib
import re

RACINE = pathlib.Path(__file__).parent
SOURCES = RACINE / "notebooks" / "_sources"
SORTIE_ETU = RACINE / "notebooks"
SORTIE_SOL = RACINE / "notebooks" / "solutions"


def decoupe_cellules(texte):
    """Découpe le fichier source en (type, lignes)."""
    cellules, courant, typ = [], [], None
    for ligne in texte.splitlines():
        if ligne.strip() == "# ===MD===" or ligne.strip() == "# ===CODE===":
            if typ is not None:
                cellules.append((typ, courant))
            typ = "markdown" if "MD" in ligne else "code"
            courant = []
        else:
            courant.append(ligne)
    if typ is not None:
        cellules.append((typ, courant))
    return cellules


def resout_solutions(lignes, mode):
    """Traite les blocs ===SOL===/===ENDSOL=== selon le mode voulu."""
    sortie, dans_bloc, indice, corps = [], False, "", []
    for ligne in lignes:
        debut = re.match(r"^(\s*)# ===SOL===\s*(.*)$", ligne)
        if debut:
            dans_bloc, indice, corps = True, debut.group(2).strip(), []
            continue
        if re.match(r"^\s*# ===ENDSOL===\s*$", ligne):
            dans_bloc = False
            if mode == "solution":
                sortie.extend(corps)
            else:
                # On garde l'indentation de la première ligne non vide du bloc.
                marge = ""
                for c in corps:
                    if c.strip():
                        marge = c[: len(c) - len(c.lstrip())]
                        break
                sortie.append(f"{marge}# ✏️ À TOI DE JOUER — {indice}")
                sortie.append(f'{marge}raise NotImplementedError("À compléter")')
            continue
        (corps if dans_bloc else sortie).append(ligne)
    if dans_bloc:
        raise SyntaxError("bloc ===SOL=== non fermé")
    return sortie


def nettoie(lignes):
    """Retire les lignes vides en tête et en queue de cellule."""
    while lignes and not lignes[0].strip():
        lignes.pop(0)
    while lignes and not lignes[-1].strip():
        lignes.pop()
    return lignes


def construit(source, mode):
    cellules = []
    for typ, lignes in decoupe_cellules(source.read_text(encoding="utf-8")):
        if typ == "code":
            lignes = resout_solutions(lignes, mode)
        lignes = nettoie(lignes)
        if not lignes:
            continue
        cellule = {
            "cell_type": typ,
            "metadata": {},
            "source": [l + "\n" for l in lignes[:-1]] + [lignes[-1]],
        }
        if typ == "code":
            cellule["outputs"] = []
            cellule["execution_count"] = None
        cellules.append(cellule)
    return {
        "cells": cellules,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    SORTIE_SOL.mkdir(parents=True, exist_ok=True)
    for source in sorted(SOURCES.glob("*.py")):
        nom = source.stem
        for mode, dossier, suffixe in (
            ("etudiant", SORTIE_ETU, ""),
            ("solution", SORTIE_SOL, "_solution"),
        ):
            cible = dossier / f"{nom}{suffixe}.ipynb"
            cible.write_text(
                json.dumps(construit(source, mode), ensure_ascii=False, indent=1),
                encoding="utf-8",
            )
            print(f"  {cible.relative_to(RACINE)}")


if __name__ == "__main__":
    main()
