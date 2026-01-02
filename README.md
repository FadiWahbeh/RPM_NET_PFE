# Projet RPM / DGCNN

Ce dépôt contient le code d'entraînement et les modèles pour le projet (fichiers principaux comme `train.py`, `data/`, `models/`, etc.).

Prérequis
- Python 3.8+
- PyTorch

Instructions pour versionner et pousser sur GitHub et Etulab

1) Initialiser le dépôt local et faire le commit initial

```bash
cd path/to/project   # ex: c:\Users\fadiw\PFE3
git init
git checkout -b main
git add .
git commit -m "Initial commit"
```

2) Créer le dépôt GitHub
- Option A (interface web): créer un nouveau dépôt sur https://github.com et copier l'URL remote.
- Option B (gh CLI): `gh repo create <owner>/<repo> --public --source=. --remote=origin` 

3) Pousser vers GitHub

```bash
git remote add origin <GITHUB_REPO_URL>
git push -u origin main
```

4) Ajouter Etulab (remote fournie par votre établissement)

```bash
git remote add etulab <ETULAB_REPO_URL>
git push -u etulab main
```

Remarques
- Remplacez `<GITHUB_REPO_URL>` et `<ETULAB_REPO_URL>` par les URLs réelles.
- Si vos données sont volumineuses, utilisez Git Large File Storage (`git lfs`) ou hébergez les jeux de données séparément.

Fichiers créés
- `.gitignore` : pour exclure les fichiers temporaires, modèles et données volumineuses.

Besoin d'aide pour créer le dépôt GitHub ou configurer Etulab ? Indiquez-moi l'URL ou si vous voulez que je génère les commandes précises.
