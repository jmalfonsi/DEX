# DEX

**Modèle de décision embarqué à raffinement conservatif.** Conception, réseau
PyTorch entraînable, export ONNX, laboratoire navigateur et tests, créés le
29 septembre 2026. Les poids fournis sont aléatoires : **aucune compréhension
bilingue, calibration métier ou supériorité sur Jev n'est encore démontrée**.

L'hypothèse : concevoir un modèle dont les lectures fines de contexte et d'options
ont une influence bornée. DEX peut alors interrompre leur calcul tout en encadrant
la probabilité du modèle complet. Toutes les options restent dans la normalisation.
Une sortie typée peut être une catégorie, un booléen ou un niveau ordinal ; aucune
génération de texte n'est nécessaire.

Cette piste se distingue du simple assemblage encodeur/cache/classifieur déjà
étudié dans CALIBRE. Les composants connus et les limites de la revendication
d'originalité sont recensés explicitement.

## Lire

- [Architecture et équations](docs/ARCHITECTURE.md) — mécanisme, cinq axes,
  complexité, mémoire et risques.
- [Recherche et antériorités](docs/RECHERCHE.md) — sources primaires, Jev,
  alternatives et contribution candidate.
- [Mesures et état réel](docs/MESURES.md) — résultats du code et du navigateur,
  sans les confondre avec la justesse.
- [Évaluation](docs/EVALUATION.md) — protocole indépendant pour confirmer ou
  rejeter la supériorité recherchée.
- [Entraînement futur](docs/ENTRAINEMENT.md) et [recette](training/recipe.json) —
  curriculum, données, calibration et préparation Modal ; aucun job lancé.
- [Contrat d'interface](docs/CONTRAT.md) — API cible et fonctions réellement disponibles.

## Exécuter

```bash
cd /home/ubuntu/DEX
python3 -m pip install -e '.[test,export]'
python3 -m pytest -q
node --test tests/test_web.mjs
python3 scripts/train_smoke.py
OPENBLAS_NUM_THREADS=1 python3 scripts/benchmark.py
```

Utiliser de préférence un environnement virtuel. Le test d'entraînement
surapprend un seul lot : c'est une vérification technique, pas un benchmark.

Les graphes et le runtime ont déjà été préparés dans ce workspace. Pour les
reproduire dans un autre environnement :

```bash
python3 scripts/export_onnx.py
python3 scripts/prepare_browser.py
python3 scripts/audit_export.py
```

La préparation télécharge ONNX Runtime Web **1.30.0** depuis sa distribution npm,
ou accepte `--runtime-source` pour réutiliser une copie locale. Les versions
effectivement utilisées pour les mesures sont dans les manifestes d'`artifacts/`.
L'export stocke les poids en int8 et garde les activations en FP32 ; il ne faut
pas interpréter cela comme une preuve de calcul int8 accéléré.

Ouvrir le laboratoire :

```bash
python3 -m http.server 8097 --bind 127.0.0.1 --directory /home/ubuntu/DEX
```

Puis visiter **http://127.0.0.1:8097/web/** depuis le même PC. Le premier banc
vérifie les bornes sur vecteurs synthétiques ; le second exécute le réseau
ONNX non entraîné. Les fichiers restent servis localement et aucun modèle
hébergé n'est appelé.

Le script `scripts/verify_browser.py --chromium /chemin/vers/chrome` automatise
la vérification via CDP avec `websocket-client`. Il démarre puis ferme son
serveur local et son navigateur. La mesure intégrée désactive le GPU : elle
valide WASM, pas WebGPU.

## État du travail

Disponible : fonction neuronale complète de référence, gradients, bornes testées,
vues typées, utilitaires de calibration, export et mesure locale du navigateur.

À réaliser : tokenizer et frontend intégrés, poids linguistiques entraînés,
cache de production, ordonnanceur ONNX adaptatif par lots, WebGPU, long contexte
réel, jeu or indépendant et campagne comparative Jev. Les cibles 128k tokens,
10k options sémantiques et 64 champs ne sont pas présentées comme déjà validées.
