# DEX — mesures du 29 septembre 2026

**Tout le réseau est initialisé aléatoirement.** Ces mesures portent sur le calcul,
les invariants et le runtime. Elles ne disent rien de sa justesse sémantique,
de son bilinguisme effectif ou de sa calibration métier.

## 1. Environnement

Intel Xeon D-1521 à 2,40 GHz, Linux x86-64 ; PyTorch 2.11.0+cpu ; NumPy 2.4.2 ;
ONNX 1.23.0 ; ONNX Runtime natif et Web 1.30.0 ; Chromium headless 153.
Le navigateur fonctionne sur ce serveur, **GPU désactivé**, WASM à un thread.
Il ne s'agit pas d'une mesure sur un portable grand public ou sur WebGPU.

Les cinq observations par phase servent à valider un ordre de grandeur. Elles
ne permettent pas un p95 fiable. Le serveur n'est pas une machine de benchmark
isolée ; les mesures finales devront suivre le protocole dédié.

## 2. Réseau et export

| Mesure | Résultat |
|---|---:|
| Nombre de paramètres | 28 396 162 |
| Poids FP32 théoriques | 108,32 Mio |
| Encodeur ONNX à poids int8 | 28 234 224 octets |
| Noyau de décision ONNX FP32 | 1 340 565 octets |
| Total des graphes | **29 574 789 octets**, soit 29,57 Mo décimaux |
| Runtime Web livré | 14 337 332 octets |
| Modèle + runtime | 43 912 121 octets, hors tokenizer et autres fichiers |
| Écart maximal noyau ONNX natif / PyTorch | 1,67 × 10⁻⁶ sur la fixture |
| Écart maximal encodeur quantifié / PyTorch FP32 | 0,0298 sur les représentations de la fixture |

Le dernier écart n'est pas une perte de justesse : il n'y a aucune étiquette
sémantique dans cet essai. Il impose une évaluation après quantification lorsque
les poids seront entraînés.

Référence retenue : **poids stockés en int8, activations FP32, graph optimization
`basic` dans les deux moteurs**. Les poids peuvent être déquantifiés en mémoire.
La taille téléchargée ne représente donc ni la RAM résidente ni une preuve de
produits matriciels accélérés en int8.

Source reproductible : [model-manifest.json](../artifacts/model-manifest.json).

## 3. Exécution réelle dans Chromium

Charge : 512 IDs de tokens synthétiques, 4 questions de 32 tokens chacune,
32 options de 32 tokens. Encodage des questions/options mis à part comme compilation
du schéma. Les sorties sont des distributions aléatoires, pas des réponses utiles.

| Phase | Mesure |
|---|---:|
| Initialisation des deux sessions | 752 ms |
| Compilation du schéma (questions + options) | 2 329 ms |
| Lecture de l'état de 512 tokens, médiane de 5 mesures | **1 020 ms** |
| Décision dense, état et schéma déjà encodés, médiane | **2,9 ms** |
| Écart maximal logits navigateur / référence ONNX native | **9,76 × 10⁻⁷** |
| Vérification numérique au seuil 10⁻⁴ | Réussie |

La lecture de l'état domine nettement le temps d'un état neuf. Annoncer « DEX
répond en 3 ms » serait trompeur : ce chiffre ne couvre que les décisions sur
représentations déjà en mémoire. Tokenisation, calibration métier, ordonnanceur
adaptatif, téléchargement distant et pic mémoire long contexte ne sont pas inclus.

Le seuil exploratoire de 800 ms WASM pour état neuf n'est pas atteint sur ce
serveur avec cette référence. Les cibles sur portables restent à mesurer. Une
campagne WebGPU et une quantification d'activations contrôlée seront nécessaires
avant toute conclusion de supériorité de vitesse.

Le noyau de bornes JavaScript a également été exécuté dans Chromium sur 1 024
options et 64 fragments synthétiques : argmax fixé et probabilité complète contenue
dans l'intervalle. Sa durée ponctuelle, environ 33 ms, mesure un planificateur
avec corrections déjà construites ; **ce n'est pas une latence d'inférence linguistique**.

Source : [browser-measurements.json](../artifacts/browser-measurements.json).

## 4. Résultat négatif utile : le raffinement peut coûter cher

La référence NumPy utilise 128 fragments, 10 000 options, rang 32, corrections
précalculées et tolérance absolue de probabilité 0,01. OpenBLAS à un thread.

| Fixture synthétique | Fragments raffinés | Options raffinées | Itérations | Temps du planificateur | Erreur sur p sélectionnée |
|---|---:|---:|---:|---:|---:|
| Distribution très concentrée | 64/128 | 0/10 000 | 1 | 8,1 ms | 1,14 × 10⁻⁶ |
| Distribution diffuse | 128/128 | 5 056/10 000 | 81 | **370 ms** | 8,66 × 10⁻⁴ |
| Faibles probabilités individuelles | 128/128 | 64/10 000 | 3 | 17,6 ms | 1,69 × 10⁻⁴ |

Les trois argmax correspondent au calcul complet. Le troisième cas rappelle
qu'une petite erreur **absolue** n'est pas une petite erreur relative quand la
probabilité vaut environ 0,0008. Pour exploiter de telles probabilités il faudra
un critère relatif ou une borne d'événement adapté.

Le deuxième cas invalide l'idée que l'adaptatif serait toujours plus rapide.
La référence recalcule trop souvent les produits et les bornes ; le runtime
doit prévoir lots plus larges, choix dense/adaptatif et repli dense. Aucun facteur
d'accélération neuronale n'est déduit de ces fixtures, puisque le coût des
corrections neuronales n'y est pas exécuté.

Sur PyTorch FP32 à quatre threads, l'encodeur de 512 tokens mesure environ
107 ms médians. La différence avec WASM montre pourquoi la vitesse d'un prototype
Python ne doit pas être annoncée comme celle du navigateur.

Source : [cpu-mechanisms.json](../artifacts/cpu-mechanisms.json).

## 5. Tests et audit

**15 tests Python et 2 suites Node passent.** Ils couvrent notamment :

- bornes comparées à tous les sommets de petites boîtes de logits ;
- intervalles des événements, scores extrêmes et classe unique ;
- validité de l'enveloppe à chaque budget de raffinement ;
- récupération d'une option qui n'était pas gagnante au premier passage ;
- conservation d'une faible probabilité dans une banque uniforme de 10 000 options ;
- indépendance des champs et équivariance aux permutations d'options en FP32 ;
- correspondance du calcul neuronal complet et des callbacks paresseux ;
- propagation des gradients jusqu'aux trois composants ;
- récupération d'une température connue sur données probabilistes synthétiques,
  puis évaluation sur échantillon distinct ;
- conservation des types JSON et versionnement des clés de cache.

Audit ONNX supplémentaire sur des dimensions différentes de l'export initial :
5 fragments de 20 tokens, 7 champs, 37 options. Écart maximal encodeur par lot
contre fragments séparés : 0 sur les tokens de cette fixture ; résumés :
1,19 × 10⁻⁷ ; champs seuls contre regroupés : 1,91 × 10⁻⁶ ; permutation des
options : 0. Aucun de ces zéros ponctuels n'est une garantie générale bit à bit.
[export-audit.json](../artifacts/export-audit.json).

Un entraînement CPU de 80 pas sur un unique lot fait baisser la perte de 1,65 à
environ 5 × 10⁻⁷. Il prouve que le réseau peut recevoir et exploiter un gradient ;
il ne prouve aucune généralisation. [training-smoke.json](../artifacts/training-smoke.json).

## 6. Investigation de la quantification

La première quantification dynamique U8S8 donnait un écart de logits d'environ
0,101 entre la référence native et WASM. L'option de plage réduite ramenait cet
écart à 0,0091, encore au-dessus du seuil choisi. La documentation ONNX explique
un risque de saturation U8S8 sur certains noyaux AVX2 et recommande notamment
`reduce_range`. Ce mécanisme explique une piste de correction, pas à lui seul
la totalité des écarts observés.
[Documentation de quantification](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html).

Après passage à des poids int8 avec activations FP32, il restait un écart de
logits d'environ 0,0034. L'instrumentation a localisé l'écart dans l'encodeur :
le noyau de décision avec entrées identiques concordait à moins de 10⁻⁶.
Dans le moteur natif, les optimisations `extended/all` réécrivaient entre autres
des opérations en `MatMulNBits` et `QuickGelu`. Les niveaux `disabled/basic`
produisaient une autre sortie ; `basic` a rétabli la concordance avec WASM.
La responsabilité individuelle de chaque réécriture n'a pas été isolée.

Le choix final privilégie une référence mathématique commune. Une optimisation
plus rapide pourra être réintroduite avec son propre audit, sur poids entraînés.
Le [rapport de l'export dynamique rejeté](../artifacts/browser-dynamic-int8-failed-parity.json)
reste disponible ; il n'est pas le résultat courant.

## 7. Ce qui n'a pas été mesuré

Qualité EN/FR, calibration réelle, comparaison directe avec Jev, performances
WebGPU, 128k tokens en bout en bout, 10k descriptions d'options sémantiques,
64 champs réels et pic de RAM navigateur. Ces travaux forment la prochaine
campagne, pas une propriété déjà acquise de DEX v0.1.
