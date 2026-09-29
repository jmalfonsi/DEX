# DEX — protocole pour décider s'il faut poursuivre

Version 0.1. Aucun score sémantique DEX contre Jev n'a encore été mesuré.
Les tests de code ne sont pas des évaluations d'intelligence.

## 1. Question expérimentale

À qualité sémantique et calibration comparables, DEX apporte-t-il une meilleure
frontière entre contexte, latence locale, mémoire, options et nombre de champs ?
La réussite d'un seul axe ne compense pas silencieusement l'échec des autres.

Trois évaluations restent séparées :

1. **Correction du calcul** : la voie adaptative approxime-t-elle effectivement
   la distribution du même réseau en calcul complet ?
2. **Qualité de la distribution** : cette distribution prédit-elle les réponses
   annotées avec une calibration utilisable ?
3. **Qualité du système** : après quantification, arrêt anticipé et abstention,
   que gagnons-nous sur une machine précise ?

## 2. Baselines nécessaires

| Référence | Pourquoi elle est nécessaire |
|---|---|
| Jev `jev-1.13.0` | Concurrent visé ; version figée, mêmes états/rubriques et horodatage des appels |
| GLiClass de taille comparable | Vérifier que le gain ne vient pas seulement d'un petit encodeur spécialisé |
| Encodeur bilingue + classifieur simple | Vérifier l'utilité de la nouvelle architecture sur tâches réellement simples |
| Lecture conditionnée par chaque question | Mesurer la perte de qualité de la lecture initiale partagée |
| CALIBRE, si checkpoint entraîné disponible | Référence interne à cache, routage et banques d'options |
| DEX dense | Référence exacte de DEX pour juger l'accélération |
| DEX sans résidus / résidus non bornés | Distinguer la capacité supplémentaire de l'intérêt des bornes |

La comparaison de noms commerciaux ne suffit pas : versions, paramètres,
quantification, tokenizer et protocole doivent être enregistrés. Aucun modèle
non entraîné ne participe à la comparaison de justesse.

Au-delà de 255 options ou de la fenêtre native de Jev, séparer deux résultats :
extension native de l'interface ; qualité d'un système Jev avec découpage ou
hiérarchie explicitement documentée. Compter alors tous ses appels et ses erreurs
de routage. Ne pas attribuer une justesse nulle à Jev simplement parce qu'une
requête hors limites a été refusée.

## 3. Données et splits

Premier jeu indépendant proposé : **6 000 situations de base**, au minimum
3 000 rédigées nativement en français et 3 000 en anglais, avec versions
parallèles sur un sous-ensemble. Les traductions d'une même situation restent
dans le même split. Plusieurs questions sur un même état restent ensemble.

- 2 000 situations réservées à la calibration et à la sélection des seuils.
- 4 000 réservées au test final, figé avant la sélection du checkpoint.
- Les données d'entraînement viennent d'autres situations, entités, gabarits et
  familles de schémas. Les familles hors domaine sont signalées séparément.

Les nombres sont un plan d'échantillonnage, pas une annonce de données disponibles.
Certaines intersections langue/type/domaine peuvent rester trop petites ; elles
doivent alors porter un intervalle large plutôt qu'un chiffre flatteur.

Double annotation et adjudication sur les désaccords. Conserver une distribution
d'annotations pour les questions réellement subjectives, avec un nombre
d'annotateurs explicite. Une majorité de trois annotateurs n'est pas une probabilité
parfaitement connue. Les cas à réponse vérifiable ont une étiquette indépendante
du professeur utilisé pour l'entraînement.

Compléments publics : MASSIVE pour intentions EN/FR ; XNLI pour entailment,
contradiction et indétermination ; MultiEURLEX pour les taxonomies multilabel ;
tâches inspirées de RULER pour contexte long. Les liens sont dans
[RECHERCHE.md](RECHERCHE.md). Respecter les structures d'étiquettes : un document
multilabel n'est pas converti arbitrairement en `Choice` mono-étiquette.

## 4. Jeux difficiles à construire

| Famille | Exemples ou transformations |
|---|---|
| Négation | « Je ne veux pas annuler » ; « ce n'est pas impossible » ; portée de « sauf » |
| Révision temporelle | Ancien accord puis retrait ; plus récente mention citée mais non endossée |
| Rôles | Demande client, réponse agent et contenu d'un document qui ne doivent pas être confondus |
| Références | Une condition dans un fragment, son exception dans un autre ; anaphore et entités proches |
| Information absente | Aucune option applicable ; faits manquants ; contradiction non résolue |
| Options | Rubriques similaires, synonymes, négations, labels sans signification, longues exceptions |
| Ajouts | Distracteurs pertinents lexicalement ; contre-preuve tardive ; duplication d'une mention |
| Français réel | Sans accents, élisions, politesse implicite, termes métiers, nombres et dates locales |
| Mélange de langues | Question anglaise/état français et inversement ; alternance au sein du même tour |
| Injection | Texte d'état qui tente de remplacer la question ou d'imposer un identifiant |
| Structure | JSON à clés permutées ; ordre des tableaux ; booléen `false` face à chaîne `"false"` |

Les agrégations et l'absence d'un élément demandent des tests couvrant l'état
entier. Ne pas prétendre traiter « aucun fragment ne mentionne X » en ne testant
qu'une récupération de passage évident.

## 5. Mesures

### Distribution et décision

Justesse, macro-F1 si utile, NLL, Brier avec convention de somme sur les classes,
ECE par intervalles de largeur égale **et** fiabilité par effectifs égaux.
Publier les effectifs et diagrammes, pas uniquement une ECE agrégée. Les probabilités
nulles sont bornées par un epsilon déclaré pour le calcul numérique de NLL.

Tracer risque-couverture pour chaque langue et type. Fixer les seuils sur la
calibration uniquement, puis mesurer leur résultat sur le test final. Comparer
les systèmes au même niveau de couverture ; un système qui s'abstient partout
n'a pas gagné.

Objectif initial : NLL DEX non inférieure à Jev avec une marge relative préfixée
de 5 %, justesse non inférieure de plus d'un point absolu, et meilleure qualité
FR lorsque l'écart est statistiquement identifiable. Cible d'usage sélectif :
risque ≤ 2 % à couverture ≥ 70 %, EN et FR séparément. ECE cible ≤ 0,03.
Ces valeurs sont des seuils d'acceptation proposés, pas des résultats.

Les intervalles à 95 % sont obtenus par bootstrap **par situation**, en gardant
ensemble questions et traductions. Pour le risque sélectif, publier aussi nombre
de décisions retenues et intervalle binomial. Préenregistrer les métriques
primaires par axe et corriger les comparaisons multiples ; ne pas choisir après
coup la métrique sur laquelle DEX gagne.

### Propriétés structurelles

- Masse totale et compléments des vues déclarées : erreur absolue mesurée.
- Permutation d'options et ajout/retrait de champs indépendants : écart maximal
  sur les probabilités, nombre de changements d'argmax hors égalités.
- Équivalences formulées librement : métrique empirique distincte, sans garantie.
- DEX adaptatif contre dense : violations de bornes, erreur de probabilité,
  changements d'argmax et proportion de budgets épuisés.
- Évaluer aussi les erreurs des **non-gagnants** et les masses d'événements ;
  la tolérance du prototype porte seulement sur la probabilité sélectionnée.

### Coût

Contextes : 512, 2k, 8k, 32k, 128k tokens. Options : 2, 32, 255, 1k, 10k,
puis 100k si la mémoire le permet. Champs : 1, 16, 64, 256. Commencer par les
coins de cette grille et les charges applicatives, puis remplir les transitions.

Rapporter p50/p95, débit, pic de mémoire, taille transférée et coût de compilation
du schéma. Au moins 100 répétitions après échauffement pour un p95 initial ;
répéter sur plusieurs sessions pour capturer compilation et variabilité.

Séparer : téléchargement froid ; initialisation après cache disque ; tokenisation ;
état neuf ; état déjà encodé ; ajout d'un tour ; modification au milieu ; options
dynamiques ; sérialisation de toutes les probabilités. Désactiver les requêtes
réseau vers des modèles pendant le test local.

Machines à figer : portable Intel avec iGPU courant, portable AMD avec iGPU,
et une machine limitée au CPU/WASM. Enregistrer modèle exact, RAM, OS, navigateur,
version, alimentation et chauffe. Les mesures du Xeon de développement ne sont
pas extrapolées à ces PC.

## 6. Ablations qui décident de l'architecture

| Test | Décision entraînée par le résultat |
|---|---|
| Dense borné vs dense non borné | Si perte de qualité > 1 point ou NLL > 5 %, revoir les bornes/capacité avant d'optimiser |
| Raffinement vs même réseau dense | Exiger gain de latence sur charge cible, incluant les transferts et le planificateur |
| Résumé seul vs lecture fine | Vérifier l'utilité réelle de la correction et sa récupération des négations |
| Rang 64/128/256 | Quantifier la perte due à la représentation compacte |
| Option à un vecteur vs quatre facettes | Préserver longues rubriques et exceptions |
| Poids de routage diffus vs plus concentrés | Vérifier le gain sans oublier la contre-preuve |
| λ et ρ : 0,25/0,5/1/2/4 | Courbe capacité contre coût de fermeture des bornes |
| FP32, W8, W4 et QAT | Comparer justesse et calibration après chaque export |
| Données EN seules vs mélange natif équilibré | Démontrer l'apport français, au lieu de seulement l'affirmer |
| Arrêt selon argmax seul vs probabilité aussi | Mesurer l'importance de conserver une probabilité exploitable |

## 7. Portes de continuation

G0 : propriétés numériques, gradients et export navigateur vérifiés.

G1 : sur un premier domaine indépendant, le modèle dense conserve la qualité des
baselines. Si G1 échoue, ne pas engager un grand préentraînement.

G2 : quantification et politique de calcul passent les portes de calibration et
de risque-couverture ; poids et manifestes gelés.

G3 : sur vrais PC, accélération adaptative mesurée contre DEX dense ; aucune
violation détectée des intervalles ; amélioration de la frontière latence/qualité.

G4 : extension aux contextes longs, grandes banques, beaucoup de champs et
généralisation EN/FR. Une revendication « dépasse Jev sur cinq axes » n'est
autorisée par les résultats que lorsque chaque axe possède sa preuve séparée.
