# DEX — trajectoire d'entraînement GPU

Ce document prépare une campagne future. Aucun entraînement linguistique ni
job Modal n'a été exécuté. Le seul entraînement livré est un test CPU qui
surapprend quatre réponses sur un unique lot, pour vérifier la chaîne de gradients.

## 1. Ce qu'il faut apprendre

La priorité est la **fonction dense de décision**. Une belle borne sur les
probabilités d'un mauvais modèle ne présente aucun intérêt.

Apprendre simultanément : compréhension bilingue locale ; extraction des critères
de la question ; représentation discriminante des options ; lecture des résumés ;
corrections fines ; distinction entre fait absent, négation et contradiction.
Le routage doit préserver les contre-preuves. Le français n'est pas un adaptateur
choisi par un détecteur de langue : les mêmes poids servent toutes les entrées.

## 2. Curriculum proposé

| Phase | Données et ordre de grandeur indicatif | Condition de passage |
|---|---|---|
| A — pilote | 50k à 200k décisions de familles variées, apprentissage dense ; comparer un encodeur initialisé et un étudiant compact | Dépasser les baselines simples sur des familles tenues à l'écart |
| B — transfert de représentation | Environ 200M à 1B tokens équilibrés EN/FR ; distillation depuis un encodeur bilingue plus capable, éventuellement objectif masqué temporaire | Compréhension et efficacité du tokenizer compatibles avec les cinq axes |
| C — décision | 1M à 5M décisions, banques et formulations renouvelées, tailles de contexte croissantes | Gain sur annotations indépendantes, pas seulement accord avec le professeur |
| D — raffinement | Mêmes décisions, budgets variables ; calcul dense conservé comme cible de référence | Réduction du coût sans perte au-delà des marges préfixées |
| E — export | QAT ou adaptation à l'erreur de quantification, mesures WASM/WebGPU, puis gel | Les portes de qualité survivent à l'export |
| F — calibration | Jeu indépendant, calibration sur sorties du runtime final, puis test gelé | Calibration et risque-couverture validés par langue et type |

Les volumes sont des enveloppes d'expérience. Ils ne sont pas une garantie qu'un
étudiant de 28 M paramètres atteindra Jev. Le pilote peut imposer plus de capacité
ou rejeter l'architecture avant les étapes coûteuses.

L'initialisation est une décision expérimentale : le code fourni part de poids
aléatoires. Un professeur candidat est [mmBERT-small](https://huggingface.co/jhu-clsp/mmBERT-small),
mais ses embeddings et ses couches ne se chargent pas automatiquement dans DEX.
Il faudra mapper ou distiller vers le vocabulaire et la géométrie retenus, avec
une baseline sans adaptation pour mesurer ce que cette compression perd.

## 3. Données

Mélange initial proposé au niveau des **situations**, pas seulement des tokens :
50 % français, 50 % anglais, avec lots mixtes supplémentaires. À l'intérieur du
français, au moins la moitié des situations doivent être rédigées en français
natif plutôt que traduites. Mesurer le biais de longueur et ajuster l'échantillonnage.

Les décisions couvrent routage, intent, entailment, comparaison à une rubrique,
reconnaissance d'information absente et évaluations ordinales. Le système apprend
à comparer des critères, pas à mémoriser une nomenclature unique.

Générer des paires contrefactuelles contrôlées : ajouter ou enlever une négation,
changer une exception, inverser une relation temporelle, remplacer une entité,
déplacer une preuve. Utiliser des contre-exemples où une modification minuscule
doit inverser le résultat, et des changements d'emballage qui ne doivent pas le
modifier. La distinction évite de rendre le modèle artificiellement invariant
à des informations importantes.

Les professeurs servent à proposer et annoter une partie des données, à fournir
des représentations ou des distributions de distillation. Leur accord n'est pas
une vérité de terrain et leur désaccord n'est pas une probabilité calibrée. Les
évaluations finales sont annotées indépendamment. Aucune sortie de Jev n'est
nécessaire à l'entraînement : la comparaison reste séparée.

Pour chaque corpus : source exacte, version, licence, droits d'utilisation,
hash, langue native/traduite, famille, entités et liens de parenté. Déduplication
et splits par famille avant génération des variantes. Les traductions,
paraphrases et fragments d'un document ne traversent jamais les splits.

## 4. Objectifs

Base : entropie croisée sur la distribution complète, ou cross-entropy vers une
distribution annotée lorsque les étiquettes ne sont pas déterministes. Le score
logarithmique est un objectif propre ; la précision top-1 seule ne l'est pas.

Auxiliaires à ablater : distillation douce ; alignement de paires EN/FR réellement
équivalentes ; supervision des fragments pertinents, incluant les contradictions ;
cohérence entre budgets. Les identités des vues déclarées sont calculées, elles
n'ont pas besoin d'être approximées par une pénalité.

Le prototype implémente le score principal et une KL de distillation optionnelle.
Les autres pertes ne sont pas encore implémentées. Éviter un terme qui récompense
simplement une petite entropie du routage : il pourrait obtenir une faible latence
en ignorant la preuve qui gêne. Un objectif de coût éventuel est ajouté après
que le réseau dense passe G1, sous contrôle de la NLL et des cas contrefactuels.

Pour grandes banques, alterner petites banques exhaustives et grandes banques.
Un softmax sur négatifs échantillonnés change le normalisateur ; si l'on utilise
une approximation corrigée à l'entraînement, en documenter l'estimateur et vérifier
sa dérive contre un calcul exhaustif. La calibration finale se fait toujours sur
les banques réellement utilisées.

## 5. Exécution sur Modal plus tard

Commencer sur un GPU avec mémoire suffisante pour mesurer le débit réel du pilote,
puis choisir A100/H100 ou une autre configuration à partir de ces mesures. La
[documentation Modal GPU](https://modal.com/docs/guide/gpu) décrit la sélection
des accélérateurs ; aucune disponibilité ni aucun prix futur n'est supposé ici.

Architecture d'un job à préparer : image avec dépendances figées ; volume de
données versionné ; seed et manifeste de splits ; checkpoint reprenable contenant
poids, optimiseur, ordonnanceur, RNG et position dans le corpus ; journal JSONL ;
évaluations sur splits gelés ; export ONNX hors de la boucle critique. Pas de
clé de service dans un checkpoint ou dans le navigateur.

Le nombre de paramètres seul ne détermine pas le budget GPU : longueur de l'état,
taille des banques, coût des professeurs et construction des contre-exemples
peuvent dominer. Utiliser :

```text
durée mesurée ≈ volume / débit effectif du pilote
budget ≈ durée × tarif observé + données/professeurs/stockage
```

Les recettes à tester incluent AdamW, BF16, accumulation de gradients, clipping,
échauffement puis décroissance du taux d'apprentissage. Ce sont des choix de
départ, pas des hyperparamètres déjà validés pour DEX. Le paramètre `ρ`/`λ`
appartient à chaque expérience et à son manifeste ; le changer invalide les
bornes, le modèle de coût et la calibration précédents.

## 6. Livrable exigé à la fin de l'entraînement

Un checkpoint bilingue ; son tokenizer ; graphe(s) ONNX testés ; banque et schéma
compilés versionnés ; rapport de qualité dense ; rapport d'accélération ;
calibrateur et domaine de validité ; limites de mémoire ; modèle de coût matériel ;
tests de bout en bout sur navigateur ; fiche modèle donnant les échecs connus.

Un fichier de poids sans ces résultats ne suffira pas à dire « meilleur que Jev ».
