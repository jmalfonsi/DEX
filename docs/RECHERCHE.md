# DEX — recherche et antériorités

Revue ciblée du 29 septembre 2026. Sources primaires : documentation des auteurs,
articles et dépôts officiels. Cette revue n'est pas une preuve d'originalité
mondiale, ni une recherche juridique d'antériorité. Les prépublications récentes
sont des pistes à reproduire, pas des résultats admis sans réserve.

## Référence précise : Jev

La référence retenue est **Jev 1.13.0 de TypeSafe AI**, confirmé par l'utilisateur.
La documentation indique 32k tokens pour l'état et la question la plus longue,
64k pour l'ensemble de la requête. L'anglais est sa langue d'entraînement
principale. Le nombre de paramètres et le matériel nécessaire à une inférence
locale ne sont pas établis dans les sources consultées : nous ne leur attribuons
aucune valeur. [Modèles officiels](https://docs.typesafe.ai/models).

Les `Choice` acceptent jusqu'à 255 options, et plusieurs questions sont déjà
traitées en parallèle. Ce parallélisme n'est donc pas une invention possible
pour DEX. Les identifiants de questions sont opaques, mais les noms d'options
et leurs descriptions sont vus par Jev. [Choice](https://docs.typesafe.ai/primitives/choice).

La documentation reconnaît des difficultés avec le contexte distrayant, les
indirections, les dates et les relations entre questions logiquement liées.
Ces limitations motivent nos expériences ; elles ne prouvent pas que DEX les
résoudra. [Limites documentées](https://docs.typesafe.ai/model-jaggedness/jev-1.13).

Le `confidence` de Jev résume la forme de sa distribution. Il ne faut pas le
confondre avec la probabilité d'un événement, ni le comparer directement à une
fréquence de décisions correctes. DEX utilisera des noms distincts pour la
probabilité, l'erreur d'approximation et la politique d'abstention.
[Confidence](https://docs.typesafe.ai/confidence).

Le discours de lancement présente un entraînement RLCD et une architecture
spécialisée. Les détails publics consultés ne permettent pas de reconstruire
ce mécanisme. Aucun résultat de supériorité de DEX ne peut être déduit de ces
annonces. [Annonce TypeSafe](https://typesafe.ai/blog/introducing-system-one-models-and-jev).

## Ce qui existe déjà

| Travail | Apport pertinent | Conséquence pour DEX |
|---|---|---|
| [GLiClass](https://arxiv.org/abs/2508.07662) | Classification avec descriptions d'étiquettes, sans génération libre | Un encodeur et une tête de classification ouverte ne suffisent pas à revendiquer une nouvelle architecture |
| [Perceiver](https://arxiv.org/abs/2103.03206) | Lecture d'une grande entrée par un ensemble compact de latents | Un petit ensemble de requêtes qui lit l'état n'est pas nouveau |
| [ColBERTv2](https://arxiv.org/abs/2112.01488) | Représentations précalculées et interaction tardive | Les banques d'options compilées sont un choix d'ingénierie connu |
| [BASED](https://arxiv.org/abs/2402.18668) | Compromis entre mémoire comprimée, débit et rappel | Une mémoire de taille fixe n'apporte pas gratuitement un contexte fidèle et illimité |
| [BLT](https://arxiv.org/abs/2412.09871) | Traitement des octets par segments latents | Abandonner les tokens n'est pas une innovation suffisante ; le préentraînement nécessaire doit être chiffré |
| [AdaptiveSoftmax](https://papers.nips.cc/paper/2024/hash/d52dbd66219dc4e432e0bd4f9c25c4c3-Abstract-Conference.html) | Estimation adaptative des plus grandes probabilités et du normalisateur | L'échantillonnage adaptatif de la softmax est antérieur |
| [CSV-Decode](https://arxiv.org/abs/2511.21702) | Bornes géométriques, certification du top-k et approximation de softmax | Nous avons abandonné la prétention d'inventer les bornes centroïde-rayon ; ce n'est pas le cœur proposé pour DEX |
| [Vertex-Softmax](https://arxiv.org/abs/2605.10974) | Optimisation d'objectifs d'attention sous intervalles de logits | Les intervalles sur une softmax et la vérification d'attention sont aussi des antériorités |
| [Circuits probabilistes](https://arxiv.org/abs/2402.00759) | Calculs probabilistes structurés et tractables | Partager une mesure entre plusieurs vues logiques est connu ; DEX v0.1 se limite à une partition catégorielle explicite |
| [Temperature scaling](https://arxiv.org/abs/1706.04599) | Calibration postérieure simple | La température sera un outil expérimental, pas une promesse universelle de calibration |
| [Prédiction conforme](https://arxiv.org/abs/2107.07511) | Ensembles prédictifs avec couverture sous hypothèses | Une couverture marginale ne certifie pas chaque décision et ne remplace pas des probabilités calibrées |

Deux publications récentes ciblent précisément les modèles de décision :

- [Type-Safe Is Not Error-Free](https://arxiv.org/abs/2609.26758) étudie l'influence
  des noms d'options lorsque leur affectation aux critères change. DEX séparera
  les identifiants utilisés par le programme des textes sémantiques ; les tests
  échangeront noms, ordre et rubriques indépendamment.
- [Beyond Calibration](https://arxiv.org/abs/2609.33209) distingue calibration et
  cohérence entre questions liées. DEX ne promet pas de déduire toutes les
  équivalences exprimées en langage naturel. Il garantit les identités uniquement
  pour des vues déclarées du même événement.

## Ce que le workspace contient déjà

Lecture du [document CALIBRE](../../CALIBRE/CONCEPTION.md), sans modification de ce
projet. CALIBRE décrit déjà : fragmentation, cache, lecteur partagé, banques
d'options, routage, corrections de scores, calibration et déploiement navigateur.
Il contient également des mesures qui déconseillent de transposer directement
une latence PyTorch à WASM. Le [README LAYA](../../LAYA/README.md) documente une
autre implémentation locale. Ces projets sont des références internes, pas des
preuves publiées de généralisation.

**DEX ne revendique aucun de ces ingrédients comme nouveau.** Une nouvelle marque
autour du même assemblage ne répondrait pas à la demande.

## Hypothèse de contribution retenue

Concevoir **la fonction de décision elle-même pour que ses deux raffinements
coûteux soient bornés structurellement**, puis exposer une enveloppe sur sa
distribution complète pendant leur évaluation partielle :

1. les fragments lus finement apportent des vecteurs résiduels de norme bornée ;
2. les options examinées finement apportent des corrections scalaires bornées ;
3. toute masse du contexte et toute option non raffinée restent comptées ;
4. un même modèle complet sert de référence pour tous les budgets de calcul ;
5. les vues typées et le calibrateur se rattachent à cette même distribution.

La nouveauté **candidate** est ce couplage entre représentation apprise,
raffinement des deux côtés et contrat probabiliste de décision, dans un modèle
embarqué. Nous ne présentons ni les inégalités utilisées, ni le cache, ni le
classifieur comme des découvertes. Une publication nécessiterait encore une revue
d'antériorité plus large et des résultats d'ablation convaincants.

L'intérêt scientifique est falsifiable : si borner les résidus dégrade trop la
compréhension, ou si fermer les intervalles coûte autant que tout calculer,
l'hypothèse doit être rejetée. La référence exécutable permet déjà de tester ces
deux questions au lieu d'en rester à un schéma.

## Sources de données et méthodes d'évaluation retenues

- [MASSIVE, dépôt des auteurs](https://huggingface.co/datasets/AmazonScience/massive) :
  intentions parallèles, utile pour comparer EN et FR ; insuffisant pour le
  jugement métier général.
- [XNLI](https://aclanthology.org/D18-1269/) : relations entre prémisses et hypothèses,
  avec partage des splits entre traductions pour éviter les fuites.
- [MultiEURLEX](https://aclanthology.org/2021.emnlp-main.559/) : documents multilingues
  et classification multilabel ; les exemples ne doivent pas être transformés
  artificiellement en questions à réponse unique lorsque plusieurs labels sont vrais.
- [RULER](https://arxiv.org/abs/2404.06654) : rappel, chaînes de dépendance et
  agrégations sur contexte long ; une simple aiguille retrouvée ne suffit pas.
- [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) et
  [WebGPU](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html) : le support
  d'opérateurs diffère selon le moteur. WASM est la première référence mesurée.
- [Modal GPU](https://modal.com/docs/guide/gpu) : infrastructure possible pour
  l'entraînement futur. Aucun job ni dépense Modal n'a été lancé.

Les versions, licences, champs utilisables et restrictions de chaque corpus
devront être inscrits dans le registre avant ingestion. Le fait qu'un dataset
soit visible sur le Web n'est pas une licence d'entraînement implicite.
