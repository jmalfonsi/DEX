# DEX — contrat d'interface cible et API disponible

## Interface produit envisagée

Ce format décrit l'API à livrer **après** entraînement et intégration du frontend.
Il n'existe pas encore de méthode de production `dex.decide(text)`.
Le code actuel reçoit des tensors ou des vecteurs et ne comprend pas le français
à partir de poids aléatoires.

```json
{
  "state": {
    "conversation": [
      {"role": "user", "content": "Je ne veux surtout pas annuler. Je souhaite déplacer le rendez-vous."}
    ]
  },
  "variables": {
    "intent": {
      "question": "Quelle est la demande actuelle du client ?",
      "type": "choice",
      "options": [
        {"id": "A", "description": "Annuler le rendez-vous sans le remplacer."},
        {"id": "B", "description": "Conserver un rendez-vous mais changer sa date ou son heure."},
        {"id": "C", "description": "La demande ne peut pas être déterminée à partir de l'état."}
      ]
    }
  },
  "views": {
    "reschedule": {"type": "boolean", "variable": "intent", "members": ["B"]}
  },
  "compute": {"probability_tolerance": 0.01, "max_refinement_rounds": 8}
}
```

`id` est transporté par le programme et n'est pas encodé comme sens d'option.
Une description vide est refusée. Une `view` réutilise une variable explicitement
nommée ; le compilateur ne suppose pas qu'elle est équivalente à une autre question
libre. Le score ordinal ajoute une liste de niveaux numériques strictement croissants.

Pour chaque réponse, l'API devra distinguer :

| Champ | Signification |
|---|---|
| `value` | Valeur appartenant au type déclaré |
| `probability` | Estimation de la probabilité de cette valeur dans la distribution complète |
| `probability_interval` | Enveloppe d'approximation du calcul complet du même modèle |
| `argmax_certified` | Critère de dominance des intervalles satisfait ; ce n'est pas une garantie de vérité |
| `calibration` | Référence et validité empirique du calibrateur ; `unvalidated` en son absence |
| `status` | `resolved`, `budget_exhausted`, `abstained`, ou erreur de contrat |
| `distribution` | Facultatif ; toutes les options, ou top-k avec masse complémentaire explicitement comptée |

Pour `boolean`, exposer aussi `p_true` pour éviter l'ambiguïté lorsque la valeur
retenue est `false`. Pour `score`, la valeur modale et sa probabilité sont séparées
de l'espérance. Aucun champ `explanation` génératif n'est prévu.

Le backend ne doit jamais choisir implicitement un service distant lorsqu'un
navigateur manque de mémoire ou dépasse son budget. Il retourne un état explicite.
Une application peut définir ensuite sa propre stratégie.

## API de recherche disponible

- `dex.model.DEX(Config)` : réseau PyTorch différentiable, initialisé aléatoirement.
  `forward(state_ids, question_ids, option_ids)` retourne des logits `[F,N]`.
  Les questions partagent ici une banque ; grouper les champs par banque pour
  des schémas hétérogènes. Chaque séquence locale ou rubrique doit tenir dans
  `window=128`, être non vide et contenir au moins un token non pad.
- `dex.model.DecisionCore` : `prepare`, `refine_state`, `refine_options` et
  référence dense. Ces appels permettent le raccord au planificateur paresseux.
- `dex.refine.ResidualDecision` : référence CPU pour un champ, callbacks paresseux
  d'état et d'options ; intervalle et compteurs d'évaluations.
- `dex.bounds` : softmax stable, intervalles de probabilités et d'événements,
  vues catégorielle/booléenne/ordinale.
- `dex.calibration` : température, NLL, Brier et ECE sur tableaux annotés.
- `dex.state.records` : représentation JSON typée et clés de cache. Il s'agit de
  sérialisation ; le cache de tokens, le tokenizer et leur orchestration restent à intégrer.
- `web/core.mjs` : noyau numérique JavaScript et fixtures synthétiques.
- `web/neural.mjs` : exécution des graphes exportés dans ONNX Runtime Web.

Les tests ne rendent pas valides des valeurs hors contrat : nombres non finis,
probabilités négatives, IDs dupliqués ou résidus dépassant leurs bornes sont refusés.
L'implémentation de référence ne prétend pas encore fournir tous les contrôles
d'une API de production sur des données arbitraires.
