# ONNX Runtime Web

Le runtime facultatif dans `vendor/` est ONNX Runtime Web 1.30.0,
Copyright Microsoft Corporation, licence MIT.

- Code et licence : https://github.com/microsoft/onnxruntime
- Distribution : https://www.npmjs.com/package/onnxruntime-web
- Provenance et empreintes de la copie locale : `../artifacts/runtime-manifest.json`.

Le script `scripts/prepare_browser.py` prépare ces fichiers. Le code DEX ne dépend
pas d'un CDN à l'exécution. Les graphes contiennent des poids aléatoires produits
localement, pas un modèle tiers entraîné.
