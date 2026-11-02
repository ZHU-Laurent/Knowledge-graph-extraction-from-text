Datasets :

SCIERC : Articles scientifiques
FRAME : Art-historical Image Descriptions

Sélection du 2026-11-02. Aucun jeu n'a encore été téléchargé : format, annotations et licences restent à contrôler sur un échantillon.

## Datasets retenus

| Étape | Dataset | Rôle | Contenu | Accès | Licence |
|---|---|---|---|---|---|
| 1. Test rapide | [CoNLL04](https://huggingface.co/datasets/DFKI-SLT/conll04) | Faire tourner et déboguer la chaîne | Presse, 1 437 phrases, 4 types d'entités, 5 relations | Hugging Face | ? |
| 2. Version finale | [SciERC](http://nlp.cs.washington.edu/sciIE/) | Entraîner, comparer les 3 extracteurs, construire le graphe | 500 résumés d'IA, 6 types d'entités, 7 relations, coréférence | Site officiel ou version du dépôt [PURE](https://github.com/princeton-nlp/PURE) | ? |
| 3. Extension : robustesse | [SciER](https://arxiv.org/abs/2410.21155) | Tester sur des articles complets, plus longs que des résumés | 106 articles d'IA, plus de 24 000 entités, plus de 12 000 relations | [GitHub](https://github.com/edzq/SciER) | ? |
| 4. Extension : autre domaine | [FRAME](https://arxiv.org/abs/2602.19133) | Vérifier si la chaîne s'adapte à un domaine très différent | Descriptions d'œuvres d'art, 37 types d'entités, relations typées, coréférence, alignement Wikidata | [Zenodo](https://doi.org/10.5281/zenodo.18724632) | CC BY 4.0 |

## Pourquoi cet ordre

- **CoNLL04 d'abord :** petit et déjà en JSONL, il permet de vérifier le code et l'évaluation sans attendre de longs entraînements. Ses scores ne servent pas de résultat final.
- **SciERC ensuite :** seul ce jeu sert aux résultats principaux. Il couvre toutes les étapes, et il sert de référence dans des articles comme PURE et PL-Marker.
- **SciER et FRAME seulement si le temps le permet :** leurs schémas diffèrent de SciERC, donc il faut adapter les étiquettes ou réentraîner. FRAME demande aussi de convertir son format XMI et de vérifier sa taille.

## Points à vérifier

- **SciERC :** éviter les copies Hugging Face, non officielles et sans licence. `nsusemiehl/SciERC` est préparée uniquement pour la classification de relations.
- **Découpages :** garder les découpages officiels, et ne pas utiliser le test de SciERC avant l'évaluation finale.
- **SciER :** relever si un test hors distribution est fourni et utilisable.

## Écartés

- **DocRED, Re-DocRED, DWIE :** utiles pour les relations entre phrases, mais plus lourds et moins nécessaires ici.
- **TACRED, Re-TACRED :** accès potentiellement soumis à licence.
- **REBEL, SREDFM, T-REx :** annotation automatique, moins fiable pour évaluer.
- **Français (POPCORN, REDFM) :** à reprendre seulement si le projet doit être en français.
