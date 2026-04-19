# 14 — Retour d'expérience Phase 2 ASR, run #3 (BiLSTM CTC, 300 k fichiers)

**Date de début** : 2026-04-18
**Phase** : 2 — ASR directe (Codec2 → texte français)
**Plateforme** : Kaggle, GPU Tesla T4 (15.6 GB VRAM)
**Statut** : **en cours** — session 1 terminée (epochs 1-17), session 2 en cours

## Contexte

Troisième run ASR, faisant suite au run #2 (80k, CER=56.9 %). L'objectif est de
confirmer que le scaling continue de fonctionner au-delà de 80k et de viser un CER < 40 %.

C'est le premier run utilisant le **système de resume** : le checkpoint complet
(model + optimizer + scheduler + history) est sauvegardé à chaque epoch dans
`/kaggle/working/{RUN_NAME}/training_state.pth`, permettant de reprendre l'entraînement
sur une nouvelle session Kaggle.

## Configuration

| Paramètre | Run #2 | **Run #3** |
|---|---|---|
| Corpus | 80 000 | **300 000** (×3.75) |
| Train / Dev / Test | 72k / 4k / 4k | **270k / 15k / 15k** |
| MAX_EPOCHS | 50 | **40** |
| PATIENCE | 7 | **10** |
| Architecture | BiLSTM 3L, h=384 | idem |
| Paramètres | 9.1 M | idem |
| Optimizer | AdamW lr=3e-4 | idem |
| Batch size | 32 | idem |
| Durée/epoch | 456 s | **~1 538 s (~25.6 min)** (×3.4) |
| Sessions Kaggle | 1 | **2** (resume) |

## Statistiques des données

| Statistique | Run #2 (80k) | **Run #3 (300k)** |
|---|---|---|
| Samples valides | 80 000 | **300 000** |
| Skipped | 9 | **37** |
| Frames/sample (mean) | 136 | **129** |
| Chars/sample (mean) | 59 | **60** |
| Chars/sample (max) | 200 | **216** |

## Session 1 — Résultats (epochs 1-17)

### Évolution par epoch

| Epoch | Train loss | Dev WER | Dev CER | lr | Note |
|---|---|---|---|---|---|
| 1 | 2.7699 | 1.081 | 0.730 | 3.0e-04 | Démarrage, "e" dominant |
| 2 | 2.6473 | 1.093 | 0.701 | 3.0e-04 | Premières syllabes |
| 3 | 2.5103 | 0.982 | 0.680 | 3.0e-04 | WER < 1.0 dès epoch 3 |
| 5 | 2.3935 | 1.005 | 0.642 | 3.0e-04 | |
| 8 | 2.2734 | 0.974 | 0.612 | 3.0e-04 | Mots partiels émergent |
| 10 | 2.1879 | 0.953 | 0.591 | 3.0e-04 | **CER < 60% dès epoch 10** |
| 13 | 2.0526 | 0.931 | 0.543 | 3.0e-04 | "propotation" ≈ "propagation" |
| 15 | 1.9474 | 0.910 | 0.519 | 3.0e-04 | CER sous 52% |
| 16 | 1.8972 | 0.895 | 0.503 | 3.0e-04 | **CER < 50%** |
| **17** | **1.8367** | **0.890** | **0.489** | 3.0e-04 | **Fin session 1** |

### Observations session 1

- **CER = 48.9% à epoch 17** — déjà meilleur que le Run #2 final (56.9% à epoch 50)
- **Le LR n'a jamais été réduit** — toujours à 3e-4, pas de plateau
- **La loss baisse de façon régulière** — aucun signe de convergence
- **WER < 0.90** — le modèle produit le bon nombre de mots
- Durée session 1 : 17 × 1538s ≈ **7.3 h**

### Comparaison à epoch équivalent (epoch 17)

| Métrique | Run #2 (80k) | **Run #3 (300k)** | Gain |
|---|---|---|---|
| CER | 0.661 | **0.489** | **−17.2 pp** |
| WER | 1.013 | **0.890** | **−12.3 pp** |
| Loss | 2.418 | **1.837** | −0.58 |

Le ×3.75 de données apporte un gain massif de −17 pp de CER à nombre d'epochs égal.

### Exemples qualitatifs (epoch 17)

| REF | HYP | Analyse |
|---|---|---|
| en cause la propagation du épidémie aiguë de fièvre aphteuse | en coe te la propotation de etio le de seestase | "la propotation" ≈ "la propagation" |
| certains satellites ont changé d'opérateur avant leur lancement ou lors de leur [...] | saton ca sélite ren change de pratere apant laencsement au lor de larvie retoe | "sélite" ≈ "satellites", "change" ≈ "changé", "lancement" ≈ "laencsement" |

**Progression majeure** : les mots de contenu sont maintenant partiellement reconnus,
pas seulement les mots fonctionnels comme dans le Run #2.

### Durée et budget Kaggle

| Étape | Durée |
|---|---|
| Preprocessing (300k MP3) | ~75 min |
| Entraînement (17 epochs × 1538s) | ~7.3 h |
| **Total session 1** | **~8.5 h** (sur 12h disponibles) |

La session s'est arrêtée probablement par timeout Kaggle (12h incluant preprocessing).
Le checkpoint `training_state.pth` a été sauvegardé à epoch 17.

---

## Session 2 — Résultats (epochs 18-??)

**Statut** : en cours

_À compléter quand la session 2 sera terminée._

| Epoch | Train loss | Dev WER | Dev CER | lr | Note |
|---|---|---|---|---|---|
| 18 | | | | | Resume depuis checkpoint |
| ... | | | | | |

---

## Projection

### Basée sur la tendance epochs 1-17

Le CER baisse d'environ **1.5 pp par epoch** (tendance linéaire sur epochs 10-17).
Si cette tendance se maintient :

| Epoch | CER projeté |
|---|---|
| 17 | 48.9 % (mesuré) |
| 25 | ~37 % |
| 30 | ~30-33 % |
| 40 | ~25-28 % |

### Avec KenLM (post-training)

Un modèle de langue n-gram français (KenLM + pyctcdecode) gagnerait typiquement
10-15 pp de CER supplémentaires :

| Scénario | CER estimé |
|---|---|
| Run #3 epoch 40 (greedy) | ~25-28 % |
| Run #3 epoch 40 + KenLM | **~15-20 %** |

## Comparaison transversale Phase 2

| Run | Corpus | Epochs | CER test | WER test | Gain CER vs précédent |
|---|---|---|---|---|---|
| #1 | 20k | 25 | 71.2 % | 115.5 % | — |
| #2 | 80k | 50 | 56.9 % | 95.0 % | −14.3 pp |
| **#3** | **300k** | **17 (session 1)** | **~48.9 % (dev)** | **~89.0 % (dev)** | **−8.0 pp (et pas fini)** |

## Fichiers produits (session 1)

- `/kaggle/working/run3_300k/training_state.pth` — checkpoint complet (resume)
- `/kaggle/working/run3_300k/best_asr.pt` — meilleur modèle (epoch 17, CER=0.489)

## Prochaine étape

1. **Session 2** : resume epochs 18-40, objectif CER < 35%
2. **Évaluation test** sur les 15k samples test
3. **KenLM** : si CER < 40%, intégrer beam search + modèle de langue français
