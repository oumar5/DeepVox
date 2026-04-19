# 14 — Retour d'expérience Phase 2 ASR, run #3 (BiLSTM CTC, 300 k fichiers)

**Date de début** : 2026-04-18
**Phase** : 2 — ASR directe (Codec2 → texte français)
**Plateforme** : Kaggle, GPU Tesla T4 (15.6 GB VRAM)
**Statut** : **en cours** — session 1 (epochs 1-17) + session 2 (epochs 18-33) terminées, session 3 à lancer

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

## Session 2 — Résultats (epochs 18-33)

**Statut** : arrêtée à epoch 33 (timeout Kaggle), évaluation test pas encore exécutée

Le resume a fonctionné correctement : `Resume from epoch 18 (best CER=0.4894)`.

### Évolution par epoch (session 2)

| Epoch | Train loss | Dev WER | Dev CER | lr | Note |
|---|---|---|---|---|---|
| 18 | 1.7821 | 0.879 | 0.478 | 3.0e-04 | Resume OK |
| 19 | 1.7346 | 0.868 | 0.460 | 3.0e-04 | |
| 20 | 1.6918 | 0.854 | 0.451 | 3.0e-04 | |
| 21 | 1.6477 | 0.841 | 0.436 | 3.0e-04 | |
| 22 | 1.6035 | 0.841 | 0.425 | 3.0e-04 | |
| 23 | 1.5623 | 0.825 | 0.416 | 3.0e-04 | |
| 24 | 1.5203 | 0.805 | 0.402 | 3.0e-04 | **CER < 40% franchie** |
| 25 | 1.4804 | 0.795 | 0.398 | 3.0e-04 | |
| 26 | 1.4497 | 0.792 | 0.388 | 3.0e-04 | |
| 27 | 1.4173 | 0.779 | 0.383 | 3.0e-04 | |
| 28 | 1.3870 | 0.769 | 0.370 | 3.0e-04 | **CER < 37%** |
| 29 | 1.3556 | 0.773 | 0.373 | 3.0e-04 | Légère remontée |
| 30 | 1.3287 | 0.755 | 0.359 | 3.0e-04 | |
| 31 | 1.3028 | 0.746 | 0.355 | 3.0e-04 | "certains" reconnu |
| 32 | 1.2789 | 0.737 | 0.350 | 3.0e-04 | "d'opérateur avant leur" |
| **33** | **1.2579** | **0.735** | **0.344** | 3.0e-04 | **Fin session 2** |

### Observations session 2

- **CER = 34.4% à epoch 33** — gain de −14.5 pp en 16 epochs (48.9% → 34.4%)
- **Le LR n'a TOUJOURS PAS été réduit** — 3e-4 depuis le début, aucun plateau
- **La loss continue de baisser** régulièrement (1.78 → 1.26)
- **WER = 73.5%** — ~3/4 des mots sont corrects en position
- Le modèle n'a pas convergé — il reste de la marge

### Exemples qualitatifs (epoch 33)

| REF | HYP | Analyse |
|---|---|---|
| en cause la propagation du épidémie aiguë de fièvre aphteuse | enpose la propacation du etidemie télu de sei rasteuses | "propacation" ≈ "propagation", "etidemie" ≈ "épidémie", "rasteuses" ≈ "aphteuse" |
| certains satellites ont changé d'opérateur avant leur lancement ou lors de leur [...] | certains satévites ent changeus d'pérateur adant ler lanement où aunors de leurs | **"certains"** exact, "changeus" ≈ "changé", "lanement" ≈ "lancement", "lors de leurs" ≈ "ou alors de leurs" |

**Progression majeure vs session 1** :
- "certains" est maintenant parfaitement reconnu (vs "saton" à epoch 17)
- "d'opérateur" est presque correct (vs "de pratere")
- "lancement" → "lanement" (1 seule lettre manquante)
- La structure syntaxique est quasi parfaite

### Durée session 2

| Étape | Durée |
|---|---|
| Preprocessing (300k MP3) | ~75 min |
| Entraînement (16 epochs × 1711s) | ~7.6 h |
| **Total session 2** | **~8.9 h** |

---

## Bilan combiné (33 epochs sur 2 sessions)

### Progression globale

| Epoch | CER | WER | Loss | Jalon |
|---|---|---|---|---|
| 1 | 73.0% | 1.081 | 2.770 | Blank collapse |
| 5 | 64.2% | 1.005 | 2.394 | Syllabes |
| 10 | 59.1% | 0.953 | 2.188 | CER < 60% |
| 17 | 48.9% | 0.890 | 1.837 | Fin session 1 |
| 20 | 45.1% | 0.854 | 1.692 | Mots reconnaissables |
| 24 | 40.2% | 0.805 | 1.520 | **CER < 40%** |
| 28 | 37.0% | 0.769 | 1.387 | CER < 37% |
| **33** | **34.4%** | **0.735** | **1.258** | **Fin session 2** |

**Vitesse de convergence** : −1.17 pp CER/epoch en moyenne sur les 33 epochs.

### Comparaison transversale Phase 2

| Run | Corpus | Epochs | Best CER (dev) | WER (dev) | Gain CER |
|---|---|---|---|---|---|
| #1 | 20k | 25 | 71.1% | 115.5% | — |
| #2 | 80k | 50 | 56.9% | 95.0% | −14.2 pp |
| **#3** | **300k** | **33** | **34.4%** | **73.5%** | **−22.5 pp** |

### Avec KenLM (post-training)

Le seuil CER < 40% est franchi (epoch 24). L'intégration de KenLM est maintenant pertinente.

Un modèle de langue n-gram français (KenLM + pyctcdecode) gagnerait typiquement
10-15 pp de CER :

| Scénario | CER estimé |
|---|---|
| Run #3 epoch 33 (greedy) | 34.4 % (mesuré) |
| Run #3 epoch 33 + KenLM | **~20-25 %** |
| Run #3 epoch 40 (greedy, projeté) | ~28-30 % |
| Run #3 epoch 40 + KenLM | **~15-20 %** |

## Fichiers produits

- `/kaggle/working/run3_300k/training_state.pth` — checkpoint complet epoch 33 (resume)
- `/kaggle/working/run3_300k/best_asr.pt` — meilleur modèle (epoch 33, CER=0.344)
- `/kaggle/working/run3_300k/checkpoint_epoch{10,20,30}.pt` — checkpoints périodiques

## Prochaines étapes

1. **Session 3** : resume epochs 34-40, objectif CER < 30%
2. **Évaluation test** sur les 15k samples test (exécuter cellules 20-21 du notebook)
3. **KenLM** : intégrer `pyctcdecode` + modèle de langue français (CER < 40% atteint)
4. **Retour d'expérience final** : compléter ce document avec résultats test et session 3
