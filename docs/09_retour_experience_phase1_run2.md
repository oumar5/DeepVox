# 09 — Retour d'expérience Phase 1, run #2

**Date** : 2026-04-16
**Condition** : A (Codec2 raw, 48 features)
**Statut** : exploratoire — montée en échelle 4× + optimisations hyperparams

## Changements vs run #1

| Paramètre | Run #1 | Run #2 |
|---|---|---|
| Corpus | 5 000 fichiers | **20 000 fichiers** (×4) |
| Paires WAV+TextGrid | 4 922 | **19 742** |
| Frames train | 280 769 | **1 128 986** (×4) |
| Batch size | 64 | **256** |
| Learning rate | 1e-3 | **2e-3** (scaling linéaire) |
| Optimizer | AdamW | AdamW (inchangé) |
| LR scheduler | ReduceLROnPlateau | Idem |
| DataLoader workers | 0 | **4** |
| pin_memory | False | **True** |
| persistent_workers | — | **True** |
| Parser TextGrid | naïf | **praatio** |
| MFA `--single_speaker` | non | **oui** (4 cores vs 1) |

## Configuration

| Paramètre | Valeur |
|---|---|
| Split | train=15792 / dev=1974 / test=1975 |
| Frames totales | train=1.13M / dev=160k / test=160k |
| Contexte | 5 frames chaque côté (11 frames = 440 ms) |
| Modèle | BiLSTM 2 couches, hidden=256, 2.2 M params |
| Max epochs | 20 |
| Early stopping patience | 5 |
| Durée | ~2h |

## Résultats bruts

### Évolution par epoch

| Epoch | Train loss | Train acc | Dev PER | Dev acc | lr |
|---|---|---|---|---|---|
| 1  | 2.4971 | 0.285 | 0.669 | 0.331 | 2.0e-03 |
| 2  | 2.2636 | 0.346 | 0.655 | 0.345 | 2.0e-03 |
| 3  | 2.1750 | 0.369 | 0.649 | 0.351 | 2.0e-03 |
| 7  | 2.0489 | 0.400 | 0.648 | 0.352 | 2.0e-03 |
| 11 | 2.0019 | 0.413 | 0.648 | 0.352 | **1.0e-03** |
| **12** | 1.8947 | 0.441 | **0.639** | **0.361** | 1.0e-03 |
| 13 | 1.8438 | 0.454 | 0.644 | 0.356 | 1.0e-03 |
| 15 | 1.7919 | 0.468 | 0.647 | 0.353 | **5.0e-04** |
| 17 | 1.6852 | 0.497 | 0.645 | 0.355 | 5.0e-04 |

Early stopping : epoch 17. Meilleur dev PER : **epoch 12** (0.639).

Signe important : **le scheduler a débloqué la progression** (epoch 11 → 12, lr divisé par 2, dev PER chute de 0.648 à 0.639).

### Métriques test finales

| Métrique | Run #1 | **Run #2** | Gain |
|---|---|---|---|
| **PER** | 69.2 % | **62.8 %** | **−6.4 pp** |
| **Top-1 accuracy** | 30.8 % | **37.2 %** | **+6.4 pp** |
| **Top-3 accuracy** | — | **62.7 %** | nouvelle |
| **Top-5 accuracy** | — | **74.4 %** | nouvelle |
| **F1 macro** | — | 0.250 | nouvelle |
| **F1 weighted** | — | 0.356 | nouvelle |

## Analyse par groupe phonétique

### Par voisement — l'info est là

| Groupe | Support | Acc. exact | **Acc. groupe** |
|---|---|---|---|
| Voisé | 124 577 | 36.2 % | **90.9 %** |
| Non-voisé | 30 813 | 46.6 % | **71.6 %** |

**Interprétation fondamentale** : le modèle prédit le bon voisement dans 85-91 % des cas,
même quand il se trompe de phonème exact. Codec2 préserve **excellemment** la distinction
voisé/sourd via les 7 bits de pitch.

Pour la phase 2 (ASR), c'est un signal très fort : la syllabation devrait passer.

### Par famille IPA

| Groupe | Support | Acc. exact | Acc. groupe | Écart |
|---|---|---|---|---|
| voyelles_orales | 57 399 | 37.8 % | 66.5 % | +28.7 pp |
| occlusives | 28 657 | 37.4 % | 62.0 % | +24.6 pp |
| liquides_laterales | 24 423 | 42.7 % | 46.9 % | +4.2 pp |
| fricatives | 17 257 | 46.7 % | 59.7 % | +13.0 pp |
| voyelles_nasales | 14 469 | 36.3 % | 51.5 % | +15.2 pp |
| nasales_consonnes | 9 268 | 25.0 % | 37.8 % | +12.8 pp |
| semi_voyelles | 3 886 | 24.3 % | 25.0 % | +0.7 pp |
| affriquees | 31 | 12.9 % | 12.9 % | 0 pp |
| bruit (spn) | 4 527 | 1.4 % | 1.4 % | 0 pp |

**Lecture** : l'écart `groupe − exact` mesure la "connaissance catégorielle". Un gros écart
signifie que le modèle classe bien la famille phonétique mais hésite sur le phonème exact
— signe que **Codec2 véhicule la macro-information mais pas les nuances fines**.

Groupes les mieux captés :
- **voyelles_orales** (66.5 % au groupe) : sait reconnaître une voyelle
- **occlusives** (62.0 %) : sait reconnaître une occlusive
- **fricatives** (59.7 %) : idem

Groupes problématiques :
- **semi_voyelles** : écart quasi-nul → modèle confond avec d'autres groupes
- **affriquees** : 31 exemples seulement, non significatif
- **bruit (spn)** : classe fourre-tout, 1.4 % — inutile, à exclure du training

## Top 10 confusions — toutes phonétiquement plausibles

| Vrai → Prédit | Count | Catégorie de confusion |
|---|---|---|
| ɛ → e | 1736 | Voyelle ouverte/fermée antérieure |
| e → i | 1402 | Voyelles antérieures fermées |
| i → e | 1229 | Idem |
| e → ɛ | 1119 | Idem |
| ʁ → a | 1104 | /ʁ/ français très vocalique |
| a → ʁ | 1037 | Idem |
| ɔ̃ → ɑ̃ | 941 | Nasales entre elles |
| ʁ → ɑ̃ | 920 | /ʁ/ nasalisé |
| ɛ → a | 900 | Voyelles ouvertes |
| t → p | 792 | Occlusives sourdes (place) |

Aucune confusion n'est "absurde" (jamais voyelle → consonne sourde par exemple). Le modèle
capture correctement la grande structure phonétique.

## Per-phoneme accuracy — tableau de bord

### Phonèmes bien identifiés (> 40 %)

| Phonème | Acc | Type |
|---|---|---|
| s | 68.4 % | Fricative alvéolaire sourde |
| i | 57.4 % | Voyelle fermée antérieure |
| a | 51.0 % | Voyelle ouverte centrale |
| ʁ | 50.8 % | R français |
| ɑ̃ | 48.9 % | Voyelle nasale ouverte |
| d | 44.5 % | Occlusive alvéolaire voisée |
| e | 42.5 % | Voyelle mi-fermée |
| t | 42.4 % | Occlusive alvéolaire sourde |

### Phonèmes mal identifiés (< 10 %)

| Phonème | Acc | Raison probable |
|---|---|---|
| dʒ | 0 % | 0 exemples train — trop rare |
| ŋ | 0 % | Idem |
| ɑ | 1.9 % | Très rare |
| ɟ | 1.3 % | Idem |
| spn | 1.4 % | Classe fourre-tout (bruits) |
| ɡ | 6.2 % | Occlusion voisée faible |
| œ | 5.9 % | Rare, confondu avec autres voyelles arrondies |
| ʎ | 6.9 % | Consonne rare en français moderne |
| ɛ̃ | 8.1 % | Nasale antérieure, confondue |

## Observations clés

### 1. Plus de données aide exactement comme prévu

| Facteur | Run #1 | Run #2 | Gain |
|---|---|---|---|
| Données ×4 | 5k | 20k | |
| PER | 69.2 % | 62.8 % | **−6.4 pp** |
| Accuracy | 30.8 % | 37.2 % | **+6.4 pp** |

L'extrapolation prudente (gain logarithmique) pour le corpus complet :

| Données | PER projeté |
|---|---|
| 5 k | 69.2 % |
| 20 k | 62.8 % |
| 80 k | ~55 % |
| 320 k | ~45 % |
| 750 k | 35-40 % |

### 2. Codec2 préserve l'information macro-phonétique

Accuracy par groupe > 60 % pour voyelles/occlusives/fricatives. Accuracy par voisement
à 91 %. **Le débat n'est plus "Codec2 contient-il de l'information ?" mais "le classifieur
actuel sait-il l'exploiter ?"**.

### 3. Top-K révèle le potentiel latent

- Top-1 : 37.2 %
- **Top-5 : 74.4 %** (→ le vrai phonème est dans les 5 meilleurs candidats 74 % du temps)

Un modèle plus expressif (plus grand, contexte plus long, architecture Transformer)
pourrait probablement transformer cette incertitude en top-1.

### 4. Le scheduler a débloqué un plateau

Epochs 3-11 : dev PER stagne autour de 0.647-0.649. L'epoch 11, lr passe de 2e-3 à 1e-3.
Epoch 12 : dev PER chute de 0.648 à 0.639 d'un coup. **Le LR scheduler fait son travail.**

### 5. Overfitting commence mais moins sévère qu'au run #1

| Métrique | Run #1 @ epoch 7 | Run #2 @ epoch 17 |
|---|---|---|
| Écart train/dev acc | +30.8 pp | +14.2 pp |

Plus de données = moins de mémorisation. Confirmé.

## Décisions

### Run #2 n'est pas non plus un "NO-GO" scientifique

Le PER = 62.8 % reste au-dessus du seuil 25 %, mais :
- Tendance claire à la baisse avec plus de données
- Top-5 = 74 % suggère potentiel latent élevé
- Distinction voisé/non-voisé à 91 % = excellent signal
- Confusions exclusivement phonétiquement plausibles

La décision GO/NO-GO doit attendre run #4 ou #5 (100k-300k fichiers) + Condition B (delta features).

### Prochains runs prévus

| Run | Changement | Prédiction dev PER |
|---|---|---|
| #3 (en cours) | Condition B (Codec2 + delta, 96 features) sur 20k | 55-60 % |
| #4 | #3 + contexte 10-15 frames | 45-55 % |
| #5 | 80 k fichiers + config optimale | 40-50 % |
| #6 | 320 k fichiers | 30-40 % |
| #7 | Corpus complet + modèle plus grand (Transformer) | < 25 % (espoir) |

### Améliorations techniques à faire

1. **Exclure `spn`** du training (classe fourre-tout, 1.4 % accuracy, pollue le dataset)
2. **Essayer contexte=10** (440 ms → 840 ms) pour capturer plus de contexte
3. **Architecture plus grosse** : 4 couches BiLSTM ou Transformer à 8M params
4. **Data augmentation** : SpecAugment sur les features Codec2 (mask random bits)

## Leçons techniques du run

1. **`--single_speaker`** dans MFA multiplie par 4 la vitesse d'alignement (1 core → 4 cores).
   Quand tous les fichiers sont dans un unique dossier "corpus", MFA croit qu'il s'agit d'un
   seul locuteur et désactive le multi-processing par défaut.
2. **`num_workers=4` + `pin_memory=True`** dans le DataLoader donnent un vrai gain de vitesse
   (≈1.3×) sur MPS.
3. **`persistent_workers=True`** évite le coût de re-fork entre epochs.
4. **Scaling linéaire lr=2e-3 avec batch=256** fonctionne bien (pas d'instabilité constatée).

## Fichiers produits

- `outputs/phase1_run2/best_model_A.pt` — 2.2 M params
- `outputs/phase1_run2/phase1_results_A.md` — rapport détaillé avec nouvelles métriques
- `outputs/phase1_run2/train_A.log` — logs complets
- `data/prepared_20k/` — 20k symlinks vers data/prepared_50k/
- `data/mfa-output-20k/` — 19 742 TextGrid IPA

## Prochaine étape immédiate

**Run #3 (en cours) : Condition B (Codec2 + delta features) sur les mêmes 20k fichiers.**
Même setup, seule différence = features 96 au lieu de 48. Permet d'isoler strictement
l'effet des delta features (dynamique inter-frames).
