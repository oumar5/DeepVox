# 08 — Retour d'expérience Phase 1, run #1

**Date** : 2026-04-16
**Condition** : A (Codec2 raw, 48 features)
**Statut** : exploratoire — premier entraînement end-to-end

## Configuration

| Paramètre | Valeur |
|---|---|
| Corpus | Common Voice French v21.0 (Kaggle) |
| Fichiers utilisés | 5 000 (0.66 % du corpus disponible) |
| Paires (WAV + TextGrid) après MFA | 4 922 (98.4 % de succès) |
| Split | train=3937 / dev=492 / test=493 |
| Frames totales | train=280 769 / dev=40 152 / test=40 061 |
| Contexte modèle | 5 frames de chaque côté (11 frames = 440 ms) |
| Architecture | BiLSTM 2 couches, hidden=256 |
| Paramètres | 2 226 220 |
| Optimizer | AdamW, lr=1e-3, wd=1e-2 |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=2) |
| Early stopping | patience=5 |
| Max epochs | 20 |
| Device | Apple MPS |
| Durée | ~1 h 15 min |

## Résultats bruts

### Évolution par epoch (dev set)

| Epoch | Train loss | Train acc | Dev PER | Dev acc | lr |
|-------|-----------|-----------|---------|---------|-----|
| 1 | 2.6796 | 0.238 | **0.736** | 0.264 | 1.0e-03 |
| 2 | 2.3800 | 0.315 | **0.713** | 0.287 | 1.0e-03 |
| 3 | 2.1564 | 0.371 | 0.716 | 0.284 | 1.0e-03 |
| 4 | 1.9630 | 0.421 | 0.730 | 0.270 | 1.0e-03 |
| 5 | 1.8166 | 0.460 | 0.736 | 0.264 | 5.0e-04 |
| 6 | 1.5525 | 0.535 | 0.736 | 0.264 | 5.0e-04 |
| 7 | 1.4367 | 0.567 | 0.741 | 0.259 | 5.0e-04 |

Meilleur modèle : **epoch 2** (dev_PER=0.713). Early stopping déclenché à l'epoch 7.

### Métriques test finales

| Métrique | Valeur | Cible protocole |
|---|---|---|
| PER | 0.6924 (69.2 %) | ≤ 15 % |
| Accuracy top-1 | 30.8 % | ≥ 85 % |
| Macro precision | 0.2251 | — |
| Baseline hasard | 2.3 % | — |

## Analyse

### Le modèle apprend quelque chose

L'accuracy test est **13× supérieure au hasard** (30.8 % vs 2.3 %). Codec2 contient
donc de l'information phonétique discriminante — la question n'est pas "y en a-t-il ?"
mais "combien ?" et "suffit-elle avec plus de données / un meilleur setup ?".

### Meilleurs phonèmes identifiés

| Phonème | Accuracy | Catégorie |
|---|---|---|
| `s` | 73.4 % | Fricative alvéolaire sourde |
| `i` | 51.4 % | Voyelle fermée antérieure |
| `t` | 49.4 % | Occlusive alvéolaire sourde |
| `ɑ̃` | 45.1 % | Voyelle nasale ouverte |
| `a` | 44.9 % | Voyelle ouverte centrale |
| `d` | 40.5 % | Occlusive alvéolaire voisée |
| `ʁ` | 39.9 % | Consonne dorsale (R français) |
| `p` | 32.9 % | Occlusive bilabiale sourde |
| `l` | 30.9 % | Liquide latérale |

### Pires phonèmes (accuracy ≤ 2 %)

- `ɑ`, `ɟ`, `ŋ`, `tʃ` : 0 % — trop rares dans 5 000 fichiers
- `ɡ` : 0.5 % — occlusive voisée mal discriminée
- `spn` : 0.2 % — catégorie fourre-tout (bruits)
- `b` : 1.2 % — souvent confondu

### Top confusions (très informatives)

| Vrai → Prédit | Count | Interprétation |
|---|---|---|
| e → i | 408 | Voyelles antérieures fermées proches |
| ʁ → a | 392 | /ʁ/ français très vocalique |
| ɛ → a | 356 | Voyelles ouvertes |
| ʁ → ɑ̃ | 329 | Idem |
| p → t | 308 | Occlusives sourdes (différence = place) |
| e → ɛ | 279 | e fermé vs ouvert |
| a → ɑ̃ | 275 | Oral vs nasal |
| i → e | 275 | Voyelles fermées antérieures |
| k → t | 271 | Occlusives sourdes |
| ɛ → e | 256 | Idem |

**Toutes les confusions sont phonétiquement plausibles.** Le modèle confond dans les
bonnes catégories — il ne fait pas n'importe quoi, il manque juste de finesse.

### Surapprentissage évident

| Signal | Valeur |
|---|---|
| Écart train/dev à l'epoch 7 | +30.8 pp (56.7 % vs 25.9 %) |
| Meilleur dev PER atteint dès | Epoch 2 |
| Evolution dev après epoch 2 | Stagnation puis dégradation |

Le modèle a commencé à mémoriser le train dès l'epoch 3-4. Dropout (0.3), weight
decay (1e-2) et early stopping n'ont pas suffi à contenir l'overfitting avec
seulement 3937 fichiers d'entraînement.

## Hypothèses sur les causes

### H1 — Manque de données (probable)

Corpus utilisé : 0.66 % du disponible. Certaines classes (`ɡ`, `b`, `tʃ`, `dʒ`, `ɟ`,
`ŋ`, `ʎ`) ont quelques dizaines à centaines d'exemples seulement. Un BiLSTM 2.2 M
params a besoin de beaucoup plus pour capturer ces distinctions fines.

### H2 — Contexte trop court (probable)

11 frames = 440 ms. Beaucoup de phonèmes ont une durée et un contexte d'influence
plus long. Whisper utilise 30 s de contexte.

### H3 — Codec2 perd effectivement de l'info phonétique (à vérifier)

Vérifiable seulement après avoir éliminé H1 et H2. Impossible de conclure avec
0.66 % du corpus.

### H4 — La grille 40 ms est trop grossière (peu probable)

MFA aligne à ~10 ms. On re-quantifie à 40 ms ce qui crée du bruit de frontière
phonétique. Mais ça n'explique pas un PER de 69 %.

## Décisions

### Ce run n'est PAS un NO-GO

Le critère ≤ 15 % PER du protocole suppose un corpus d'entraînement complet.
**Invalider l'hypothèse Codec2 sur 0.66 % des données serait prématuré.**

### Prochains runs à faire avant toute conclusion

| Run | Changement clé | Prédiction |
|---|---|---|
| **#2** | 50 000 fichiers (10× plus) | Dev PER 45-55 % |
| **#3** | #2 + Condition B (delta) | Dev PER 40-50 % |
| **#4** | #3 + context=15 frames | Dev PER 35-45 % |
| **#5** | 200 000 fichiers + config optimale | Dev PER 25-35 % |
| **#6** | Corpus complet 755k | Dev PER < 25 % (espoir) |

La décision GO / NO-GO se prendra après le run #5 ou #6, pas avant.

### Métriques à ajouter dans le prochain rapport

Déjà codées mais pas activées dans ce run (fichier eval/metrics.py) :
- Top-3 / Top-5 accuracy
- F1 macro / weighted
- Accuracy par groupe IPA (voyelles orales/nasales, occlusives, fricatives, …)
- Accuracy par voisement

Ces métriques sont bien plus informatives que le PER brut pour comprendre
quel type d'information Codec2 préserve.

## Leçons techniques

1. **La grille 40 ms Codec2 est un réel choix architectural**, pas une simplification
   — elle définit la résolution temporelle maximale du système.
2. **MFA ne produit pas du SAMPA-FR mais de l'IPA** — adapté dans le code au run #1.
3. **Kaggle fragmente les datasets en milliers de dossiers** — il faut consolider
   avant usage, opération lente (~2 h pour 840 k fichiers via `mv -exec`).
4. **`pycodec2` avec NumPy 2 casse `torch 2.2`** — downgrade NumPy à 1.26 pour
   compatibilité avec PyTorch installé.
5. **AdamW + ReduceLROnPlateau sont indispensables** — Adam seul + lr fixe
   saturaient encore plus vite.
6. **Le bug batch_size (64 vs 704)** initial venait du fait que le dataset retourne
   1 label par séquence (frame centrale), pas un label par frame. Fix : `logits[:, center, :]`.

## Fichiers produits

- `outputs/phase1/best_model_A.pt` — checkpoint meilleur dev
- `outputs/phase1/phase1_results_A.md` — rapport test
- `outputs/phase1/train_A.log` — logs complets d'entraînement
- `data/prepared/` — 5000 WAV 8 kHz + .lab
- `data/mfa-output/` — 4922 TextGrid IPA alignés

## Prochaine étape

Attendre la fin du `mv` en arrière-plan (consolidation des 840 k clips MP3) pour
pouvoir préprocesser un sous-ensemble 10× plus gros (50 000 fichiers) et lancer
le run #2.
