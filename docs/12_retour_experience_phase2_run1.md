# 12 — Retour d'expérience Phase 2 ASR, run #1 (BiLSTM CTC, 20 k fichiers)

**Date** : 2026-04-17
**Phase** : 2 — ASR directe (Codec2 → texte français)
**Plateforme** : Kaggle, GPU Tesla T4 (15.6 GB VRAM)
**Statut** : **terminé** (early stopping epoch 25)

## Contexte

Premier run ASR bout-en-bout : le modèle reçoit des frames Codec2 brutes (48 features,
40 ms/frame) et doit prédire directement le texte français correspondant, entraîné avec
CTC loss. Ce run valide la faisabilité de la pipeline complète sur GPU cloud après avoir
constaté que l'entraînement CTC n'est pas supporté nativement sur MPS (Apple Silicon).

## Configuration

| Paramètre | Valeur |
|---|---|
| Corpus | Common Voice FR v21.0 (Kaggle) |
| Échantillons | 20 000 (18k train / 1k dev / 1k test) |
| Entrée | Codec2 1200 bps, 48 features / 40 ms |
| Sortie | Caractères français, 49 classes (blank + unk + 47 chars) |
| Architecture | BiLSTM 3 couches, embed=256, hidden=384 |
| Paramètres | **9 112 625** (9.1 M) |
| Taille modèle | **36.5 MB** (float32) |
| Loss | CTC (blank=0, zero_infinity=True) |
| Optimizer | AdamW (lr=3e-4, weight_decay=1e-2) |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=2) |
| Gradient clipping | max_norm=5.0 |
| Batch size | 32 |
| Max epochs | 30 |
| Early stopping | patience=5 sur dev CER |
| Durée/epoch | ~113 s (≈ 47 min total) |

## Statistiques des données

| Statistique | Valeur |
|---|---|
| Samples valides | 20 000 / 20 003 tentés |
| Frames/sample | min=32, max=292, mean=140 |
| Chars/sample | min=3, max=135, mean=59 |
| Durée audio moyenne | ~5.6 s |
| Max durée filtrée | 12 s |

## Résultats — évolution par epoch

| Epoch | Train loss | Dev WER | Dev CER | lr | Note |
|---|---|---|---|---|---|
| 1 | 3.0877 | 1.000 | 1.000 | 3.0e-04 | Sortie vide (blank collapse) |
| 2 | 2.9911 | 1.000 | 0.985 | 3.0e-04 | Premier caractère : "l" |
| 3 | 2.8546 | 1.442 | 0.746 | 3.0e-04 | "e" dominant partout |
| 5 | 2.7550 | 1.196 | 0.744 | 3.0e-04 | "e", "a" apparaissent |
| 7 | 2.7186 | 1.248 | 0.738 | 3.0e-04 | "il" reconnu (premier mot) |
| 10 | 2.6617 | 1.177 | 0.730 | 3.0e-04 | "s", "o" émergent |
| 15 | 2.6062 | 1.143 | 0.728 | 3.0e-04 | "so", "oo" syllabes |
| 17 | 2.6025 | 1.193 | 0.717 | 3.0e-04 | "se", "ce" apparaissent |
| **20** | **2.5911** | **1.143** | **0.711** | 3.0e-04 | **Best CER** — "sen", "te" |
| 21 | 2.5915 | 0.990 | 0.812 | 3.0e-04 | Spike CER (instabilité) |
| 22 | 2.7901 | 0.980 | 0.803 | 3.0e-04 | Loss remonte |
| 23 | 2.7246 | 1.004 | 0.748 | 1.5e-04 | LR réduite ×0.5 |
| 25 | 2.6831 | 1.011 | 0.761 | 1.5e-04 | **Early stopping** |

## Résultats test (meilleur checkpoint, epoch 20)

| Métrique | Valeur |
|---|---|
| **WER** | **1.1546 (115.5 %)** |
| **CER** | **0.7117 (71.2 %)** |
| Échantillons test | 1 000 |

### Exemples qualitatifs (test set)

| REF | HYP |
|---|---|
| il se marie avec marie comtesse von brühl | il se e e eae o e e oe |
| tolède est la capitale politique et religieuse du royaume | a e e e a i e ae e e aa |
| il est une nouvelle fois produit par sascha paeth | il e e e o o e e se eie |
| tout le monde reprit le juge la croyait veuve elle-même prétendait l'être | a o en e e e pe aee ee e e ema |

**Observations :**
- "il" et "il se" sont parfois correctement reconnus (mots très fréquents)
- Le modèle produit surtout des voyelles isolées (e, a, o, i)
- Les consonnes rares et les mots longs sont quasi absents
- Aucun mot de contenu (noms, verbes conjugués) n'est reconnu

## Analyse de la progression CTC

### Phases observées

1. **Blank collapse** (epochs 1-2) : le modèle prédit uniquement le token blank → sortie vide, CER=100%
2. **Caractère dominant** (epochs 3-7) : découverte de "e" (lettre la plus fréquente du français, ~15% des caractères), puis "l", "a"
3. **Diversification lente** (epochs 8-20) : apparition de "s", "o", "c", "t", "n" mais sans former de vrais mots
4. **Plateau + instabilité** (epochs 21-25) : CER oscille, loss remonte, le modèle n'arrive pas à franchir le palier 70%

### Diagnostic : pourquoi le CER plafonne à 71% ?

| Facteur | Impact | Explication |
|---|---|---|
| **Données insuffisantes** | Fort | 18k samples est très peu pour de l'ASR CTC. Les systèmes CTC performants utilisent 1000h+ (≈ 600k utterances) |
| **Résolution temporelle** | Moyen | 25 fps (40 ms/frame) vs 100 fps standard → 4× moins de résolution temporelle, le modèle manque de granularité pour aligner les caractères |
| **Pas de Language Model** | Fort | Le décodage greedy sans LM ne peut pas corriger les erreurs. Un beam search + LM français gagnerait 15-20pp de CER |
| **Architecture simple** | Moyen | BiLSTM 9M params vs Conformer 30-100M params dans les systèmes ASR modernes |
| **Features Codec2** | Faible | Phase 1 a montré que l'info phonétique est là (top-5=80%), le problème est le décodeur, pas l'encodeur |

### Comparaison avec Phase 1

| Aspect | Phase 1 (phonème) | Phase 2 (ASR) |
|---|---|---|
| Tâche | Classification frame → phonème | Séquence frames → texte |
| Meilleur résultat | PER=56% (80k, epoch 4) | CER=71% (20k, epoch 20) |
| Signal positif | Top-5=80%, voisement=93% | "il", "il se" reconnus |
| Données utilisées | 80 000 fichiers | 20 000 fichiers |
| Difficulté | Plus facile (1 frame → 1 label) | Plus difficile (alignement CTC) |

## Décisions

### 1. Ce run est un baseline, pas un échec

CER=71% est élevé mais **attendu** pour un premier run CTC avec si peu de données et sans LM.
À titre de comparaison, les premiers systèmes CTC (Graves et al., 2006) avaient des CER
similaires avant l'ajout de LM et de données massives.

### 2. Prochaines améliorations prioritaires (Run #2)

Par ordre d'impact attendu :

| Action | Gain estimé | Effort |
|---|---|---|
| **×4 données** (80k samples) | −10-15 pp CER | Faible (changer MAX_SAMPLES) |
| **Plus d'epochs** (50-100) | −5-10 pp CER | Faible (changer MAX_EPOCHS) |
| **Beam search + KenLM** | −15-20 pp CER | Moyen (intégrer pyctcdecode) |
| **Augmentation audio** | −3-5 pp CER | Moyen (speed perturbation, noise) |
| **Conformer** | −10-20 pp CER | Fort (nouvelle architecture) |

### 3. Stratégie Run #2

Commencer par les gains faciles :
- `MAX_SAMPLES = 80000` (×4)
- `MAX_EPOCHS = 50`
- Garder la même architecture BiLSTM pour comparer
- Si CER < 50%, ajouter KenLM en post-traitement

### 4. Notebook unique paramétré

On garde `02_phase2_asr_kaggle.ipynb` comme notebook unique avec une cellule de config.
L'historique des runs est capturé dans les documents de retour d'expérience.

## Fichiers produits

- `outputs/phase2_run1/deepvox_asr_phase2.pt` — checkpoint best (36.5 MB, epoch 20)
- `outputs/phase2_run1/training_curves.png` — courbes loss/WER/CER/LR
- `notebooks/02_phase2_asr_kaggle.ipynb` — notebook Kaggle avec résultats complets

## Durée totale

- Preprocessing : ~5 min (20k MP3 → Codec2)
- Entraînement : ~47 min (25 epochs × 113 s)
- Évaluation : ~10 s
- **Total : ~52 min** sur Tesla T4

## Prochaine étape

**Phase 2 — Run #2** : relancer sur 80k fichiers avec 50 epochs sur Kaggle.
Objectif : CER < 50% (sans LM), confirmant que le scaling de données fonctionne aussi pour l'ASR.
