# 17 — Proposition Phase 3 : Migration Conformer pour DeepVox ASR

**Auteur** : équipe DeepVox
**Date** : 2026-04-30
**Statut** : Proposition (à valider avant lancement)
**Document de référence** : [`16_etude_comparative_bilstm_conformer.md`](16_etude_comparative_bilstm_conformer.md)

---

## 1. Résumé exécutif

Phase 2 (BiLSTM CTC) atteint **WER 42.6% / CER 20.5%** sur Common Voice FR test, en plateau à epoch 47. L'écart avec Vosk small (WER 23.95%) est trop large pour publier de manière compétitive.

**Phase 3 propose le passage à une architecture Conformer-CTC** :

- **Même budget paramétrique** (~9 M params, ~36 MB)
- **Gain attendu** : 8-12 pp WER → cible **WER 30-33% en Run #5**, **22-25% en Run #6** (avec données étendues)
- **Coût** : ~2 sessions Kaggle T4 (~16-20 h GPU) pour Run #5
- **ROI** : le levier le plus efficace identifié, devant tuning KenLM (+3 pp) ou augmentation données seule (+5 pp)

**Demande** : autorisation de lancer Run #5 (Conformer from-scratch sur 586k) selon le plan ci-dessous.

---

## 2. Objectif

Réduire l'écart DeepVox ↔ Vosk small fr 0.22 sur Common Voice FR test, **tout en préservant le différenciateur Codec2 1200 bps** (input ultra-bas-débit, use case radio/satellite/DTN).

### 2.1 Cibles chiffrées

| Étape | WER cible | CER cible | Statut publication |
|---|---|---|---|
| Run #4 (état actuel) | 42.6% | 20.5% | Pas publiable seul |
| **Run #5 (Conformer from-scratch)** | **≤ 35%** | **≤ 17%** | Publiable comme blog technique |
| Run #6 (Conformer + data ext.) | ≤ 28% | ≤ 14% | Égalité Vosk → publiable model card HF |
| Run #7 (Conformer + Attention) | ≤ 25% | ≤ 12% | Compétitif avec wav2vec2-fr small |

### 2.2 Critères de succès Run #5

- **Critère go/no-go** : WER ≤ 38% sur dev set 29k samples (vs 42.6% actuel)
- **Stretch goal** : WER ≤ 35%
- **Échec** : WER > 40% → revenir à BiLSTM + autres optimisations (data, hyperparams)

---

## 3. Périmètre

### 3.1 Ce qui change

| Composant | Phase 2 | **Phase 3 Run #5** |
|---|---|---|
| Architecture | BiLSTM 3 couches | **Conformer-CTC 12 blocs** |
| Lib | `torch.nn.LSTM` | **`torchaudio.models.Conformer`** |
| Optimizer | AdamW lr=3e-4 | AdamW lr=1e-3 + warmup 1000 steps |
| Scheduler | ReduceLROnPlateau | Noam ou cosine warmup |
| Notebook | `02_phase2_asr_kaggle.ipynb` | **`03_phase3_conformer_kaggle.ipynb`** (nouveau) |
| Modèle source | `ctc_asr.py` | **`conformer_asr.py`** (nouveau) |

### 3.2 Ce qui reste identique

- Vocabulaire FR 49 caractères (`text.py`)
- Encodage Codec2 1200 bps en input (48 features × 25 fps)
- Loss CTC
- Pipeline de preprocessing local (`preprocess_for_kaggle.py`)
- Dataset pickle 586k (`deepvox_586k.pkl`)
- Évaluation greedy + KenLM (`evaluate_with_kenlm.py`)
- Métriques WER/CER

### 3.3 Ce qui n'est PAS dans le périmètre Run #5

- Données externes (MLS FR, VoxPopuli FR) → Run #6
- Attention decoder hybride → Run #7
- Quantization INT8 / déploiement mobile → Phase 4
- Streaming temps réel → Phase 4
- Self-supervised pre-training → Phase 5

---

## 4. Plan d'exécution

### 4.1 Livrables (5 fichiers)

1. **`docs/16_etude_comparative_bilstm_conformer.md`** ✅ Créé — étude technique détaillée
2. **`docs/17_proposition_phase3_conformer.md`** ✅ Ce document
3. **`src/deepvox/models/conformer_asr.py`** — implémentation Conformer-CTC
4. **`notebooks/03_phase3_conformer_kaggle.ipynb`** — notebook Kaggle pour Run #5
5. **`docs/18_retour_experience_phase3_run5.md`** — rétro post-Run #5

### 4.2 Étapes opérationnelles

| Étape | Durée | Validation |
|---|---|---|
| 1. Implémenter `conformer_asr.py` | 2 h | Test unitaire forward pass + count_parameters() |
| 2. Tester forward pass localement (CPU) | 1 h | Sortie shape (B, T, 49), pas de NaN |
| 3. Créer notebook Kaggle adapté | 2 h | Cellules : config, data load, train, eval greedy + KenLM |
| 4. **Lancer Run #5 sur Kaggle** | ~2 sessions × 10 h | Resume cross-session |
| 5. Évaluation greedy + KenLM | 1 h | WER/CER sur 29k test |
| 6. Rédiger rétro `18_*.md` | 2 h | Comparaison BiLSTM vs Conformer |

**Total effort humain** : ~8 h hors training. **Total GPU Kaggle** : 16-20 h.

### 4.3 Calendrier proposé

- **Semaine 1 (cette semaine)** : étapes 1-3 (code + notebook)
- **Semaine 2** : étapes 4-5 (training Run #5)
- **Semaine 3** : étape 6 (rétro) + décision Run #6

---

## 5. Configuration Run #5 (proposée)

```python
# notebooks/03_phase3_conformer_kaggle.ipynb — config initiale
RUN_NAME = "run5_conformer_586k"

# Architecture Conformer
CONFORMER_D_MODEL = 144
CONFORMER_NHEAD = 4
CONFORMER_NUM_LAYERS = 12
CONFORMER_FFN_DIM = 576
CONFORMER_CONV_KERNEL = 31
CONFORMER_DROPOUT = 0.1

# Training
MAX_SAMPLES = 586_000
MAX_EPOCHS = 30
BATCH_SIZE = 24            # ↓ vs BiLSTM (32) — attention plus VRAM-hungry
LEARNING_RATE = 1e-3       # ↑ vs BiLSTM (1e-4) — Transformer optimal
WARMUP_STEPS = 1000        # nouveau pour Conformer
PATIENCE = 7
WEIGHT_DECAY = 1e-2
GRAD_CLIP = 5.0

# Mixed precision (gain VRAM + vitesse sur T4)
USE_AMP = True
```

### 5.1 Justifications hyperparams

| Choix | Raison |
|---|---|
| `d_model=144` | Compromis taille (~9 M total) vs capacité. Plus bas (96) sous-paramètre, plus haut (192) gonfle au-delà de 12 M. |
| `num_layers=12` | Standard "small" pour Conformer (Gulati 2020 small=10, medium=16). 12 = bon middle ground. |
| `nhead=4` | `d_model / nhead = 36` (head dim acceptable). Plus de têtes (8) avec d=144 → heads de 18 = trop petits. |
| `conv_kernel=31` | Capture ~1.24 s de contexte local (31 frames × 40 ms). Standard Conformer. |
| `lr=1e-3` | LR Transformer optimal post-warmup. BiLSTM utilisait 3e-4 / 1e-4 (récurrent stable à plus bas LR). |
| `warmup=1000 steps` | Empêche divergence initiale de l'attention. |
| `batch=24` | T4 a 15.6 GB, attention quadratique en T (max ~300 frames) → safer que 32. À ajuster selon OOM. |
| `AMP=True` | bf16 → ~30% gain VRAM, ~20% gain vitesse, sans perte qualité notable sur ASR. |

---

## 6. Risques et mitigations

| Risque | Probabilité | Impact | Mitigation |
|---|---|---|---|
| OOM VRAM avec batch 24 | Moyenne | Run avorté | Tester batch 16, AMP activé, `torch.compile()` désactivé en cas |
| Convergence lente (>30 epochs) | Faible | +1 session Kaggle | MAX_EPOCHS=40 si besoin, scheduler agressif |
| WER ne s'améliore pas | Faible | Décision difficile | Critère go/no-go ≤ 38% — sinon retour BiLSTM + data augmentation |
| Gradients explosifs (attention) | Faible | Run avorté | gradient clipping 5.0 + warmup |
| Bug dans `Conformer` API torchaudio | Faible | 1-2 jours debug | Test unitaire avant lancement Kaggle |
| Pickle 586k incompatible | Très faible | Re-preprocess | Le format ne change pas (features Codec2 + char_ids) |

---

## 7. Communication post-Run #5

### Si Run #5 atteint WER ≤ 35% (succès attendu)

- Mettre à jour `docs/18_retour_experience_phase3_run5.md`
- Mettre à jour le tableau historique en tête du notebook
- Préparer un brouillon d'article blog "DeepVox Conformer : -10 pp WER sur Codec2"
- Lancer Run #6 (données étendues) la semaine suivante

### Si Run #5 atteint WER > 38% (échec)

- Rétro détaillée des causes (architecture sous-dimensionnée ? data ? hyperparams ?)
- Garder le code Conformer pour Phase 4 (avec données étendues)
- Pivoter sur tuning KenLM (alpha/beta + .arpa) + augmentation Codec2 (time mask, speed perturb)
- Probable plafond Codec2 1200 bps confirmé → revoir l'angle de publication

---

## 8. Décision attendue

**À valider** :

- [x] Étude comparative complète : `docs/16_etude_comparative_bilstm_conformer.md`
- [x] Périmètre clarifié : Run #5 only (pas Run #6/#7 pour l'instant)
- [ ] **GO sur étapes 1-3** (code + notebook) → 5-6 h de travail local
- [ ] **GO sur étape 4** (lancement Kaggle) → après validation forward pass

**Si validation** :
1. Création de `src/deepvox/models/conformer_asr.py`
2. Création de `notebooks/03_phase3_conformer_kaggle.ipynb`
3. Test forward pass local (CPU)
4. Push GitHub
5. Lancement Kaggle Run #5

---

## 9. Annexes

### 9.1 Tableau récapitulatif Phase 2 → Phase 3

| Run | Phase | Architecture | Corpus | WER (KenLM) | CER (KenLM) | Doc |
|---|---|---|---|---|---|---|
| #1 | 2 | BiLSTM | 20k | ~115% | 71% | `12_*.md` |
| #2 | 2 | BiLSTM | 80k | ~95% | 57% | `13_*.md` |
| #3 | 2 | BiLSTM | 300k | ~71% | 32% | `14_*.md` |
| #4 | 2 | BiLSTM finetune | 586k | **42.6%** | **20.5%** | `15_*.md` |
| **#5** | **3** | **Conformer** | **586k** | **30-33% (cible)** | **15-17% (cible)** | **`18_*.md` (à venir)** |
| #6 | 3 | Conformer | 586k + ext | 22-25% (cible) | 12-14% (cible) | `19_*.md` (à venir) |
| #7 | 3 | Conformer + Att | 586k + ext | ≤ 22% (cible) | ≤ 11% (cible) | `20_*.md` (à venir) |

### 9.2 Stack code post-Run #5

```
src/deepvox/models/
├── ctc_asr.py              ← BiLSTM (Phase 2, gardé pour reproductibilité)
├── conformer_asr.py        ← Conformer (Phase 3, nouveau) ★
└── phoneme_classifier.py   ← Phase 1

notebooks/
├── 01_exploration_pipeline.ipynb
├── 02_phase2_asr_kaggle.ipynb         ← BiLSTM (Phase 2)
└── 03_phase3_conformer_kaggle.ipynb   ← Conformer (Phase 3) ★

docs/
├── 15_retour_experience_phase2_run4.md  ← Phase 2 final
├── 16_etude_comparative_bilstm_conformer.md  ← Cette comparaison ★
├── 17_proposition_phase3_conformer.md   ← Cette proposition ★
└── 18_retour_experience_phase3_run5.md  ← À créer post-Run #5 ★
```

### 9.3 Quick reference — différences code BiLSTM ↔ Conformer

| | BiLSTM (`ctc_asr.py`) | Conformer (`conformer_asr.py`) |
|---|---|---|
| Forward signature | `forward(x)` | `forward(x, lengths)` |
| Output shape | `(B, T, V)` log-probs | `(B, T, V)` log-probs |
| Padding handling | Implicite (LSTM ignore zéros) | Explicite via `lengths` (mask attention) |
| Train mode setup | Standard | + warmup scheduler |
| AMP friendliness | OK fp32 | **Recommandé bf16** |
