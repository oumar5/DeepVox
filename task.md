# DeepVox — Suivi des tâches

## Phase 1 : Validation phonétique de Codec2 sur le français

### Setup environnement

- [x] Créer et configurer l'environnement virtuel Python
- [x] Installer Codec2 (binaire `c2enc` / `c2dec` fonctionnel + pycodec2)
- [x] Installer Montreal Forced Aligner (MFA) + modèle/dictionnaire français
- [x] Installer Miniconda + env conda `mfa`

### Données

- [x] Télécharger Common Voice French v21.0 (Kaggle) — 840k clips
- [x] Resampling 16 kHz → 8 kHz (5000 échantillons de test préparés)
- [x] Alignement forcé MFA → annotations phonétiques (4922 TextGrid)
- [x] Encodage Codec2 1200 bps (validé sur 1 fichier)
- [x] Mapping temporel : bornes phonétiques → grille 40 ms Codec2
- [ ] Constituer les splits train / dev / test / test-OOD

### Code — modules

- [x] `src/deepvox/codec2/encoder.py` — wrapper Python autour de c2enc/c2dec
- [x] `src/deepvox/data/preprocess.py` — pipeline resampling + MFA + mapping grille
- [x] `src/deepvox/data/dataset.py` — Dataset PyTorch (frame_codec2, phonème)
- [x] `src/deepvox/models/phoneme_classifier.py` — BiLSTM 2 couches, hidden=256, 36 phonèmes
- [x] `src/deepvox/training/phase1.py` — boucle d'entraînement (Adam, lr=1e-3, early stopping)
- [x] `src/deepvox/eval/metrics.py` — PER, précision macro, matrice de confusion

### Code — scripts

- [x] `scripts/prepare_data.py` — preprocessing Common Voice → WAV 8kHz + .lab
- [x] `scripts/phase1_phoneme_classification.py` — pipeline complet 1 condition
- [x] `scripts/run_all_conditions.py` — lance les 4 conditions + rapport comparatif
- [x] `scripts/ablation_codec2_fields.py` — ablation LSP/pitch/énergie
- [x] `scripts/noise_robustness.py` — robustesse bruit blanc/rose à différents SNR

### Code — qualité

- [x] 24 tests unitaires (encoder, modèle, métriques) — all pass
- [x] Lint ruff — zero warnings
- [x] Notebook d'exploration `notebooks/01_exploration_pipeline.ipynb`
- [x] `Makefile` — commandes simplifiées

### Expériences

- [x] Condition A — Codec2 raw (48 features) — run #1 (5k, PER=69.2%) + run #2 (20k, PER=62.8%)
- [ ] Condition B — Codec2 + delta (96 features) — run en cours sur 20k
- [ ] Condition C — Spectrogramme mel baseline (80 features)
- [ ] Condition D — PCM brut baseline basse (320 features)

### Runs de scaling

- [x] Run #1 — Condition A, 5k fichiers (PER=69.2%)
- [x] Run #2 — Condition A, 20k fichiers (PER=62.8%)
- [ ] Run #3 — Condition B, 20k fichiers — en cours
- [x] Run #4 — Condition A, 80k fichiers (PER=56.0%, Top-5=80.7%, interrompu epoch 4)
- [ ] Run #5 — Corpus complet (750k) — reporté, Phase 2 prioritaire

### Analyses complémentaires

- [ ] Ablation par champ Codec2 (LSP, pitch, énergie, voicing)
- [ ] Robustesse au bruit (bruit blanc/rose avant encodage)
- [ ] Comparaison mode 1200 vs 700C
- [ ] Robustesse aux variantes francophones (African Accented French)
- [ ] Analyse spécifique des voyelles nasales

### Livrables

- [x] Script reproductible `phase1_phoneme_classification.py`
- [ ] Rapport `phase1_results.md` (tableaux, matrices, conclusions)
- [ ] Modèles entraînés versionnés
- [ ] Décision GO / NO-GO pour Phase 2
