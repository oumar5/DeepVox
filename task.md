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
- [x] Condition B — Codec2 + delta (96 features) — échec (PER=65%, pire que A, abandonnée)
- [ ] Condition C — Spectrogramme mel baseline (80 features)
- [ ] Condition D — PCM brut baseline basse (320 features)

### Runs de scaling

- [x] Run #1 — Condition A, 5k fichiers (PER=69.2%)
- [x] Run #2 — Condition A, 20k fichiers (PER=62.8%)
- [x] Run #3 — Condition B, 20k fichiers — échec (PER=65%, abandonnée)
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
- [x] Décision GO / NO-GO pour Phase 2 — **GO** (voisement 93%, top-5 80%, tendance claire)

## Phase 2 : ASR directe — Codec2 → texte français

### Code — modules

- [x] `src/deepvox/data/text.py` — tokenizer caractères français (49 classes)
- [x] `src/deepvox/data/ctc_dataset.py` — Dataset CTC + collate_fn
- [x] `src/deepvox/models/ctc_asr.py` — BiLSTM 3 couches, hidden=384, CTC (9.1M params)
- [x] `src/deepvox/training/phase2_asr.py` — boucle entraînement CTC
- [x] `src/deepvox/eval/wer.py` — WER, CER (Levenshtein)

### Code — scripts & notebooks

- [x] `scripts/phase2_asr.py` — CLI Phase 2 (local)
- [x] `notebooks/02_phase2_asr_kaggle.ipynb` — notebook Kaggle GPU T4

### Tests

- [x] `tests/test_text.py` — 15 tests tokenizer
- [x] `tests/test_wer.py` — 11 tests WER/CER

### Runs ASR

- [x] Run #1 — 20k fichiers, Kaggle T4, 25 epochs (early stop) — **WER=115.5%, CER=71.2%**
- [ ] Run #2 — 80k fichiers, Kaggle T4 — à lancer
- [ ] Run #3 — 80k + KenLM beam search — après run #2 si CER < 50%

### Livrables

- [x] Retour d'expérience `docs/12_retour_experience_phase2_run1.md`
- [ ] Rapport final Phase 2
- [ ] Décision sur architecture (BiLSTM vs Conformer)
