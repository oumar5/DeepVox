# DeepVox — Suivi des tâches

## Phase 1 : Validation phonétique de Codec2 sur le français

### Setup environnement
- [ ] Créer et configurer l'environnement virtuel Python
- [ ] Installer Codec2 (binaire `c2enc` / `c2dec` fonctionnel)
- [ ] Installer Montreal Forced Aligner (MFA) + modèle/dictionnaire français

### Données
- [ ] Télécharger Common Voice French v21.0 (Kaggle)
- [ ] Resampling 16 kHz → 8 kHz
- [ ] Alignement forcé MFA → annotations phonétiques (TextGrid)
- [ ] Encodage Codec2 1200 bps de chaque énoncé
- [ ] Mapping temporel : bornes phonétiques → grille 40 ms Codec2
- [ ] Constituer les splits train / dev / test / test-OOD

### Code — modules
- [x] `src/deepvox/codec2/encoder.py` — wrapper Python autour de c2enc/c2dec
- [x] `src/deepvox/data/preprocess.py` — pipeline resampling + MFA + mapping grille
- [x] `src/deepvox/data/dataset.py` — Dataset PyTorch (frame_codec2, phonème)
- [x] `src/deepvox/models/phoneme_classifier.py` — BiLSTM 2 couches, hidden=256, 36 phonèmes
- [x] `src/deepvox/training/phase1.py` — boucle d'entraînement (Adam, lr=1e-3, early stopping)
- [x] `src/deepvox/eval/metrics.py` — PER, précision macro, matrice de confusion

### Expériences
- [ ] Condition A — Codec2 raw (48 features)
- [ ] Condition B — Codec2 + delta (96 features)
- [ ] Condition C — Spectrogramme mel baseline (80 features)
- [ ] Condition D — PCM brut baseline basse (320 features)

### Analyses complémentaires
- [ ] Ablation par champ Codec2 (LSP, pitch, énergie, voicing)
- [ ] Robustesse au bruit (bruit blanc/rose avant encodage)
- [ ] Comparaison mode 1200 vs 700C
- [ ] Robustesse aux variantes francophones (African Accented French)
- [ ] Analyse spécifique des voyelles nasales

### Livrables
- [ ] Script reproductible `phase1_phoneme_classification.py`
- [ ] Rapport `phase1_results.md` (tableaux, matrices, conclusions)
- [ ] Modèles entraînés versionnés
- [ ] Décision GO / NO-GO pour Phase 2
