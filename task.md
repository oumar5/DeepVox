# DeepVox — Suivi des tâches

## Phase 1 : Validation phonétique de Codec2 sur le français

### Setup environnement

- [x] Créer et configurer l'environnement virtuel Python (Python 3.12, venv)
- [x] Installer Codec2 (binaire `c2enc` / `c2dec` fonctionnel + pycodec2)
- [x] Installer Montreal Forced Aligner (MFA) + modèle/dictionnaire français
- [x] Installer Miniconda + env conda `mfa`

### Données

- [x] Télécharger Common Voice French v21.0 (Kaggle) — 840k clips
- [x] Resampling 16 kHz → 8 kHz (5000 échantillons de test préparés)
- [x] Alignement forcé MFA → annotations phonétiques (4922 TextGrid)
- [x] Encodage Codec2 1200 bps (validé sur 1 fichier)
- [x] Mapping temporel : bornes phonétiques → grille 40 ms Codec2
- [ ] Constituer les splits train / dev / test / test-OOD — reporté (pas nécessaire pour valider H0)

### Code — modules

- [x] `src/deepvox/codec2/encoder.py` — wrapper Python autour de c2enc/c2dec (pycodec2 + CLI)
- [x] `src/deepvox/data/preprocess.py` — pipeline resampling + MFA + mapping grille (44 phonèmes IPA)
- [x] `src/deepvox/data/dataset.py` — Dataset PyTorch 4 conditions (CODEC2_RAW, CODEC2_DELTA, MEL_SPEC, PCM_RAW)
- [x] `src/deepvox/models/phoneme_classifier.py` — BiLSTM 2 couches, hidden=256, 44 phonèmes IPA
- [x] `src/deepvox/training/phase1.py` — boucle d'entraînement (AdamW, lr=2e-3, early stopping, ReduceLROnPlateau)
- [x] `src/deepvox/eval/metrics.py` — PER, Top-K accuracy, F1 macro/weighted, accuracy par groupe IPA/voisement

### Code — scripts

- [x] `scripts/prepare_data.py` — preprocessing Common Voice → WAV 8kHz + .lab
- [x] `scripts/phase1_phoneme_classification.py` — pipeline complet 1 condition
- [x] `scripts/run_all_conditions.py` — lance les 4 conditions + rapport comparatif
- [x] `scripts/ablation_codec2_fields.py` — ablation LSP/pitch/énergie
- [x] `scripts/noise_robustness.py` — robustesse bruit blanc/rose à différents SNR
- [x] `scripts/evaluate_checkpoint.py` — évaluation d'un checkpoint sauvegardé (créé après crash run #4)

### Code — qualité

- [x] 24 tests unitaires (encoder, modèle, métriques) — all pass
- [x] Lint ruff — zero warnings
- [x] Notebook d'exploration `notebooks/01_exploration_pipeline.ipynb`
- [x] `Makefile` — commandes simplifiées
- [x] Correction `.gitignore` (`data/` → `/data/` pour ne pas ignorer `src/deepvox/data/`)
- [x] Migration tqdm → `tqdm.auto` (auto-détection notebook/terminal)
- [x] Migration Adam → AdamW (weight_decay=1e-2)
- [x] Migration SAMPA → IPA (36 → 44 phonèmes, MFA french_mfa utilise IPA)
- [x] Remplacement parseur TextGrid naïf par praatio

### Expériences

- [x] Condition A — Codec2 raw (48 features) — validée sur 4 runs
- [x] Condition B — Codec2 + delta (96 features) — échec (PER=65%, pire que A, abandonnée)
- [ ] Condition C — Spectrogramme mel baseline (80 features) — reportée (comparatif article)
- [ ] Condition D — PCM brut baseline basse (320 features) — reportée (comparatif article)

### Runs de scaling

- [x] Run #1 — Condition A, 5k fichiers, 7 epochs — PER=69.2% — `docs/08_retour_experience_phase1_run1.md`
- [x] Run #2 — Condition A, 20k fichiers, 17 epochs — PER=62.8%, voisement=90.9% — `docs/09_retour_experience_phase1_run2.md`
- [x] Run #3 — Condition B, 20k fichiers — PER=65.0%, échec — `docs/10_retour_experience_phase1_run3_conditionB.md`
- [x] Run #4 — Condition A, 80k fichiers, 4 epochs (interrompu reboot) — PER=56.0%, Top-5=80.7%, voisement=93.0% — `docs/11_retour_experience_phase1_run4.md`
- [ ] Run #5 — Corpus complet (750k) — reporté, Phase 2 prioritaire

### Analyses complémentaires

- [ ] Ablation par champ Codec2 (LSP, pitch, énergie, voicing) — reportée
- [ ] Robustesse au bruit (bruit blanc/rose avant encodage) — reportée
- [ ] Comparaison mode 1200 vs 700C — reportée
- [ ] Robustesse aux variantes francophones (African Accented French) — reportée
- [ ] Analyse spécifique des voyelles nasales — reportée

### Livrables

- [x] Script reproductible `phase1_phoneme_classification.py`
- [x] 4 retours d'expérience détaillés (docs 08-11)
- [x] Décision GO / NO-GO pour Phase 2 — **GO** (voisement 93%, top-5 80%, tendance log-linéaire claire)
- [ ] Rapport final `phase1_results.md` (tableaux, matrices, conclusions) — à faire pour l'article
- [ ] Conditions C et D pour comparatif article — reportées

---

## Phase 2 : ASR directe — Codec2 → texte français

### Setup

- [x] Pivot vers Kaggle (T4 GPU) — CTC loss non supportée sur MPS, OOM sur Apple Silicon
- [x] Compte Kaggle configuré, dataset Common Voice FR ajouté en input

### Code — modules

- [x] `src/deepvox/data/text.py` — tokenizer caractères français (49 classes : blank + unk + 47 chars)
- [x] `src/deepvox/data/ctc_dataset.py` — Dataset CTC + `ctc_collate_fn()` pour padding variable
- [x] `src/deepvox/models/ctc_asr.py` — BiLSTM 3 couches, embed=256, hidden=384, 9.1M params, `greedy_decode()`
- [x] `src/deepvox/training/phase2_asr.py` — boucle CTC (gradient clipping 5.0, CER-based early stopping)
- [x] `src/deepvox/eval/wer.py` — WER, CER (Levenshtein), `format_asr_report()`

### Code — scripts & notebooks

- [x] `scripts/phase2_asr.py` — CLI Phase 2 (local, non utilisé — MPS incompatible)
- [x] `notebooks/02_phase2_asr_kaggle.ipynb` — notebook Kaggle GPU T4, paramétré par cellule de config
- [x] Cellule de configuration centralisée (`RUN_NAME`, `MAX_SAMPLES`, `MAX_EPOCHS`, etc.)
- [x] Système de resume (checkpoint complet : model + optimizer + scheduler + history)
- [x] `gc.collect()` + `torch.cuda.empty_cache()` à chaque epoch
- [x] `num_workers=0` pour Kaggle (évite erreur multiprocessing)
- [x] Indexation MP3 par `subprocess.run(['find', ...])` (gère espaces dans chemins)

### Tests

- [x] `tests/test_text.py` — 15 tests tokenizer
- [x] `tests/test_wer.py` — 11 tests WER/CER

### Runs ASR

- [x] Run #1 — 20k, Kaggle T4, 25 epochs (early stop) — **WER=115.5%, CER=71.2%** — `docs/12_retour_experience_phase2_run1.md`
- [x] Run #2 — 80k, Kaggle T4, 50 epochs (complet) — **WER=95.0%, CER=56.9%** (−14.3 pp) — `docs/13_retour_experience_phase2_run2.md`
- [ ] Run #3 — 300k, Kaggle T4, 40 epochs (2 sessions avec resume) — **à lancer**
- [ ] Run #4 — KenLM beam search — après run #3 si CER < 40%

### Livrables

- [x] Retour d'expérience `docs/12_retour_experience_phase2_run1.md`
- [x] Retour d'expérience `docs/13_retour_experience_phase2_run2.md`
- [x] Retour d'expérience `docs/14_retour_experience_phase2_run3.md` (en cours, session 1 documentée)
- [ ] Rapport final Phase 2
- [ ] Décision sur architecture (BiLSTM vs Conformer)

---

## Phases futures (non démarrées)

### Phase 3 : Évaluation multilingue

- [ ] Tester sur anglais (Common Voice EN)
- [ ] Tester sur arabe (Common Voice AR)
- [ ] Comparaison CER cross-lingue

### Phase 4 : Traduction vocale

- [ ] Pipeline Codec2 → texte source → texte cible
- [ ] Intégration modèle de traduction (MarianMT ou similaire)
- [ ] Métrique BLEU

### Phase 5 : Optimisation et compression

- [ ] Quantization INT8 / pruning
- [ ] Objectif : modèle < 200 MB total pour 3 langues
- [ ] Benchmark latence sur smartphone

### Phase 6 : Déploiement

- [ ] Export ONNX / TFLite
- [ ] Application mobile prototype
- [ ] Tests utilisateurs

---

## Résumé des résultats clés

| Phase | Run | Corpus | Métrique | Résultat |
|---|---|---|---|---|
| 1 | #1 | 5k | PER | 69.2% |
| 1 | #2 | 20k | PER | 62.8% |
| 1 | #3 | 20k (delta) | PER | 65.0% (échec) |
| 1 | #4 | 80k | PER | 56.0%, Top-5=80.7%, voisement=93% |
| 2 | #1 | 20k | CER | 71.2% |
| 2 | #2 | 80k | CER | **56.9%**, WER=95.0% |
| 2 | #3 | 300k | CER | en cours |

**Tendance** : chaque ×4 de données = −6-7 pp PER (Phase 1) / −14 pp CER (Phase 2)
