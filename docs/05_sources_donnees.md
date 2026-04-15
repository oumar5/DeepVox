# 05 — Sources de données d'entraînement

Vue d'ensemble des corpus publics utilisables pour chaque phase. **DeepVox priorise le français** comme langue principale de validation et d'entraînement, en cohérence avec la cible SmsVox (utilisateurs francophones, notamment des régions à faible connectivité).

Critères : **licence**, **langues couvertes**, **alignements parallèles** (essentiels pour la traduction), **téléchargement effectif** (URL et taille), **disponibilité Kaggle** quand pertinent.

---

## Section A — Données francophones (priorité 1)

### A.1 — Common Voice French (le pilier)

**Mozilla Common Voice** est le corpus francophone open le plus volumineux. C'est notre dataset principal pour les Phases 1 et 2.

| Source | URL | Volume | Avantage |
|---|---|---|---|
| Officielle Mozilla | https://commonvoice.mozilla.org/fr/datasets | ~1000 h validées | Toujours à jour, version la plus récente |
| Kaggle v21.0 (mars 2025) | https://www.kaggle.com/datasets/fredrelec/common-voice-french-21-0-2025 | Snapshot v21.0 | Téléchargement direct sans compte Mozilla, pratique pour notebooks Kaggle |
| Kaggle (rahulbhalley) | https://www.kaggle.com/datasets/rahulbhalley/common-voice-french | Snapshot ancien | Backup |
| Kaggle (olmatz) | https://www.kaggle.com/datasets/olmatz/commonvoicefr | Snapshot ancien | Backup |
| Kaggle (Mozilla officiel) | https://www.kaggle.com/datasets/mozillaorg/common-voice | Multilingue | Inclut FR + autres langues |

- **Licence :** CC0 (domaine public)
- **Format :** MP3 16 kHz + transcriptions CSV
- **Locuteurs :** milliers, accents variés (France, Belgique, Suisse, Canada, Afrique francophone)
- **Annotations phonétiques :** non fournies, à générer via Montreal Forced Aligner

### A.2 — MLS French (Multilingual LibriSpeech)

**Audiobooks LibriVox français**, qualité studio, alignement texte-audio fourni.

- URL : https://www.openslr.org/94
- Volume : ~1100 h
- Licence : CC BY 4.0
- Avantage : qualité audio supérieure à Common Voice
- Usage : fine-tuning Phase 2, validation propre

### A.3 — CSS10 French (TTS mono-locuteur)

**Voix unique** francophone, idéale pour entraînement TTS de base.

| Source | URL | Volume |
|---|---|---|
| Officielle | https://github.com/Kyubyong/css10 | ~10 h |
| Kaggle | https://www.kaggle.com/datasets/bryanpark/french-single-speaker-speech-dataset | ~10 h |

- Licence : domaine public
- Usage : Phase 3 (TTS), point de départ minimal

### A.4 — African Accented French (cible SmsVox)

**Variantes francophones africaines** — essentiel pour valider la robustesse sur l'audience SmsVox réelle.

- URL : http://www.openslr.org/57/
- Volume : ~22 h
- Locuteurs : Sénégal, Maroc, RDC, Côte d'Ivoire, Tunisie, Algérie
- Licence : CC BY-SA 4.0
- Usage : test out-of-distribution Phase 1, fine-tuning Phase 2

### A.5 — Datasets Kaggle francophones supplémentaires

| Dataset | URL Kaggle | Type |
|---|---|---|
| **French Speech Recognition Dataset** (unidpro) | https://www.kaggle.com/datasets/unidpro/french-speech-recognition-dataset | ASR français commercial-grade |
| **French Spontaneous Dialogue** (nexdatafrank) | https://www.kaggle.com/datasets/nexdatafrank/french-spontaneous-dialogue-speech-dataset | Parole spontanée (≠ parole lue) |

### A.6 — Sources francophones additionnelles (hors Kaggle)

- **SynPaFlex** (français expressif) — https://www.ortolang.fr/market/corpora/synpaflex-corpus — ~87 h
- **VoxForge French** (historique, gratuit) — http://www.voxforge.org/fr — ~30 h
- **PFC** (Phonologie du Français Contemporain) — https://www.projet-pfc.net/ — variable, dialectologique
- **ESLO** (Enquêtes SocioLinguistiques à Orléans) — https://eslo.huma-num.fr/ — variable
- **OFROM** (Oral Français de Romandie) — https://ofrom.unine.ch/ — variable

---

## Section B — Annotations phonétiques pour le français

Aucun des datasets francophones ci-dessus ne fournit d'annotations phonétiques manuelles. Pour la Phase 1, on génère ces annotations automatiquement.

### Outil principal : Montreal Forced Aligner (MFA)

- URL : https://montreal-forced-aligner.readthedocs.io/
- Modèle français pré-entraîné : `french_mfa`
- Lexique : `french_mfa.dict`
- Sortie : TextGrid avec bornes phonétiques (~10 ms de précision)
- Licence : MIT

### Alternative : eSpeak NG comme G2P

- Phonémiseur grapheme-to-phoneme open-source
- Génère uniquement la séquence phonémique attendue (pas les bornes temporelles)
- Utile pour vérifier la cohérence avec MFA

### Inventaire phonémique français

36 phonèmes en SAMPA-FR : 16 voyelles (4 nasales), 17 consonnes, 3 semi-voyelles. Document de référence : https://www.phon.ucl.ac.uk/home/sampa/french.htm

---

## Section C — Données pour la traduction (Phases 4-6)

Pour la traduction (FR↔EN comme paire principale), il faut des **paires parallèles** (audio FR + texte EN ou audio EN, etc.).

### C.1 — MuST-C v1.2 (référence académique)

- URL : https://ict.fbk.eu/must-c/
- Paires : EN↔FR disponible (~400 h)
- Format : audio WAV + alignement YAML + textes parallèles
- Licence : CC BY-NC-ND 4.0 (attention : usage commercial restreint)
- Usage : Phase 5 S2TT principal

### C.2 — CoVoST 2

- URL : https://github.com/facebookresearch/covost
- Paires : FR→EN et EN→FR disponibles
- Volume total : 2880 h
- Licence : CC0 (utilisation libre)
- Avantage : dérivé de Common Voice, qualité hétérogène mais grand volume

### C.3 — CVSS (S2ST de référence)

- URL : https://github.com/google-research-datasets/cvss
- Audio source réel + audio cible synthétisé par TTS de qualité
- Inclut FR→EN
- Licence : CC BY 4.0
- Usage : Phase 5 S2ST principal

### C.4 — Europarl-ST

- URL : https://www.mllp.upv.es/europarl-st/
- Paires bidirectionnelles entre 9 langues européennes (dont FR)
- Volume : ~1300 h total
- Licence : CC BY 4.0
- Usage : domaine politique/formel, complément de MuST-C

---

## Section D — Modèles MT pré-entraînés (à réutiliser, pas à entraîner)

Pour la Phase 4 (cascade baseline), on n'entraîne pas de modèle MT — on utilise des modèles existants distillés :

- **NLLB-200 distillé 600M** (Meta) — https://huggingface.co/facebook/nllb-200-distilled-600M — 200 langues
- **M2M-100** (Meta) — alternative, 100 langues
- **OPUS-MT FR-EN** (Helsinki NLP) — https://huggingface.co/Helsinki-NLP/opus-mt-fr-en — modèle bilingue compact

---

## Section E — Évaluation multilingue (Phase 6)

- **FLEURS** — https://huggingface.co/datasets/google/fleurs — 102 langues dont FR, ~12 h/langue, audio + texte source + texte cible
- Indispensable pour positionner DeepVox dans le paysage des modèles multilingues

---

## Tableau récapitulatif (priorisé)

| Corpus | Tâches | Langues | Heures | Licence | Source principale |
|---|---|---|---|---|---|
| **Common Voice FR v21.0** | Phase 1, Phase 2 | FR | ~1000 h | CC0 | Kaggle + Mozilla |
| **MLS French** | Phase 2 | FR | ~1100 h | CC BY 4.0 | OpenSLR |
| **CSS10 French** | Phase 3 | FR | ~10 h | Domaine public | Kaggle + GitHub |
| **African Accented French** | Phase 1 OOD, Phase 2 | FR (variants) | ~22 h | CC BY-SA 4.0 | OpenSLR 57 |
| **MuST-C EN-FR** | Phase 5 S2TT | EN↔FR | ~400 h | CC BY-NC-ND 4.0 | FBK |
| **CoVoST 2** | Phase 5 S2TT | FR↔EN | 2880 h total | CC0 | Meta GitHub |
| **CVSS** | Phase 5 S2ST | FR→EN | 1900 h total | CC BY 4.0 | Google |
| **FLEURS** | Phase 6 | 102 langues | ~12 h/langue | CC BY 4.0 | HuggingFace |
| **NLLB-200** (modèle) | Phase 4 baseline | 200 langues | — | CC BY-NC | HuggingFace |

---

## Stratégie de téléchargement Kaggle

L'utilisation de Kaggle apporte plusieurs avantages opérationnels :

1. **Pas de compte Mozilla requis** pour Common Voice (formulaire sinon obligatoire)
2. **Notebooks Kaggle gratuits** avec GPU pour exploration (jusqu'à 30 h/semaine)
3. **Téléchargement programmatique** via Kaggle CLI :

```bash
# Installation
pip install kaggle

# Configuration : placer kaggle.json dans ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Téléchargement Common Voice FR v21.0
kaggle datasets download -d fredrelec/common-voice-french-21-0-2025

# Téléchargement CSS10 French
kaggle datasets download -d bryanpark/french-single-speaker-speech-dataset

# Téléchargement African Accented French (depuis OpenSLR, pas Kaggle)
wget http://www.openslr.org/resources/57/African_Accented_French.tar.gz
```

Un script d'orchestration sera fourni dans `scripts/download_datasets.py` une fois la Phase 1 démarrée.

---

## Considérations légales et éthiques

- **CC BY-NC-ND** (MuST-C) : interdit usage commercial et dérivés. Usage académique uniquement. Si DeepVox vise un déploiement commercial via SmsVox, **éviter MuST-C** ou cantonner son usage à l'évaluation
- **CC0 / CC BY / CC BY-SA** (Common Voice, CoVoST, CVSS, MLS, African Accented FR, Europarl-ST, FLEURS) : aucun problème, même usage commercial
- **NLLB-200** : CC BY-NC pour le modèle Meta (non-commercial). Pour usage commercial, considérer M2M-100 ou OPUS-MT
- **Voice cloning** (Phase 6) : risques deepfake vocal. Watermarking systématique (AudioSeal de Meta), limitation aux mêmes locuteurs
- **Données africaines** : importance éthique de retourner des bénéfices aux communautés représentées (collaborations, attribution claire, modèles ouverts pour ces langues)

---

## Pré-traitement commun à tout le projet

Pour toutes les phases, un script centralisé doit produire la version "Codec2-encoded" de chaque corpus :

```
[corpus_original]/
  audio/*.wav            (44.1 ou 16 kHz)

→ pré-traitement →

[corpus_codec2]/
  audio_8k/*.wav         (8 kHz mono)
  codec2_1200/*.c2       (frames Codec2 mode 1200)
  codec2_700c/*.c2       (frames Codec2 mode 700C)
  metadata.json          (alignements MFA, transcriptions, traductions)
```

Ce pipeline réutilise les binaires `c2enc` / `c2dec` du projet codec2 (https://github.com/drowe67/codec2), installés via `brew install codec2` ou `apt install codec2` (voir `src/deepvox/codec2/` pour les wrappers Python subprocess).

---

## Estimation des volumes

| Phase | Données nécessaires | Stockage |
|---|---|---|
| 1 | Common Voice FR + MLS French + African Accented + variantes encodées | ~100 GB |
| 2 | + autres langues Common Voice si extension multilingue | ~200 GB |
| 3 | CSS10 French + SynPaFlex + variantes TTS | ~30 GB |
| 5 | MuST-C EN-FR + CVSS FR-EN + CoVoST 2 | ~400 GB |
| 6 | + FLEURS pour évaluation | +50 GB |

**Total recommandé : 800 GB - 1 TB de stockage SSD** pour confort de travail incluant variantes encodées Codec2 et checkpoints intermédiaires.
