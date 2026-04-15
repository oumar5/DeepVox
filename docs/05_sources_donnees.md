# 05 — Sources de données d'entraînement

Vue d'ensemble des corpus publics utilisables pour chaque phase. Une attention particulière est portée à : **licence**, **langues couvertes**, **alignements parallèles** (essentiels pour la traduction), et **téléchargement effectif** (URL et taille).

## Tableau synthétique

| Corpus | Tâches | Langues | Heures | Licence | Alignement parallèle |
|---|---|---|---|---|---|
| TIMIT | Phonétique (Phase 1) | EN | 5 h | LDC payant | Non |
| LibriSpeech | ASR (Phase 2) | EN | 1000 h | CC BY 4.0 | Non |
| Common Voice | ASR (Phase 2) | 100+ | 30 000 h+ | CC0 | Non |
| LJSpeech | TTS (Phase 3) | EN | 24 h | Domaine public | Non |
| MAILABS | TTS (Phase 3) | 9 langues | ~1000 h | BSD-like | Non |
| **MuST-C** | S2TT (Phase 5) | EN ↔ 14 langues | ~400 h/paire | CC BY-NC-ND 4.0 | **Oui (3-way: audio + texte src + texte tgt)** |
| **CoVoST 2** | S2TT (Phase 5) | 21 → EN, EN → 15 | 2880 h | CC0 | **Oui** |
| **Europarl-ST** | S2TT (Phase 5) | 9 langues européennes (toutes paires) | 1300 h | CC BY 4.0 | **Oui** |
| **CVSS** | S2ST (Phase 5) | 21 → EN | 1900 h | CC BY 4.0 | **Oui (audio src + audio tgt synthétisé)** |
| **VoxPopuli** | ASR + S2TT | 23 langues | 100 000 h | CC0 | **Partiel** |
| FLEURS | Évaluation multilingue | 102 | ~12 h/langue | CC BY 4.0 | **Oui (3-way)** |
| NLLB-200 (texte) | MT (Phase 4 baseline) | 200 langues | — | CC BY-NC | **Oui (texte parallèle)** |

## Détails par phase

### Phase 1 — Validation phonétique

**Recommandé : TIMIT** (référence de la littérature, comparaisons faciles).

**Alternatives gratuites :**
- **Buckeye Corpus** (anglais conversationnel, ~38 h, licence académique gratuite) — http://buckeyecorpus.osu.edu/
- **L2-ARCTIC** (anglais L2 non-natif, ~27 h, CC BY 4.0) — https://psi.engr.tamu.edu/l2-arctic-corpus/
- **CMU ARCTIC** (anglais lu, ~7 locuteurs, gratuit) — http://www.festvox.org/cmu_arctic/

Pour le **français** : **BREF120** (LDC payant) ou portion française de **Common Voice** avec alignement forcé via **Montreal Forced Aligner** (gratuit).

### Phase 2 — ASR depuis frames Codec2

**Pré-entraînement multilingue : Common Voice** — https://commonvoice.mozilla.org/datasets
- Téléchargement par langue, format MP3 + transcription CSV
- Avantages : énorme volume, multilingue, gratuit, CC0
- Inconvénients : qualité variable (crowdsourcé), accents très divers (peut être un atout)

**Fine-tuning anglais propre : LibriSpeech** — https://www.openslr.org/12
- 1000 h d'audiobooks lus, qualité studio
- Splits standards : train-clean-100, train-clean-360, train-other-500, dev-clean, test-clean

**Français : MLS (Multilingual LibriSpeech)** — https://www.openslr.org/94
- ~1100 h français + 7 autres langues, format compatible LibriSpeech

### Phase 3 — TTS texte → frames Codec2

**Anglais mono-locuteur : LJSpeech** — https://keithito.com/LJ-Speech-Dataset/
- 24 h, une seule voix féminine, qualité audiobook, alignement texte-audio fourni

**Multilingue mono-locuteur par langue : M-AILABS Speech Dataset**
- Anglais, allemand, espagnol, italien, ukrainien, russe, polonais, français, néerlandais
- ~1000 h total, voix variées par langue

**Français : SynPaFlex** ou **CSS10 français** — voix unique, ~10 h, gratuit

**Multi-locuteur (pour voice cloning Phase 6) : VCTK** — https://datashare.ed.ac.uk/handle/10283/3443
- 110 locuteurs anglais natifs, ~44 h, accents variés

### Phase 4 — Cascade baseline (MT texte)

**Modèles MT pré-entraînés à réutiliser, pas à entraîner :**
- **NLLB-200 (Meta)** — 200 langues, distillé en versions 600M / 1.3B / 3.3B / 54.5B paramètres
  - https://huggingface.co/facebook/nllb-200-distilled-600M
- **M2M-100 (Meta)** — 100 langues, alternative
- **OPUS-MT (Helsinki NLP)** — modèles bilingues légers spécialisés par paire

### Phase 5 — End-to-end S2TT et S2ST

#### Pour S2TT (parole source → texte cible)

**MuST-C v1.2** — https://ict.fbk.eu/must-c/
- Conférences TED, audio EN + transcription EN + traduction dans 14 langues cibles
- ~400 h par paire EN→{DE, ES, FR, IT, NL, PT, RO, RU, ZH, JA, AR, FA, TR, VI}
- Format : audio WAV + alignement YAML + textes parallèles
- Le dataset le plus utilisé en S2TT académique

**CoVoST 2** — https://github.com/facebookresearch/covost
- Dérivé de Common Voice, 21 langues vers anglais + anglais vers 15 langues
- Total 2880 h, gratuit (CC0)
- Inclut les paires non-anglaises grâce à pivot : utile pour évaluer en zero-shot

**Europarl-ST** — https://www.mllp.upv.es/europarl-st/
- Débats du Parlement européen, 9 langues, toutes paires bidirectionnelles (72 paires)
- ~1300 h total, langage formel/politique

#### Pour S2ST (parole source → parole cible)

**CVSS (Common Voice-based Speech-to-Speech)** — https://github.com/google-research-datasets/cvss
- 21 langues vers anglais, audio cible synthétisé par TTS de qualité
- 1900 h, CC BY 4.0
- Le dataset de référence pour S2ST end-to-end

**VoxPopuli (subset transcrit + traduit)** — https://github.com/facebookresearch/voxpopuli
- 100 000 h non transcrites + sous-ensembles annotés en ASR et ST
- 23 langues européennes, gratuit

### Phase 6 — Évaluation multilingue large

**FLEURS (Few-shot Learning Evaluation of Universal Representations of Speech)** — https://huggingface.co/datasets/google/fleurs
- 102 langues, ~12 h par langue
- Audio + texte source + texte cible (toutes paires) + transcription phonétique
- Conçu spécifiquement pour benchmark multilingue zero-shot/few-shot
- Indispensable pour positionner les résultats dans la littérature

## Données manquantes ou à constituer

Aucun corpus existant ne fournit directement des **paires de frames Codec2 alignées entre langues**. Il faudra :

1. Encoder en Codec2 les corpus audio sélectionnés (étape de préparation)
2. Pour S2ST : utiliser CVSS (audio cible déjà synthétisé) puis encoder les deux côtés en Codec2
3. Pour les langues sous-ressourcées non couvertes par CVSS : générer de l'audio cible via TTS multilingue (XTTS-v2, MMS) — accepté en recherche, à documenter clairement comme "synthetic target"

## Considérations légales et éthiques

- **CC BY-NC-ND** (MuST-C) : interdit commercial et dérivés. Utilisation académique OK, mais publication de modèles dérivés dans un produit commercial = problème
- **CC0 / CC BY** (CoVoST, Common Voice, Europarl-ST, FLEURS) : aucun problème, même usage commercial
- **TIMIT** (LDC) : licence individuelle, ne pas redistribuer le corpus brut
- **Voice cloning** : risques d'usage malveillant (deepfake vocal). Toute publication doit inclure une section éthique et envisager un watermarking audio (cf. AudioSeal de Meta)

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
  metadata.json          (alignements, transcriptions, traductions)
```

Ce pipeline peut réutiliser les binaires `c2enc` / `c2dec` du projet codec2 (https://github.com/drowe67/codec2) ou la JNI déjà intégrée dans SmsVox.

## Estimation des volumes

| Phase | Données nécessaires | Stockage |
|---|---|---|
| 1 | TIMIT ou équivalent | < 50 GB |
| 2 | LibriSpeech + Common Voice (1 langue) | ~200 GB |
| 3 | LJSpeech + M-AILABS (1 langue) | ~50 GB |
| 5 | MuST-C + CVSS (3 paires de langues) | ~500 GB |
| 6 | + FLEURS pour évaluation | +50 GB |

**Total recommandé : 1 TB de stockage SSD pour confort de travail.**
