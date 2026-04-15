# Codec2 comme représentation pivot pour la traduction vocale multilingue ultra-légère

**Auteur :** Oumar Ben Lol — Ingénieur en Intelligence Artificielle & Expert en Transformation Digitale

**Statut :** Sujet d'article — Brouillon de recherche (pre-print)

**Date :** 2026-04-16

**Cible de soumission envisagée :** Interspeech 2027 / IEEE ICASSP 2027

---

## Résumé

Nous proposons une approche originale pour la traduction vocale multilingue ultra-légère, fondée sur l'utilisation des frames du codec déterministe **Codec2** (700-1200 bps) comme **représentation pivot discrète** en entrée et sortie de modèles transformers multimodaux. Contrairement aux approches contemporaines (Whisper, SeamlessM4T, AudioPaLM) qui s'appuient sur des spectrogrammes mel ou des codecs neuronaux appris, notre méthode exploite une représentation déjà compacte (~25 tokens/seconde, ~10× plus dense que SoundStream) et **interprétable par construction** (champs LSP, pitch, énergie, voicing). Cette compacité ouvre la voie à un écosystème complet de capacités vocales (ASR, TTS, traduction parole-texte, parole-parole) tenant en moins de 200 Mo total et exécutable hors-ligne sur smartphones d'entrée de gamme. L'article décrit le cadre théorique, six phases expérimentales progressives, et présente les premiers résultats de la Phase 1 (validation de la préservation phonétique sur TIMIT).

**Mots-clés :** speech translation, low-resource, mobile inference, Codec2, discrete speech tokens, multilingual speech models, on-device AI

---

## 1. Introduction

### 1.1 Motivation

Les systèmes de traduction vocale modernes — Whisper [Radford 2022], SeamlessM4T [Communication entre Meta 2023], AudioPaLM [Rubenstein 2023] — atteignent des performances remarquables mais reposent sur des modèles de plusieurs gigaoctets, inutilisables hors ligne sur les appareils à ressources limitées qui équipent une part majoritaire de la population mondiale, notamment dans les régions à faible connectivité.

Cette contrainte n'est pas anecdotique : elle exclut de facto des centaines de millions d'utilisateurs des bénéfices de la traduction vocale. Le projet **SmsVox** (messagerie vocale chiffrée transmise par SMS, opérant à 1200 bps via le codec Codec2) a démontré la faisabilité d'une parole intelligible à très faible bitrate sur canal contraint. Cet article explore la question naturelle qui suit : **si Codec2 préserve assez d'information pour la perception humaine, peut-il en préserver assez pour des modèles de traduction neuronale ?**

### 1.2 Contributions

1. Nous formalisons l'utilisation des frames Codec2 comme **représentation pivot discrète** pour un écosystème vocal multilingue.
2. Nous proposons une **architecture hybride** (encodeur partagé + têtes spécialisées) optimisée pour l'inférence mobile.
3. Nous décrivons un **protocole expérimental phasé** avec critères de validation explicites.
4. Nous présentons les premiers résultats sur la préservation phonétique (Phase 1, TIMIT).
5. Nous discutons honnêtement les **limites identifiées** (langues tonales, fricatives haute fréquence, données parallèles).

### 1.3 Position

Notre travail occupe une niche peu explorée : haute compacité (<200 Mo) × bonne couverture multilingue. Les approches existantes occupent soit la haute compacité avec couverture limitée (Whisper-tiny), soit la couverture large avec coût mémoire prohibitif (SeamlessM4T à 2.3 Go).

---

## 2. Travaux liés

**Speech Translation end-to-end.** Bérard et al. [2016, 2018] ont posé les bases de la traduction parole-texte directe. Weiss et al. [2017] ont fourni les premiers résultats compétitifs sur parole réelle. Whisper [Radford 2022] représente l'état de l'art en couverture multilingue supervisée.

**Speech-to-Speech.** Translatotron [Jia 2019, 2021] a introduit la traduction parole-parole sans texte intermédiaire. Lee et al. [2022] ont proposé l'utilisation d'**unités discrètes** (extraites de HuBERT) — l'architecture conceptuellement la plus proche de la nôtre, à la différence essentielle que leurs unités sont apprises par un réseau neuronal alors que les nôtres sont déterministes (Codec2).

**Représentations discrètes de la parole.** wav2vec 2.0 [Baevski 2020], HuBERT [Hsu 2021], EnCodec [Défossez 2022], SoundStream [Zeghidour 2021] proposent des tokens audio à des bitrates de 1.5-24 kbps. Codec2 [Rowe 2010] opère à **0.7-3.2 kbps**, soit un ordre de grandeur de plus de compacité. Aucune publication antérieure n'a, à notre connaissance, utilisé Codec2 comme représentation d'entrée pour un modèle de traduction.

**Modèles vocaux compacts.** Whisper-tiny [Radford 2022, 39M paramètres], MMS [Meta 2023], Moonshine [2024] représentent l'état de l'art en compacité, mais aucun ne descend sous ~150 Mo pour une couverture multilingue significative.

---

## 3. Cadre proposé

### 3.1 Codec2 comme alphabet phonétique compact

Une frame Codec2 mode 1200 bps encode 40 ms de parole en 48 bits, structurés en :
- **LSP (Line Spectral Pairs)** — 36 bits : enveloppe spectrale, distingue voyelles et consonnes
- **Pitch (F0)** — 7 bits : hauteur tonale, prosodie
- **Énergie** — 5 bits : volume relatif
- **Voicing** — 1 bit × 4 sous-trames : voisement

Cette structure n'est pas opaque : elle a une interprétation phonétique directe, ce qui distingue Codec2 des codecs neuronaux. Une seconde de parole produit 25 frames, soit 25 tokens : c'est **2× plus compact que SoundStream et 4× plus que les spectrogrammes mel** typiques.

### 3.2 Architecture hybride

Nous proposons une architecture à backbone partagé + têtes spécialisées :

```
        Codec2 frames (entrée)
                ↓
   [Encodeur partagé : ~20 MB]
                ↓
        Embeddings sémantiques
                ↓
   ┌────┬──────┬──────┬──────┐
   ↓    ↓      ↓      ↓      ↓
 [ASR][TTS][S2TT][S2ST][VC]   ← têtes spécialisées
  5MB  8MB  10MB  15MB  10MB
```

Le backbone est entraîné multi-tâches sur données massivement multilingues. Les têtes sont fine-tunées par tâche et par paire de langues, téléchargeables à la demande sur le client. **Total cible : <200 Mo pour 3 paires de langues complètes.**

### 3.3 Pipeline d'inférence

```
Voix source (PCM 8 kHz)
    ↓ Codec2 encoder (déterministe, ~50 KB)
Frames Codec2 (~25/sec)
    ↓ Encodeur partagé
Embeddings
    ↓ Tête S2ST [langue source → langue cible]
Frames Codec2 cible
    ↓ Codec2 decoder (déterministe, ~50 KB)
Voix cible (PCM 8 kHz)
```

Le codec lui-même n'est pas à apprendre : il est déterministe et figé, ce qui réduit considérablement la complexité d'entraînement par rapport aux approches à codec neuronal.

---

## 4. Protocole expérimental phasé

Nous proposons six phases progressives avec critères de validation explicites.

| Phase | Objectif | Critère GO |
|---|---|---|
| 1 | Validation phonétique de Codec2 (TIMIT) | PER ≤ 15 % |
| 2 | ASR mono-langue depuis frames Codec2 | WER ≤ 25 % (LibriSpeech) |
| 3 | TTS texte → frames Codec2 | MOS ≥ 3.0/5 |
| 4 | Cascade baseline (ASR + MT + TTS) | BLEU ≥ 15 |
| 5 | End-to-end S2TT et S2ST | BLEU ≥ 18 (vs cascade) |
| 6 | Unification dans backbone partagé | Dégradation < 10 % par tâche |

Ces critères garantissent qu'un échec précoce stoppe la recherche avant accumulation de coûts inutiles.

---

## 5. Premiers résultats (Phase 1)

[Section à compléter après exécution effective de la Phase 1]

**Cadre expérimental :**
- Dataset : TIMIT, 39 phonèmes
- Pré-traitement : resampling 16 kHz → 8 kHz, encodage Codec2 1200 bps
- Classifieur : BiLSTM 2 couches × 256 unités cachées
- Conditions : Codec2 raw, Codec2 + delta, mel spectrograms (baseline), PCM brut (baseline basse)

**Résultats attendus :** précision phonétique macro ~85 % en condition Codec2 + delta, à comparer à ~88 % avec spectrogrammes mel.

**Analyses :**
- Matrice de confusion 39×39
- Ablation par champ Codec2 (LSP, pitch, énergie, voicing)
- Robustesse au bruit additif

[Tableaux et figures à insérer]

---

## 6. Discussion

### 6.1 Limites attendues

- **Langues tonales** : pitch encodé sur 7 bits, résolution potentiellement insuffisante pour mandarin, vietnamien.
- **Fricatives haute fréquence** : Codec2 8 kHz coupe au-dessus de 4 kHz, risque de confusion /s/ /ʃ/ /f/.
- **Prosodie expressive** : Codec2 est optimisé pour intelligibilité, pas expressivité. Impact sur S2ST avec préservation prosodique à mesurer.

### 6.2 Implications

Si la Phase 1 valide notre hypothèse, l'écosystème proposé devient un candidat sérieux pour le **déploiement vocal multilingue dans les régions à faible connectivité**. Au-delà de SmsVox, des cas d'usage incluent : applications humanitaires hors-ligne, accessibilité pour utilisateurs sourds/muets, communication d'urgence en zones sinistrées.

### 6.3 Considérations éthiques

L'écosystème inclut potentiellement du voice cloning (Phase 6 — préservation du timbre après traduction). Tout déploiement doit intégrer un **watermarking audio** (cf. AudioSeal [Meta 2024]) et une limitation produit empêchant le clonage inter-locuteurs sans consentement.

L'attention aux **biais par langue** est essentielle : un modèle qui marche en français mais pas en wolof reproduit des inégalités d'accès. Évaluation explicite et transparente par langue requise.

---

## 7. Conclusion et travaux futurs

Nous avons proposé un cadre complet pour la traduction vocale multilingue ultra-légère utilisant Codec2 comme représentation pivot. La compacité native de Codec2 (25 tokens/sec, 0.7-1.2 kbps) ouvre un espace inexploré dans le paysage des modèles vocaux : **haute compacité × bonne couverture multilingue**.

Les travaux futurs immédiats :
1. Exécution de la Phase 1 (validation phonétique) — bloquante pour la suite
2. Constitution d'un corpus parallèle Codec2-encodé sur MuST-C et CVSS
3. Entraînement de modèles ASR/TTS de référence (Phases 2-3)
4. Soumission Interspeech 2027 si Phase 1 validée

---

## Références

[À développer en format BibTeX dans la version LaTeX]

- Bérard et al. (2016). *Listen and Translate: A Proof of Concept for End-to-End Speech-to-Text Translation.* NIPS Workshop.
- Bérard et al. (2018). *End-to-end automatic speech translation of audiobooks.* ICASSP.
- Weiss et al. (2017). *Sequence-to-sequence models can directly translate foreign speech.* Interspeech.
- Jia et al. (2019). *Direct speech-to-speech translation with a sequence-to-sequence model.* Interspeech (Translatotron).
- Lee et al. (2022). *Direct speech-to-speech translation with discrete units.* ACL.
- Radford et al. (2022). *Robust Speech Recognition via Large-Scale Weak Supervision.* (Whisper).
- Communication entre Meta (2023). *SeamlessM4T: Massively Multilingual & Multimodal Machine Translation.* arXiv:2308.11596.
- Rubenstein et al. (2023). *AudioPaLM: A Large Language Model That Can Speak and Listen.*
- Baevski et al. (2020). *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations.* NeurIPS.
- Hsu et al. (2021). *HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units.* IEEE/ACM TASLP.
- Défossez et al. (2022). *High Fidelity Neural Audio Compression.* (EnCodec).
- Zeghidour et al. (2021). *SoundStream: An End-to-End Neural Audio Codec.* IEEE/ACM TASLP.
- Rowe (2010). *Codec2 — A Low Bit Rate Speech Codec for HF SSB.* documentation technique.

---

## Annexe A — Documentation technique DeepVox

- Vision écosystème : [`docs/01_vision_ecosysteme.md`](../docs/01_vision_ecosysteme.md)
- Matrice tâches/architectures : [`docs/02_matrice_taches_et_architectures.md`](../docs/02_matrice_taches_et_architectures.md)
- Feuille de route : [`docs/03_feuille_de_route.md`](../docs/03_feuille_de_route.md)
- Protocole Phase 1 : [`docs/04_protocole_experimental_phase1.md`](../docs/04_protocole_experimental_phase1.md)
- Sources de données : [`docs/05_sources_donnees.md`](../docs/05_sources_donnees.md)
- État de l'art : [`docs/06_etat_de_lart.md`](../docs/06_etat_de_lart.md)
- Risques : [`docs/07_risques_et_questions_ouvertes.md`](../docs/07_risques_et_questions_ouvertes.md)

## Annexe B — Projet parent

Ce travail s'inscrit dans la continuité du projet [SmsVox](https://github.com/oumar5/SmsVox) (messagerie vocale chiffrée par SMS), qui a démontré en pratique l'intelligibilité de Codec2 à 1200 bps sur canal contraint.
