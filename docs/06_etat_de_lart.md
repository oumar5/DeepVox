# 06 — État de l'art

Vue structurée des travaux pertinents, organisée par axe. L'objectif n'est pas l'exhaustivité mais l'identification des points d'ancrage et des écarts qui justifient la nouveauté de notre piste.

## Axe 1 — Traduction parole-à-texte end-to-end (S2TT)

| Année | Référence | Apport | Position vs. notre travail |
|---|---|---|---|
| 2016 | Bérard et al., *Listen and Translate* (NIPS WS) | Premier modèle E2E speech translation, FR→EN sur audio synthétique | Fondateur conceptuel |
| 2017 | Weiss et al. (Google), *Sequence-to-sequence models can directly translate foreign speech* (Interspeech) | Premier résultat compétitif sur vraie parole, ES→EN | Référence baseline historique |
| 2018 | Bérard et al., *End-to-end automatic speech translation of audiobooks* (ICASSP) | Premier corpus dédié (Augmented LibriSpeech), résultats reproductibles | — |
| 2020 | Wang et al., *fairseq S2T* (ACL) | Framework open-source, devenu standard | Outil potentiel |
| 2022 | Radford et al. (OpenAI), *Whisper* | 680 000 h supervisées, 100 langues, S2TT vers anglais | **Concurrent direct sur la qualité** |
| 2023 | Communication entre Meta, *SeamlessM4T* (arXiv 2308.11596) | Multimodal S2TT + S2ST, ~100 langues, modèle de 2.3 GB | **Concurrent direct sur la couverture** |

## Axe 2 — Traduction parole-à-parole (S2ST)

| Année | Référence | Apport | Position vs. notre travail |
|---|---|---|---|
| 2019 | Jia et al. (Google), *Translatotron* | Premier S2ST end-to-end (sans texte intermédiaire) | Fondateur |
| 2021 | Jia et al., *Translatotron 2* | Amélioration qualité, voix préservée | — |
| 2022 | Lee et al. (Meta), *Direct speech-to-speech translation with discrete units* | Utilisation d'unités discrètes audio (HuBERT) — **architecture la plus proche de la nôtre conceptuellement** | **Référence clé : compare leurs unités HuBERT à nos frames Codec2** |
| 2023 | Rubenstein et al. (Google), *AudioPaLM* | Fusion LLM + tokens audio (SoundStream) | Référence pour l'architecture A unifiée |
| 2023 | Communication entre Meta, *SeamlessExpressive* | Préservation prosodique, expressivité | À comparer pour Phase 6 (voice cloning) |

## Axe 3 — Représentations discrètes de la parole

| Année | Référence | Représentation | Bitrate équivalent |
|---|---|---|---|
| 2020 | Baevski et al. (Meta), *wav2vec 2.0* | Tokens issus de quantization vectorielle apprise | ~50 tokens/sec |
| 2021 | Hsu et al. (Meta), *HuBERT* | Tokens issus de clustering itératif d'embeddings | ~50 tokens/sec |
| 2022 | Défossez et al. (Meta), *EnCodec* | Codec neuronal multi-codebook | 1.5-24 kbps |
| 2021 | Zeghidour et al. (Google), *SoundStream* | Codec neuronal end-to-end | 3-18 kbps |
| 2010 | Rowe, *Codec2* | Codec déterministe par analyse-synthèse LPC | **0.7-3.2 kbps** |

**Observation centrale :** Codec2 est environ 3-10× plus compact que les codecs neuronaux modernes, tout en restant intelligible. Aucun travail publié n'a utilisé Codec2 comme représentation d'entrée pour un modèle de speech translation. C'est précisément l'écart que notre recherche vise à explorer.

## Axe 4 — Modèles vocaux compacts pour mobile

| Année | Référence | Apport | Taille |
|---|---|---|---|
| 2022 | Whisper-tiny (OpenAI) | ASR multilingue | 39 M paramètres / ~150 MB |
| 2023 | DistilWhisper (HuggingFace) | Distillation de Whisper | 166 M / ~600 MB |
| 2023 | MMS (Meta), *Massively Multilingual Speech* | ASR + TTS pour 1000+ langues | Variable, modèles >300 MB |
| 2024 | Moonshine (Useful Sensors) | ASR ultra-rapide pour edge | 27-61 M paramètres |

**Aucun de ces modèles ne descend sous ~150 MB pour une couverture multilingue utile.** L'angle "écosystème complet <200 MB" est ouvert.

## Axe 5 — Codec2 dans la littérature scientifique

Codec2 est essentiellement étudié dans le contexte des télécommunications HF / radioamateur, pas du machine learning :

- Rowe, *Codec2 — A Low Bit Rate Speech Codec for Amateur Radio* (DEFCON 2014, communications HF) — paper original
- Quelques travaux comparant Codec2 à AMR/G.729 en téléphonie satellite
- **Zéro publication en speech recognition / translation utilisant Codec2 comme représentation d'entrée**

C'est à la fois une opportunité (terrain inexploré) et un risque (peut-être que d'autres ont essayé en interne et n'ont rien publié faute de résultats).

## Synthèse — Positionnement

```
                    Couverture multilingue
                            ↑
                            │
              SeamlessM4T   │
                  ●         │
                            │
              Whisper-large │
                  ●         │ AudioPaLM
                            │      ●
              Whisper-tiny  │
                  ●         │
   ─────────────────────────┼─────────────────────→
                            │           Compacité (inverse de la taille)
                            │
                            │
                            │
                            │      ┌─────────────┐
                            │      │ NOTRE PISTE │
                            │      │ (Codec2)    │
                            │      └─────────────┘
                            ↓
```

L'espace en bas à droite — **haute compacité × bonne couverture multilingue** — est essentiellement vide. C'est la niche que notre recherche vise.

## Travaux à lire en priorité

Pour quiconque démarre la Phase 1, lecture obligatoire dans cet ordre :

1. **Lee et al. 2022** (Meta, *Direct speech-to-speech translation with discrete units*) — l'architecture de référence à adapter
2. **Hsu et al. 2021** (HuBERT) — pour comprendre la philosophie des tokens audio discrets et comment ils sont entraînés
3. **Radford et al. 2022** (Whisper) — pour les techniques d'entraînement multilingue à grande échelle
4. **Bérard et al. 2018** — pour le protocole d'évaluation propre des systèmes E2E ST
5. **Rowe 2014** (Codec2) — pour comprendre intimement la représentation utilisée en entrée

## Conférences et journaux cibles

| Venue | Type | Cycle de soumission |
|---|---|---|
| **Interspeech** | Conf. parole générale | Mars/septembre annuel |
| **ICASSP** | Conf. IEEE signal processing | Octobre annuel |
| **ACL / EMNLP / NAACL** | Conf. NLP, sections speech | Variables |
| **NeurIPS / ICML** | Conf. ML générale | Mai / janvier |
| **IEEE/ACM TASLP** | Journal référence audio/speech | Continu |

Pour SmsVox, **Interspeech** est probablement la cible la plus pertinente (forte tradition en codecs et en speech translation, ouverture aux travaux orientés systèmes contraints).
