# DeepVox

> Ultra-light multilingual speech translation built on Codec2 discrete representations.

**Auteur :** Oumar Ben Lol — Ingénieur en Intelligence Artificielle & Expert en Transformation Digitale

**Statut :** Recherche en cours — Phase 1 (validation phonétique) à démarrer.

**Licence :** Apache 2.0

---

## Vision

DeepVox explore l'utilisation des **frames du codec déterministe Codec2** (700-1200 bps) comme représentation pivot discrète pour entraîner un écosystème complet de modèles vocaux multilingues (ASR, TTS, traduction parole-texte, traduction parole-parole), avec une cible de **moins de 200 Mo total** pour 3 paires de langues complètes — exécutable hors-ligne sur smartphones d'entrée de gamme.

L'hypothèse centrale : *si Codec2 préserve assez d'information pour qu'un humain comprenne la parole reconstruite, un transformer peut probablement opérer directement sur cette représentation binaire compacte.*

## Position vs. l'état de l'art

| Approche existante | Notre approche |
|---|---|
| Spectrogrammes mel (Whisper, SeamlessM4T) | Frames Codec2 déterministes |
| Codecs neuronaux appris (EnCodec, SoundStream) | Codec déterministe figé |
| Modèles >2 Go | Cible : écosystème complet <200 Mo |
| GPU/cloud requis | Inférence mobile bas de gamme |

## Relation avec SmsVox

DeepVox est issu du projet [SmsVox](https://github.com/oumar5/SmsVox) (messagerie vocale chiffrée par SMS) qui a démontré la faisabilité de Codec2 pour la transmission vocale ultra-bas-débit. DeepVox produit des modèles `.tflite` que SmsVox embarque comme artefacts binaires : **aucun couplage code-source**, juste des modèles téléchargeables.

```
[DeepVox] ──produit──→ modèles .tflite + papers
                              ↓
                  téléchargés/embarqués par
                              ↓
                          [SmsVox app]
```

## Documentation

| Document | Sujet |
|---|---|
| [docs/01_vision_ecosysteme.md](docs/01_vision_ecosysteme.md) | Vision globale, schéma écosystème |
| [docs/02_matrice_taches_et_architectures.md](docs/02_matrice_taches_et_architectures.md) | 7 tâches cibles, 3 architectures candidates |
| [docs/03_feuille_de_route.md](docs/03_feuille_de_route.md) | 6 phases progressives |
| [docs/04_protocole_experimental_phase1.md](docs/04_protocole_experimental_phase1.md) | Protocole de validation phonétique |
| [docs/05_sources_donnees.md](docs/05_sources_donnees.md) | Corpus francophones et parallèles |
| [docs/06_etat_de_lart.md](docs/06_etat_de_lart.md) | Travaux liés (Whisper, SeamlessM4T, etc.) |
| [docs/07_risques_et_questions_ouvertes.md](docs/07_risques_et_questions_ouvertes.md) | Limites identifiées |

## Article de recherche

[paper/article_fr.md](paper/article_fr.md) — Brouillon de soumission Interspeech / ICASSP.

## Structure du dépôt

```
DeepVox/
├── README.md                 ← ce fichier
├── LICENSE                   ← Apache 2.0
├── pyproject.toml            ← packaging Python
├── docs/                     ← documentation de recherche
├── paper/                    ← article scientifique
├── src/deepvox/              ← code source
│   ├── codec2/               ← bindings Codec2
│   ├── data/                 ← chargeurs de datasets
│   ├── models/               ← architectures (encodeur, têtes)
│   ├── training/             ← scripts d'entraînement par phase
│   └── eval/                 ← métriques (PER, WER, BLEU, MOS)
├── scripts/                  ← scripts utilitaires (préprocessing, exports)
├── tests/                    ← tests unitaires
└── notebooks/                ← exploration interactive
```

## Installation (à venir)

```bash
git clone https://github.com/oumar5/DeepVox.git
cd DeepVox
pip install -e ".[dev]"
```

## Roadmap

- [ ] Phase 1 — Validation phonétique de Codec2 sur TIMIT
- [ ] Phase 2 — ASR monolingue depuis frames Codec2
- [ ] Phase 3 — TTS texte → frames Codec2
- [ ] Phase 4 — Cascade baseline (ASR + MT + TTS)
- [ ] Phase 5 — Modèles end-to-end S2TT et S2ST
- [ ] Phase 6 — Unification dans backbone partagé

Voir [docs/03_feuille_de_route.md](docs/03_feuille_de_route.md) pour les critères de validation détaillés.

## Considérations éthiques

L'écosystème inclut potentiellement du voice cloning (Phase 6). Tout déploiement intégrera un watermarking audio (cf. AudioSeal) et limitera le clonage inter-locuteurs sans consentement. Évaluation par langue obligatoire pour éviter les biais.

## Contribution

Projet en phase exploratoire. Les contributions seront ouvertes après validation de la Phase 1.
