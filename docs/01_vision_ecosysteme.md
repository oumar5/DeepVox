# 01 — Vision écosystème

## Du codec à la représentation pivot

SmsVox a démontré qu'une voix humaine de 5 secondes peut être encodée en ~750 bits utiles via Codec2 (mode 1200 bps), tout en restant intelligible après reconstruction. Ce résultat est la pierre angulaire d'une seconde idée : **si Codec2 préserve assez d'information pour qu'un humain comprenne la parole reconstruite, alors un modèle de deep learning peut probablement opérer directement sur cette représentation binaire compacte**.

L'analogie est directe avec le passage du texte brut aux tokens BPE/SentencePiece : on remplace une représentation continue (formes d'onde, ou caractères Unicode) par une séquence discrète compacte, sur laquelle les transformers excellent.

## Le rôle de Codec2

Codec2 est un codec déterministe développé par David Rowe pour la radioamateur HF. Une frame Codec2 1200 bps contient :

| Champ | Bits | Sens phonétique |
|---|---|---|
| LSP (Line Spectral Pairs) | 36 | Timbre, formants → distinction voyelles/consonnes |
| Pitch (F0) | 7 | Hauteur tonale → prosodie, accent, langue tonale |
| Énergie | 5 | Volume, accent d'intensité |
| Voicing | 1 (×4 sous-trames) | Voisé / non-voisé → distinction sourde/sonore |

Ces champs ne sont pas opaques : ils ont une interprétation phonétique directe. C'est ce qui distingue Codec2 des codecs neuronaux (EnCodec, SoundStream), dont les tokens sont appris et opaques.

## Pourquoi un écosystème, pas un seul modèle

Une fois qu'on dispose d'une représentation pivot stable, on peut construire un graphe de capacités où **toutes les arêtes partagent le même alphabet de tokens audio**. Cela autorise du transfert d'apprentissage entre tâches et un déploiement modulaire.

```
                          ┌──────────────┐
                          │   TEXTE FR   │
                          └──────┬───────┘
                          ASR ↑     ↓ TTS
       ┌──────────────┐         │              ┌──────────────┐
       │  TEXTE EN    │←── MT ──┤              │  TEXTE EN    │
       └──────┬───────┘         │              └──────┬───────┘
              │ TTS_EN          ▼ S2ST                │ TTS_EN
              ▼          ┌──────────────┐             ▼
       ┌──────────────┐  │ FRAMES       │      ┌──────────────┐
       │  CODEC2 EN   │←─│ CODEC2 FR    │─────→│  CODEC2 EN   │
       └──────┬───────┘  └──────┬───────┘      └──────┬───────┘
              ▼                 ▲                     ▼
          [VOIX EN]         [VOIX FR]             [VOIX EN]
```

Le nœud central — les frames Codec2 — est partagé par toutes les tâches. C'est ce qui permet d'envisager un **backbone neuronal partagé** avec des têtes spécialisées légères (architecture hybride détaillée dans le document 02).

## Position par rapport au cœur de SmsVox

Cette recherche est **complémentaire**, pas concurrente, à la mission première de SmsVox (messagerie vocale chiffrée par SMS) :

| Capacité | Bénéfice pour SmsVox |
|---|---|
| **ASR Codec2 → texte** | Recherche dans l'historique vocal, sous-titrage automatique, accessibilité (sourds) |
| **TTS texte → Codec2** | Envoyer un message vocal à partir d'une saisie clavier (utilisateur muet, environnement bruyant) |
| **S2TT (parole → texte étranger)** | Comprendre un correspondant qui parle une autre langue |
| **S2ST (parole → parole étrangère)** | Conversation vocale multilingue asynchrone par SMS |
| **Voice cloning préservant le timbre** | Le destinataire entend la voix de l'expéditeur, même après traduction |

Toutes ces capacités sont impossibles aujourd'hui dans des régions à faible connectivité, faute de modèles assez compacts. C'est l'opportunité de recherche.

## Critère de succès global

Un écosystème complet (ASR + TTS + S2TT + S2ST pour au moins 3 paires de langues) tenant en **moins de 200 MB total**, exécutable hors-ligne sur un smartphone Android d'entrée de gamme (2 GB RAM, CPU ARM Cortex-A53), avec une qualité de traduction comparable aux baselines cascade utilisant Whisper-small + NLLB-200-distilled-600M.
