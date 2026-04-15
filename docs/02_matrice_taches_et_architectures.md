# 02 — Matrice des tâches et architectures candidates

## Les sept tâches cibles

| # | Acronyme | Tâche | Entrée | Sortie | Cas d'usage SmsVox principal |
|---|---|---|---|---|---|
| 1 | **ASR** | Automatic Speech Recognition | Frames Codec2 (langue X) | Texte (langue X) | Recherche, sous-titres, accessibilité |
| 2 | **TTS** | Text-to-Speech | Texte (langue X) | Frames Codec2 (langue X) | Envoyer un vocal sans parler |
| 3 | **MT** | Machine Translation (texte) | Texte (langue X) | Texte (langue Y) | Traduction écrite (peut réutiliser modèles existants) |
| 4 | **S2TT** | Speech-to-Text Translation | Frames Codec2 (langue X) | Texte (langue Y) | Comprendre un vocal étranger |
| 5 | **TTST** | Text-to-Speech Translation | Texte (langue X) | Frames Codec2 (langue Y) | Envoyer un vocal dans une langue qu'on ne parle pas |
| 6 | **S2ST** | Speech-to-Speech Translation | Frames Codec2 (langue X) | Frames Codec2 (langue Y) | Conversation multilingue complète |
| 7 | **VC** | Voice Cloning translaté | Frames Codec2 (X) + texte ou frames (Y) | Frames Codec2 (Y) avec timbre source | Préserver l'identité vocale après traduction |

## Trois architectures candidates

### Architecture A — Modèle unifié multimodal

Inspiré d'AudioPaLM et SeamlessM4T. Un seul transformer avec un vocabulaire fusionné texte + tokens Codec2, conditionné par tokens spéciaux de tâche.

```
Vocabulaire = {tokens_BPE_texte} ∪ {tokens_codec2}

Conditionnement :
  [ASR_FR]  voix → texte_fr
  [S2TT_FR_EN] voix_fr → texte_en
  [S2ST_FR_EN] voix_fr → voix_en
  [TTS_FR]  texte_fr → voix
  ...
```

**Avantages :** transfert positif entre tâches, une seule mémoire chargée, capacités émergentes possibles, maintenance simplifiée.

**Inconvénients :** modèle plus gros, entraînement plus coûteux, difficile à diagnostiquer, peu modulaire.

### Architecture B — Modèles spécialisés indépendants

Plusieurs modèles compacts, chacun ~10-50 M paramètres, distillés et quantifiés indépendamment.

```
codec2_asr_fr.tflite        ~15 MB
codec2_tts_fr.tflite        ~20 MB
codec2_s2tt_fr_en.tflite    ~30 MB
codec2_s2st_fr_en.tflite    ~40 MB
...
```

**Avantages :** très petits modèles individuels, téléchargement à la carte par paire de langues, optimisation fine par tâche, itération rapide.

**Inconvénients :** pas de transfert entre tâches, surcoût mémoire si plusieurs modèles actifs, surface de maintenance large.

### Architecture C — Hybride (recommandée)

Backbone partagé qui produit des embeddings sémantiques à partir des frames Codec2, plus des têtes spécialisées légères (style adaptateurs LoRA + decoders dédiés).

```
┌─────────────────────────────────────────────┐
│  ENCODEUR PARTAGÉ                           │
│  Codec2 frames → embeddings sémantiques     │
│  ~20 MB, entraîné une fois sur multi-tâches │
└──────────────┬──────────────────────────────┘
               │
       ┌───────┼────────┬──────────┬──────────┐
       ↓       ↓        ↓          ↓          ↓
   [ASR head][TTS head][S2TT head][S2ST head][VC head]
    ~5 MB    ~8 MB     ~10 MB     ~15 MB     ~10 MB
```

C'est le pattern utilisé par Whisper (encodeur partagé, decoder conditionné par token de tâche) et par les modèles multi-tâches NLP modernes (T5, mT5, ByT5).

**Avantages :** combine les forces de A et B, ajout d'une nouvelle paire de langues = nouvelle head sans réentraînement complet du backbone, économie de RAM (un seul encodeur en mémoire).

**Inconvénients :** complexité d'orchestration accrue, choix non trivial du dimensionnement entre backbone et heads.

## Recommandation

**Architecture C — Hybride.** Elle permet de :
- Démarrer petit (juste backbone + ASR head pour valider la Phase 1)
- Ajouter des capacités progressivement sans tout réentraîner
- Respecter la cible <200 MB pour un écosystème complet
- Bénéficier du transfert entre tâches là où c'est utile, sans le payer là où ça ne l'est pas

## Comparaison synthétique

| Critère | A (unifié) | B (spécialisés) | C (hybride) |
|---|---|---|---|
| Taille totale | Moyenne (1 gros) | Variable (somme) | Petite (backbone + heads) |
| Transfert entre tâches | Excellent | Nul | Bon (via backbone) |
| Modularité | Faible | Excellente | Excellente |
| Coût d'entraînement initial | Élevé | Distribué | Moyen |
| Ajout d'une langue | Réentraînement | Nouveau modèle | Nouvelle head |
| Adapté à mobile bas de gamme | Limite | Oui | Oui |
| Risque diagnostic | Élevé | Faible | Faible |
