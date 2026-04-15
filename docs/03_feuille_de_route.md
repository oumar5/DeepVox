# 03 — Feuille de route phasée

Six phases progressives, chacune avec un livrable mesurable. Chaque phase doit être validée avant de passer à la suivante : l'échec d'une phase précoce doit faire pivoter ou arrêter la recherche, pas être ignoré.

## Phase 1 — Validation phonétique de Codec2

**Question :** les frames Codec2 préservent-elles assez d'information phonétique pour discriminer les phonèmes d'une langue ?

**Livrable :** classifieur de phonèmes (entrée = séquence de frames Codec2, sortie = phonème ARPABET) avec mesure de précision.

**Critère de succès :** précision phonétique ≥85 % sur TIMIT (corpus de référence pour ce type d'évaluation), à comparer aux baselines spectrogramme (~88 %) et features brutes PCM (~70 %).

**Si échec :** explorer Codec2 700C (encore plus comprimé, contre-intuitif mais à tester), ou ajouter un pré-traitement (delta features sur LSP). Si toujours échec : la piste est invalidée.

**Détail :** voir [04_protocole_experimental_phase1.md](04_protocole_experimental_phase1.md).

## Phase 2 — ASR monolingue depuis frames Codec2

**Question :** peut-on faire de la reconnaissance vocale complète (frames → texte) avec qualité acceptable ?

**Livrable :** modèle ASR seq2seq (encodeur transformer + decoder CTC ou attention) sur LibriSpeech ou Common Voice français.

**Critère de succès :** WER (Word Error Rate) ≤25 % sur LibriSpeech test-clean, à comparer à Whisper-tiny (~12 % WER mais 39 M paramètres). Cible de taille : <30 M paramètres.

## Phase 3 — TTS texte → frames Codec2

**Question :** peut-on générer des frames Codec2 cohérentes à partir du texte, qui après reconstruction Codec2 produisent une voix intelligible ?

**Livrable :** modèle TTS de type FastSpeech-like, sortie = séquence de frames Codec2 quantifiées.

**Critère de succès :** MOS (Mean Opinion Score) ≥3.0/5 en évaluation humaine sur 50 phrases. Intelligibilité ≥90 % (mots correctement reconnus à l'écoute).

## Phase 4 — Cascade Codec2 (baseline composite)

**Question :** la composition naïve ASR + MT + TTS donne-t-elle une baseline S2ST fonctionnelle ?

**Livrable :** pipeline cascade : Codec2 FR → [Phase 2 ASR] → texte FR → [NLLB-200 distilled] → texte EN → [Phase 3 TTS] → Codec2 EN. Aucun nouveau modèle à entraîner ici, juste assemblage et mesure.

**Critère de succès :** BLEU ≥15 sur la sortie texte intermédiaire (vs. référence MuST-C). Cette phase fournit la baseline à battre par la Phase 5.

## Phase 5 — Modèles end-to-end S2TT et S2ST

**Question :** un modèle direct fait-il mieux que le cascade, à taille équivalente ?

**Livrable :** deux modèles end-to-end :
- S2TT : Codec2 FR → texte EN
- S2ST : Codec2 FR → Codec2 EN

**Critère de succès :** S2TT ≥18 BLEU (vs. baseline Phase 4 à ~15). S2ST mesuré via ASR-BLEU (transcription automatique de la sortie audio puis BLEU).

## Phase 6 — Unification dans backbone partagé

**Question :** peut-on factoriser les modèles des Phases 2, 3, 5 dans une architecture hybride (C) qui réduit la taille totale tout en préservant les performances ?

**Livrable :** backbone partagé + heads spécialisées, pesant au total <200 MB pour 3 paires de langues (FR↔EN, FR↔ES, FR↔AR par exemple).

**Critère de succès :** dégradation <10 % sur chaque tâche par rapport aux modèles indépendants des phases précédentes.

## Tableau récapitulatif

| Phase | Durée estimée | Compute requis | Livrable | Risque d'échec |
|---|---|---|---|---|
| 1 — Validation phonétique | 2-4 semaines | 1 GPU consumer | Classifieur phonèmes | **Bloquant si échec** |
| 2 — ASR Codec2 | 4-8 semaines | 1-2 GPU consumer | Modèle ASR <30 M | Modéré |
| 3 — TTS Codec2 | 4-8 semaines | 1-2 GPU consumer | Modèle TTS <40 M | Modéré |
| 4 — Cascade baseline | 1-2 semaines | CPU possible | Pipeline + métriques | Faible (assemblage) |
| 5 — End-to-end | 8-12 semaines | 2-4 GPU | Modèles S2TT, S2ST | Élevé |
| 6 — Unification | 6-10 semaines | 2-4 GPU | Architecture hybride finale | Modéré |

**Total estimé :** 25-44 semaines (6-11 mois) en travail séquentiel par une personne. Phases 2 et 3 peuvent être parallélisées si ressources disponibles.

## Principe de validation continue

Après chaque phase, se poser les questions :
1. Le critère de succès est-il atteint ?
2. Si non, est-ce un problème de méthode (ajustable) ou de fondation (la piste elle-même est en cause) ?
3. Le résultat publiable seul justifie-t-il un papier intermédiaire ?

Les Phases 1, 2 et 5 peuvent chacune donner lieu à une publication indépendante.
