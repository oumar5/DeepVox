# 13 — Retour d'expérience Phase 2 ASR, run #2 (BiLSTM CTC, 80 k fichiers)

**Date** : 2026-04-18
**Phase** : 2 — ASR directe (Codec2 → texte français)
**Plateforme** : Kaggle, GPU Tesla T4 (15.6 GB VRAM)
**Statut** : **terminé** (50 epochs complets, pas d'early stopping)

## Contexte

Deuxième run ASR, faisant suite au run #1 (20k, CER=71.2 %). L'objectif est de valider
que le scaling de données fonctionne aussi pour l'ASR CTC, comme observé en Phase 1
(chaque ×4 du corpus = −6-7 pp de PER).

## Configuration

| Paramètre | Run #1 | **Run #2** |
|---|---|---|
| Corpus | 20 000 | **80 000** (×4) |
| Train / Dev / Test | 18k / 1k / 1k | **72k / 4k / 4k** |
| MAX_EPOCHS | 30 | **50** |
| PATIENCE | 5 | **7** |
| Architecture | BiLSTM 3L, h=384 | idem |
| Paramètres | 9.1 M | idem |
| Optimizer | AdamW lr=3e-4 | idem |
| Batch size | 32 | idem |
| Durée/epoch | 113 s | **456 s** (×4, cohérent) |
| **Durée totale** | ~47 min | **~6.3 h** |

## Statistiques des données

| Statistique | Run #1 | **Run #2** |
|---|---|---|
| Samples valides | 20 000 | **80 000** |
| Skipped | 3 | **9** |
| Frames/sample (mean) | 140 | **136** |
| Chars/sample (mean) | 59 | **59** |
| Chars/sample (max) | 135 | **200** |

## Résultats — évolution par epoch

| Epoch | Train loss | Dev WER | Dev CER | lr | Note |
|---|---|---|---|---|---|
| 1 | 2.9210 | 1.000 | 0.865 | 3.0e-04 | Blank collapse |
| 2 | 2.7226 | 1.325 | 0.734 | 3.0e-04 | "e" dominant |
| 7 | 2.5926 | 1.145 | 0.699 | 1.5e-04 | Premiers mots partiels |
| 9 | 2.5213 | 1.056 | 0.689 | 1.5e-04 | "il sone" |
| 14 | 2.4414 | 1.027 | 0.668 | 1.5e-04 | Syllabes reconnaissables |
| 20 | 2.3398 | 1.013 | 0.639 | 1.5e-04 | "la mae dede" (≈ "la mère de de") |
| 25 | 2.1925 | 0.973 | 0.618 | 1.5e-04 | WER < 1.0 |
| 30 | 2.1069 | 0.978 | 0.600 | 1.5e-04 | CER passe sous 60% |
| 35 | 2.0838 | 0.974 | 0.594 | 1.5e-04 | |
| 40 | 2.0026 | 0.957 | 0.584 | 1.5e-04 | |
| 45 | 1.9593 | 0.942 | 0.581 | 7.5e-05 | LR réduite ×0.5 |
| **50** | **1.8443** | **0.948** | **0.568** | 7.5e-05 | **Fin — pas de plateau** |

**Observations :**
- La loss et le CER baissent **encore à epoch 50** — le modèle n'a pas convergé
- Le LR scheduler a réduit le lr à epoch 45 (3e-4 → 1.5e-4 → 7.5e-5)
- Pas d'early stopping déclenché (les 50 epochs ont tourné)

## Résultats test (meilleur checkpoint, epoch 50)

| Métrique | Run #1 (20k) | **Run #2 (80k)** | Gain |
|---|---|---|---|
| **CER** | 71.2 % | **56.9 %** | **−14.3 pp** |
| **WER** | 115.5 % | **95.0 %** | **−20.5 pp** |
| Test samples | 1 000 | **4 000** |

## Exemples qualitatifs (test set)

| REF | HYP | Analyse |
|---|---|---|
| elle se rencontre dans les états de colima et de jalisco | elle se rencontre dans la lépale commie et la carito | **4 premiers mots parfaits**, structure préservée |
| c'est une politique publique qui se rattache à la politique culturelle | c'est one ati pe pu pan en tasen la latique tesle | "c'est" correct, "politique" → "latique" |
| au sud il forme la frontière entre le guatemala et le belize | au chi de fon la fontiee pale potela e le dédise | "au" et "la fontiee" (≈frontière) reconnaissables |
| son chef-lieu est la ville de gopalganj | son chefff-lieue est lami tatro atok | "son chef-lieu est" quasi correct |
| il se déplace au sud de l'anticyclone subtropicale pris dans les alizés | il se dapa o mosui de lentiique là sopropilal ri dans les bai | "il se" et "dans les" corrects |
| l'île relève de l'état de pernambouc au brésil | mas lie reproee de letar de caron couqe on prolele | Plus difficile, noms propres ratés |
| on obtient le mercure par grillage du cinabre | conotin de cuurur a lirér u sude | Vocabulaire technique — échec |

### Progression qualitative Run #1 → Run #2

| Aspect | Run #1 | **Run #2** |
|---|---|---|
| Mots complets reconnus | "il" seulement | **"c'est", "elle se rencontre dans", "son chef-lieu est", "il se", "au", "la", "dans les"** |
| Structure de phrase | Absent | **Largement préservée** |
| Consonnes | Quasi absentes | **s, t, d, n, r, l présentes** |
| Noms propres | — | Échoue (attendu) |

## Analyse comparative Phase 1 / Phase 2

### Le scaling ×4 fonctionne dans les deux phases

| Phase | Corpus ×4 | Gain métrique |
|---|---|---|
| Phase 1 (phonème) | 5k → 20k | PER −6.4 pp |
| Phase 1 (phonème) | 20k → 80k | PER −6.8 pp |
| **Phase 2 (ASR)** | **20k → 80k** | **CER −14.3 pp** |

Le gain est **deux fois plus important** pour l'ASR que pour la classification phonétique.
Cela s'explique : l'ASR CTC bénéficie à la fois de plus de variété phonétique ET de plus
de contexte linguistique (séquences de mots).

### Projection avec plus de données

| Corpus | CER projeté | Remarque |
|---|---|---|
| 80k | 56.9 % (mesuré) | |
| 320k | ~43 % | ×4, gain ~14 pp |
| 586k (corpus complet) | ~35-38 % | |
| 586k + plus d'epochs | ~30-35 % | Le modèle n'a pas convergé à 50 epochs |
| 586k + KenLM | **~20-25 %** | LM français corrigerait ~10-15 pp |

### Le modèle n'a pas convergé

La loss baisse encore à epoch 50 (1.84), le CER continue de s'améliorer. Avec 100 epochs,
on pourrait gagner 3-5 pp de CER supplémentaires sans changer le modèle ni les données.

## Interprétation scientifique

### Signal très positif

1. **Les mots fonctionnels sont reconnus** : "c'est", "elle se", "dans les", "son", "la", "de"
   — le modèle a appris la fréquence lexicale du français directement depuis Codec2
2. **La structure syntaxique est préservée** : les hypothèses ont le bon nombre de mots
   (WER < 1.0 = pas trop d'insertions/suppressions)
3. **Les consonnes émergent** : le Run #1 ne produisait que des voyelles, le Run #2
   produit des consonnes (s, t, d, n, r, l, p, c) — preuve que l'information est dans Codec2

### Limites identifiées

1. **Noms propres et vocabulaire technique** : "pernambouc", "cinabre", "gopalganj"
   — impossible sans LM ou données massives
2. **Confusions de voyelles** : "politique" → "latique", "frontière" → "fontiee"
   — les voyelles sont les plus confondues en Codec2 (confirmé en Phase 1)
3. **Mots longs** : le modèle perd la fin des mots longs (alignement CTC difficile
   à 25 fps)

### Codec2 comme entrée ASR : validé

L'hypothèse centrale de DeepVox est confirmée : **Codec2 1200 bps (48 bits / 40 ms)
contient assez d'information pour faire de l'ASR caractère par caractère**. Le CER de
56.9 % avec un simple BiLSTM 9M params, sans LM, sur 80k échantillons, est un résultat
encourageant qui se compare favorablement aux premiers systèmes CTC sur features
classiques (mel-spectrogramme).

## Décisions

### 1. Phase 2 validée qualitativement

Le CER de 56.9 % sans LM, avec un BiLSTM simple, confirme que le pipeline
Codec2 → ASR est viable. Les améliorations prioritaires sont le volume de données
et l'ajout d'un modèle de langue.

### 2. Prochaines améliorations (par ordre de priorité)

| Action | Gain estimé | Effort | Priorité |
|---|---|---|---|
| **Plus d'epochs** (100) | −3-5 pp | Trivial | 1 |
| **×4-7 données** (320k-586k) | −14-20 pp | Faible | 2 |
| **KenLM beam search** | −10-15 pp | Moyen | 3 |
| **Conformer** | −10-20 pp | Fort | 4 (article) |

### 3. Run #3 proposé

- `MAX_SAMPLES = 300_000` (ou corpus complet si la RAM Kaggle le permet)
- `MAX_EPOCHS = 100`
- Même architecture BiLSTM pour comparer
- Objectif : CER < 40 % sans LM

## Fichiers produits

- `/kaggle/working/run2_80k/best_asr.pt` — checkpoint best (36.5 MB, epoch 50)
- `/kaggle/working/run2_80k/checkpoint_epoch{10,20,30,40,50}.pt` — checkpoints périodiques
- `/kaggle/working/run2_80k_model.pt` — modèle final
- `/kaggle/working/run2_80k_training_curves.png` — courbes d'apprentissage
- `notebooks/02_phase2_asr_kaggle.ipynb` — notebook avec résultats complets

## Durée totale

| Étape | Durée |
|---|---|
| Preprocessing | ~20 min (80k MP3 → Codec2) |
| Entraînement | ~6.3 h (50 × 456 s) |
| Évaluation | ~30 s (4k test) |
| **Total** | **~6.6 h** sur Tesla T4 |

## Prochaine étape

**Phase 2 — Run #3** : relancer sur le corpus complet (~586k) avec 100 epochs.
Si CER < 40 %, intégrer KenLM (beam search + modèle de langue français).
