# 15 — Retour d'expérience Phase 2 ASR, run #4 (Fine-tune 586 k fichiers)

**Date de début** : 2026-04-21
**Phase** : 2 — ASR directe (Codec2 → texte français)
**Plateforme** : Kaggle, GPU Tesla T4 (15.6 GB VRAM)
**Statut** : **en cours** — session 1 terminée (epochs 1-9)

## Contexte

Quatrième run ASR, fine-tune du meilleur modèle du run #3 (CER=32.3% à epoch 38) sur
le **corpus complet Common Voice FR** (586 k fichiers). Stratégie choisie : fine-tune
plutôt que from-scratch pour économiser du temps GPU et tirer parti du modèle déjà
partiellement entraîné.

**Deux innovations majeures par rapport aux runs précédents** :
1. **Preprocessing en local** (~3h sur i9/64GB) au lieu de Kaggle (~8h) — données
   sauvegardées en pickle et uploadées comme dataset Kaggle. Chargement en ~30 sec
   au lieu de 8h à chaque session.
2. **Fine-tune** avec `PRETRAINED_PATH` — on charge les poids du best run3 au démarrage
   et on recrée un optimizer frais avec LR réduit.

## Configuration

| Paramètre | Run #3 | **Run #4** |
|---|---|---|
| Corpus | 300 000 | **586 000** (corpus complet) |
| Train / Dev / Test | 270k / 15k / 15k | **527k / 29k / 29k** |
| MAX_EPOCHS | 50 | **25** |
| PATIENCE | 10 | **7** |
| LEARNING_RATE | 3e-4 | **1e-4** (×3 plus bas, fine-tune) |
| Architecture | BiLSTM 3L, h=384 | idem |
| Paramètres | 9.1 M | idem |
| Point de départ | From scratch | **Best run3 (CER=32.3%)** |
| Optimizer | AdamW (reset) | AdamW (reset, pas de resume) |
| Scheduler | ReduceLROnPlateau | idem |
| Batch size | 32 | idem |
| Durée/epoch | ~1 538 s | **~3 087 s (~51 min)** (×2, 586k vs 300k) |
| Sessions Kaggle prévues | 3 | **~3** (9h de training/session) |

## Statistiques des données

| Statistique | Run #3 (300k) | **Run #4 (586k)** |
|---|---|---|
| Samples valides | 300 000 | **586 000** |
| Skipped | 37 | **56** (preprocessing local) |
| Frames/sample (mean) | 129 | **129** |
| Chars/sample (mean) | 60 | **61** |
| Chars/sample (max) | 216 | **227** |

## Session 1 — Résultats (epochs 1-9)

### Évolution par epoch

| Epoch | Train loss | Dev WER | Dev CER | lr | Δ CER vs run3 best |
|---|---|---|---|---|---|
| (run3 best) | — | — | 0.323 | — | baseline |
| 1 | 1.1030 | 0.647 | **0.278** | 1.0e-04 | **−4.5 pp en 1 epoch !** |
| 2 | 1.0627 | 0.639 | 0.272 | 1.0e-04 | |
| 3 | 1.0393 | 0.632 | 0.270 | 1.0e-04 | |
| 4 | 1.0215 | 0.628 | 0.266 | 1.0e-04 | |
| 5 | 1.0067 | 0.622 | 0.265 | 1.0e-04 | **Loss < 1.0** |
| 6 | 0.9923 | 0.620 | 0.261 | 1.0e-04 | |
| 7 | 0.9796 | 0.613 | 0.259 | 1.0e-04 | |
| 8 | 0.9684 | 0.611 | 0.257 | 1.0e-04 | |
| **9** | **0.9579** | **0.608** | **0.255** | 1.0e-04 | **−6.8 pp** |

### Observations session 1

- **Chute immédiate du CER à l'epoch 1** : −4.5 pp en une seule epoch, preuve que le
  fine-tune avec LR=1e-4 était le bon choix
- **La loss passe sous 1.0 dès l'epoch 5** (vs 1.84 au début du run #3)
- **WER < 0.65 pour la première fois** (run #3 : 0.71 à epoch 38)
- **Le LR n'a pas encore été réduit** — le modèle apprend encore activement
- Durée session 1 : 9 × 3087s ≈ **7.7 h** sur 12h disponibles

### Exemples qualitatifs (epoch 9)

| REF | HYP |
|---|---|
| ralph depalma s'y est imposé à trois reprises | **race de ballmasie ampona à trois reprise** |
| je m'y oppose régulièrement au sein de la commission des finances et en séance | **jili opose régulièrement au seile de la commission des finales sué con fiance** |

**Mots parfaitement reconnus** :
- "régulièrement" (6 lettres conservées)
- "commission" (intégralement)
- "à trois reprise"
- "des finales" (très proche de "des finances")
- "au sein de la"

**Erreurs typiques** :
- Noms propres : "ralph depalma" → "race de ballmasie" (aucun LM pour aider)
- Consonnes proches : "b" / "p" ("ampona" vs "imposé")
- Diacritiques : "su e" vs "séance"

### Durée et budget Kaggle

| Étape | Durée |
|---|---|
| Preprocessing | **~30 sec** (depuis pickle, vs ~8h sinon) |
| Entraînement (9 epochs × 3087s) | ~7.7 h |
| **Total session 1** | **~7.8 h** |

Le preprocessing local (~3h sur i9/64GB) a complètement éliminé le goulot
d'étranglement Kaggle.

---

## Session 2 — Résultats (epochs 10-??)

**Statut** : en cours

Note : la session 1 a en fait fait 10 epochs (epoch 10 sauvegardée dans le checkpoint
mais output non affiché). Resume `from epoch 11 (best CER=0.2548)` confirme.

| Epoch | Train loss | Dev WER | Dev CER | lr | Note |
|---|---|---|---|---|---|
| 11 | 0.9385 | 0.604 | **0.252** | 1.0e-04 | Resume OK, **CER < 25.5%** |
| 12 | — | — | — | — | (epoch tournée mais output non affiché, best CER=0.2509) |
| 13 | 0.9213 | 0.597 | **0.248** | 1.0e-04 | **CER < 25%** |
| 14 | 0.9143 | 0.593 | **0.247** | 1.0e-04 | "siest imposé à trois reprises" quasi parfait |
| 15 | 0.9055 | 0.592 | **0.246** | 1.0e-04 | "au sein de la commission" parfait |
| 16 | 0.8983 | 0.592 | **0.246** | 1.0e-04 | Plateau apparent, loss baisse encore |
| 17 | — | — | — | — | (output non affiché, best CER=0.2451) |
| 18 | 0.8846 | 0.585 | **0.244** | 1.0e-04 | |
| 19 | 0.8780 | 0.584 | **0.243** | 1.0e-04 | "si est imposé à trois reprises" parfait |
| 20 | 0.8717 | 0.584 | **0.241** | 1.0e-04 | |
| 21 | 0.8660 | 0.579 | **0.241** | 1.0e-04 | |
| 22 | 0.8606 | 0.578 | **0.239** | 1.0e-04 | **CER < 24%** |
| 23 | 0.8544 | 0.573 | **0.238** | 1.0e-04 | |
| 24 | 0.8482 | 0.574 | **0.238** | 1.0e-04 | |
| **25** | **0.8427** | **0.575** | **0.238** | 1.0e-04 | **FIN — Best CER=0.2379** |

### Résultat à epoch 25 (premier checkpoint final)

**CER = 23.79% / WER = 57.5%** sur le dev set (29k samples).

La loss baissait encore (0.843), le LR n'a jamais été réduit. Décision : continuer
l'entraînement avec MAX_EPOCHS=50.

---

## Session 5 — Continuation au-delà de 25 epochs (epochs 26-??)

| Epoch | Train loss | Dev WER | Dev CER | lr | Note |
|---|---|---|---|---|---|
| 26 | 0.8380 | 0.568 | **0.236** | 1.0e-04 | Saved best |
| 27 | 0.8322 | 0.562 | **0.232** | 1.0e-04 | |
| 28 | 0.8277 | 0.567 | 0.233 | 1.0e-04 | |
| 29 | 0.8233 | 0.565 | 0.232 | 1.0e-04 | |
| 30 | 0.8184 | 0.561 | **0.231** | 1.0e-04 | |
| 31 | 0.8131 | 0.558 | **0.229** | 1.0e-04 | **CER < 23%** |
| 32 | 0.8085 | 0.557 | **0.228** | 1.0e-04 | |
| 33 | 0.8035 | 0.561 | 0.229 | 1.0e-04 | |
| 34 | 0.7992 | 0.559 | **0.228** | 1.0e-04 | |
| 35 | 0.7953 | 0.554 | **0.227** | 1.0e-04 | |
| **36** | **0.7913** | **0.552** | **0.226** | 1.0e-04 | Best CER = 22.64% |
| 37 | 0.7871 | 0.556 | 0.228 | 1.0e-04 | légère remontée |
| 38 | 0.7836 | 0.553 | 0.227 | 1.0e-04 | |
| **39** | **0.7795** | **0.550** | **0.224** | 1.0e-04 | **Best CER = 22.42%** |
| 40 | 0.7762 | 0.548 | 0.224 | 1.0e-04 | **WER < 55%** pour la 1ère fois |
| 41 | 0.7723 | 0.555 | 0.227 | 1.0e-04 | "siy est imposé à trois reprises" quasi parfait |

### Progression épisode 25 → 36

- **CER : 23.79% → 22.64%** (−1.15 pp en 11 epochs)
- **WER : 57.5% → 55.2%** (−2.3 pp)
- **Loss : 0.843 → 0.791** (baisse régulière)
- LR scheduler n'a toujours pas réduit le LR

**Vitesse de convergence ralentie** : ~−0.1 pp/epoch (vs ~−0.4 pp/epoch sur epochs 1-9).
Le modèle approche de son plateau mais progresse encore.

### Résultat final Run #4 (à compléter)

**CER actuel : 22.64% / WER : 55.2%** sur dev set (29k samples).

Comparé au point de départ (run #3 best : CER=32.3%) :
- **Gain : −9.7 pp CER en 36 epochs de fine-tune** sur 586k

### Exemples qualitatifs (epoch 11)

| REF | HYP |
|---|---|
| ralph depalma s'y est imposé à trois reprises | rac de palma sie componé à trois reprise |
| je m'y oppose régulièrement au sein de la commission des finances et en séance | jemi opose rigulièrement au seine de la commission de finals su te con fiance |

**Progression vs epoch 9** :
- "**la commission**" toujours parfait
- "**au seine de**" (vs "au seile de" à epoch 9)
- "**jemi**" très proche de "je m'y" (vs "jili" à epoch 9)
- "trois reprise" parfait

---

## Session 3 — Résultats (epochs ??-??)

**Statut** : à planifier

_À compléter si besoin d'une 3e session._

---

## Projection (à affiner)

### Basée sur la tendance epochs 1-9

Vitesse de convergence : ~−0.25 pp CER/epoch sur epochs 5-9 (slowdown normal de fine-tune).

| Epoch | CER projeté |
|---|---|
| 9 | 25.5 % (mesuré) |
| 15 | ~23-24 % |
| 20 | ~21-22 % |
| 25 | **~19-20 %** |

### Avec KenLM (post-training)

Un modèle de langue n-gram français (KenLM + pyctcdecode) gagnerait typiquement
10-15 pp de CER :

| Scénario | CER estimé |
|---|---|
| Run #4 epoch 9 (greedy) | 25.5 % (mesuré) |
| Run #4 epoch 25 (greedy, projeté) | ~19-20 % |
| Run #4 epoch 25 + KenLM | **~12-15 %** |

## Comparaison transversale Phase 2

| Run | Corpus | Epochs | Best CER (dev) | WER (dev) | Stratégie |
|---|---|---|---|---|---|
| #1 | 20k | 25 | 71.1% | 115.5% | From scratch |
| #2 | 80k | 50 | 56.9% | 95.0% | From scratch |
| #3 | 300k | 38 | 32.3% | 70.7% | From scratch |
| **#4** | **586k** | **9** (session 1) | **25.5%** | **60.8%** | **Fine-tune from #3** |

**Gain du fine-tune** : −6.8 pp CER en seulement 9 epochs (vs 38 epochs pour le run #3
from scratch qui a atteint 32.3%).

## Fichiers produits

- `/kaggle/working/run4_finetune_586k/training_state.pth` — checkpoint complet resume
- `/kaggle/working/run4_finetune_586k/best_asr.pt` — meilleur modèle (epoch 9, CER=0.255)

## Prochaines étapes

1. **Session 2** : resume epochs 10-20, objectif CER < 22%
2. **Session 3 éventuelle** : epochs 21-25, objectif CER < 20%
3. **Évaluation test** sur les 29k samples test (cellules 18-19 du notebook)
4. **KenLM** : intégrer `pyctcdecode` + modèle de langue français — devrait amener
   le CER autour de 12-15%
5. **Retour d'expérience final** : compléter ce document avec résultats test et sessions suivantes
