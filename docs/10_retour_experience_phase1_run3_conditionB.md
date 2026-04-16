# 10 — Retour d'expérience Phase 1, run #3 (Condition B — delta features)

**Date** : 2026-04-16
**Condition** : B (Codec2 raw + delta features, 96 features)
**Statut** : exploratoire — test de l'hypothèse "les deltas aident à discriminer les phonèmes"

## Hypothèse testée

Selon le protocole (doc 04), la Condition B ajoute les **différences entre frames
consécutives** aux features Codec2 brutes. L'intuition standard en traitement du signal :
les delta et delta-delta MFCC améliorent systématiquement l'accuracy phonétique de
3-8 points sur les spectrogrammes.

**Hypothèse** : les delta Codec2 devraient apporter un gain similaire (~3-5 pp PER).

## Configuration

Strictement identique au run #2 (Condition A), seule la feature change.

| Paramètre | Valeur |
|---|---|
| Corpus | 20 000 fichiers (identique run #2) |
| Paires alignées | 19 742 (identique) |
| Split | train=15792 / dev=1974 / test=1975 |
| Frames totales | train=1.13M / dev=160k / test=160k |
| **Features** | **96 (48 Codec2 + 48 delta)** |
| Contexte | 5 frames chaque côté |
| Modèle | BiLSTM 2 couches, hidden=256 |
| Paramètres | 2 275 372 (+49 k vs 2 226 220 à cause du premier W) |
| Optimizer | AdamW, lr=2e-3, wd=1e-2 |
| Batch size | 256 |
| Max epochs | 20 |
| Early stopping | patience=5 |
| Durée réelle | ~2h |

## Résultats bruts

### Évolution par epoch

| Epoch | Train loss | Train acc | Dev PER | Dev acc | lr |
|---|---|---|---|---|---|
| 1  | 2.4750 | 0.291 | 0.676 | 0.324 | 2.0e-03 |
| 2  | 2.2581 | 0.347 | 0.669 | 0.331 | 2.0e-03 |
| 3  | 2.1760 | 0.369 | 0.669 | 0.331 | 2.0e-03 |
| 6  | 2.0771 | 0.395 | 0.662 | 0.338 | 2.0e-03 |
| 9  | 2.0335 | 0.405 | 0.665 | 0.335 | **1.0e-03** |
| 10 | 1.9104 | 0.438 | 0.660 | 0.340 | 1.0e-03 |
| **11** | 1.8376 | 0.456 | **0.659** | **0.341** | 1.0e-03 |
| 12 | 1.7952 | 0.468 | 0.664 | 0.336 | 1.0e-03 |
| 14 | 1.7331 | 0.485 | 0.668 | 0.332 | **5.0e-04** |
| 16 | 1.5913 | 0.524 | 0.668 | 0.332 | 5.0e-04 |

Early stopping epoch 16. Meilleur dev PER : **epoch 11** (0.659).

### Métriques test finales

| Métrique | **Condition A** (run #2) | **Condition B** (ce run) | Δ |
|---|---|---|---|
| PER | **0.628** | 0.650 | **+2.2 pp (pire)** |
| Top-1 accuracy | **37.2 %** | 35.0 % | −2.2 pp |
| Top-3 accuracy | **62.7 %** | 60.5 % | −2.2 pp |
| Top-5 accuracy | **74.4 %** | 72.6 % | −1.8 pp |
| F1 macro | **0.250** | 0.229 | −2.1 pp |
| F1 weighted | **0.356** | 0.334 | −2.2 pp |

## Analyse par groupe — structure identique, tout légèrement dégradé

### Voisement

| Groupe | Condition A | Condition B |
|---|---|---|
| Voisé (group acc) | 90.9 % | 90.6 % |
| Non-voisé (group acc) | 71.6 % | 68.8 % |

### Groupes IPA

| Groupe | A exact | B exact | Δ |
|---|---|---|---|
| voyelles_orales | 37.8 % | 36.0 % | −1.8 pp |
| occlusives | 37.4 % | 34.9 % | −2.5 pp |
| liquides_laterales | 42.7 % | 41.3 % | −1.4 pp |
| fricatives | 46.7 % | 43.1 % | −3.6 pp |
| voyelles_nasales | 36.3 % | 31.7 % | −4.6 pp |
| nasales_consonnes | 25.0 % | 23.3 % | −1.7 pp |
| bruit (spn) | 1.4 % | 2.2 % | +0.8 pp (anecdotique) |

**Tous les groupes principaux sont dégradés.** Pas de compensation : aucun type de
phonème ne bénéficie des delta. Les nasales (−4.6 pp) et fricatives (−3.6 pp) sont les
plus affectées.

## Interprétation — pourquoi les deltas ne marchent pas ici

### Les delta MFCC marchent parce que MFCC est continu

Les MFCC sont des valeurs réelles continues. Leur dérivée temporelle capture la
**dynamique spectrale** (transitions entre phonèmes, qualité articulatoire).

### Les Codec2 bits sont discrets

Une frame Codec2 1200 bps = 48 **bits** (0 ou 1). La "différence" entre deux frames
consécutives donne :
- `+1` (bit passe de 0 à 1)
- `0` (bit inchangé)
- `-1` (bit passe de 1 à 0)

Pour chacun des 48 bits, indépendamment. Pas de courbe lisse à dériver, pas de
« vitesse de changement » — juste un signal ternaire bruité qui confond le classifieur.

### Hypothèse alternative : sur-paramétrage

Le modèle passe de 2.23 M à 2.28 M paramètres (+49 k), mais la première couche LSTM
doit apprendre à utiliser 96 features au lieu de 48 avec la même capacité hidden=256.
Possible que le gradient soit diffusé sur plus de paramètres sans gain d'information.

### Hypothèse alternative : redondance

Les delta sont **fonctionnellement redondants** avec le contexte temporel déjà vu
par le BiLSTM. Le LSTM apprend *déjà* les transitions entre frames dans ses gates —
lui donner explicitement `x[t] - x[t-1]` est redondant et peut introduire du bruit.

### Signal qui aurait pu marcher (à tester plus tard)

- **Pooled delta** : moyenne/max sur fenêtre glissante (plus robuste au bruit)
- **Delta champs structurés** : différence au niveau des champs LSP/pitch/energy
  plutôt que bit-par-bit
- **Deltas de features continues** : si on dé-quantifiait Codec2 en reconstruisant
  les LSP réels, les deltas pourraient avoir du sens

## Conclusion pour le protocole

### La Condition B du protocole initial est abandonnée

Elle ne produit aucun gain. Inutile de la retester sur 80k / 320k / corpus complet.

### Les Conditions C (mel) et D (PCM raw) restent pertinentes

Le protocole prévoit 4 conditions. C (mel spectrogram baseline) est indispensable
pour positionner Codec2 vs représentation classique. D (PCM brut) comme lower bound.

### Ce qu'on retient scientifiquement

1. **Les delta features ne transfèrent pas de MFCC vers Codec2.** Le fait que ça
   fonctionne sur MFCC/spectrogrammes ne garantit rien sur une représentation discrète.
2. **Le BiLSTM capture déjà la dynamique temporelle** via ses gates — pas besoin de
   l'expliciter.
3. **La redondance de features peut dégrader** même dans un espace à haute dimension.

Cela **renforce l'intérêt** de Codec2 raw : pas besoin de pré-processing avancé, le
signal est déjà suffisamment compact et informatif.

## Leçons techniques

1. **Condition B plus lente** à cause de 96 features au lieu de 48 : ~4 batch/s au lieu
   de ~10, soit ~2.3× plus lent en pratique.
2. **Même rythme d'overfitting** : train acc monte à 52.4 % à l'epoch 16, alors que A
   était à 49.7 % à l'epoch 17. Les deltas ne protègent pas de l'overfitting.

## Fichiers produits

- `outputs/phase1_run2_B/best_model_B.pt` — 2.3 M params (non utilisable en production
  car sous-performant vs A)
- `outputs/phase1_run2_B/phase1_results_B.md` — rapport détaillé
- `outputs/phase1_run2_B/train_B.log` — logs complets

## Décisions

1. **Condition B définitivement abandonnée.** Inutile de dépenser du compute sur B
   à plus grande échelle.
2. **Continuer avec Condition A sur 80k** (run #4 lancé en parallèle — MFA déjà terminé).
3. **Ajouter Condition C (mel baseline)** dans un run futur pour le comparatif final
   exigé par le protocole.

## Prochaine étape

Run #4 (Condition A sur 80k fichiers) en cours.
