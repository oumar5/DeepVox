# 04 — Protocole expérimental Phase 1

## Question de recherche

> Les frames Codec2 (mode 1200 bps) préservent-elles assez d'information phonétique discriminante pour qu'un classifieur supervisé puisse identifier le phonème prononcé avec une précision comparable à celle obtenue depuis des spectrogrammes mel ?

C'est la question fondatrice. Si la réponse est non, toute la suite de la recherche est invalidée. Si la réponse est oui, le passage aux Phases 2-6 est justifié.

## Hypothèse à tester

H1 : la précision phonétique sur TIMIT depuis frames Codec2 est ≥85 %, à comparer à ~88 % avec spectrogrammes mel et ~70 % avec PCM brut (baselines de référence dans la littérature).

H0 (à rejeter) : Codec2 a éliminé trop d'information phonétique pour permettre une classification compétitive.

## Dataset

**TIMIT (Linguistic Data Consortium, LDC93S1)** — corpus de référence pour l'évaluation phonétique :
- 6 300 phrases en anglais américain
- 630 locuteurs (438 H, 192 F)
- Annotations phonétiques manuelles au niveau du frame
- Inventaire : 61 phonèmes (réduits classiquement à 39 pour évaluation)
- Découpage standard train/dev/test fourni

**Note de licence :** TIMIT est payant via LDC (~250 USD pour les institutions non-membres). Alternative gratuite : **TIMIT-like portion of the Buckeye Corpus** ou **L2-ARCTIC**, avec réannotation phonétique automatique via Montreal Forced Aligner.

**Pré-traitement obligatoire :**
1. Resampling 16 kHz → 8 kHz (Codec2 1200 bps n'accepte que 8 kHz)
2. Encodage Codec2 1200 bps de chaque énoncé
3. Alignement temporel : chaque frame Codec2 = 40 ms ; mapping vers les annotations phonétiques en re-quantifiant les bornes phonétiques sur la grille 40 ms

## Représentations à comparer

Trois conditions expérimentales, mêmes données, mêmes splits, mêmes hyperparamètres de classifieur :

| Condition | Entrée par frame | Dimension |
|---|---|---|
| **A — Codec2 raw** | 48 bits Codec2 décomposés en champs (LSP×36, pitch×7, énergie×5) | 48 features |
| **B — Codec2 + delta** | A + différence avec frame précédente | 96 features |
| **C — Spectrogramme mel** (baseline) | 80 mel bands sur fenêtre 25 ms / hop 10 ms, repackagé sur grille 40 ms | 80 features |
| **D — PCM brut** (baseline basse) | Échantillons PCM 8 kHz dans la fenêtre de 40 ms | 320 features |

## Architecture du classifieur

Choix d'un modèle volontairement simple pour éviter que la capacité du classifieur masque les différences entre représentations :

```
Input frame features
    ↓
BiLSTM 2 couches, hidden=256
    ↓
Linear(256 → 39)
    ↓
Softmax → phonème
```

~1.5 M paramètres. Identique pour les 4 conditions.

**Entraînement :** Adam, lr=1e-3, batch=64, max 50 epochs, early stopping sur dev set.

## Métriques

| Métrique | Calcul | Cible |
|---|---|---|
| **Phone Error Rate (PER)** | 1 − (frames correctement classifiés / total) | ≤15 % (cond. A ou B) |
| **Précision macro** | Moyenne par phonème | ≥85 % |
| **Matrice de confusion** | 39×39 | Pour analyse qualitative |
| **Confusions saillantes** | Top 10 paires confondues | Pour identifier ce que Codec2 perd |

## Analyses complémentaires

1. **Ablation par champ Codec2** : entraîner un classifieur en masquant successivement LSP, pitch, énergie, voicing → quel champ porte quelle proportion d'information phonétique ?

2. **Robustesse au bruit** : ajouter du bruit blanc ou rose au PCM avant encodage Codec2, mesurer la chute de PER.

3. **Comparaison mode 1200 vs 700C** : refaire l'expérience avec Codec2 700C (encore plus comprimé, 28 bits/frame). Hypothèse : la chute est significative mais peut rester acceptable pour certaines tâches.

4. **Test inter-langues** : si possible, refaire l'expérience sur du français (corpus BREF ou ESTER) pour vérifier que la conclusion n'est pas spécifique à l'anglais.

## Livrables

- Script reproductible (`phase1_phoneme_classification.py`) prenant un dataset annoté en entrée
- Rapport `phase1_results.md` avec tableaux, matrices de confusion, conclusions
- Modèles entraînés versionnés
- Décision documentée : **GO / NO-GO** pour la Phase 2

## Critères de décision

| Résultat | Interprétation | Action |
|---|---|---|
| PER ≤15 % | Hypothèse validée | GO Phase 2 |
| 15 % < PER ≤25 % | Information dégradée mais utilisable | GO conditionnel, attentes ajustées |
| PER > 25 % | Information insuffisante | Tester delta features (cond. B), Codec2 700C, ajout MFCC. Si toujours échec : NO-GO, pivoter vers une représentation hybride |

## Risques spécifiques à cette phase

- **Désalignement temporel** entre la grille TIMIT (variable) et la grille Codec2 (40 ms fixe) — peut introduire jusqu'à ±20 ms d'erreur d'annotation
- **Biais d'évaluation** : TIMIT est anglais, parole lue, locuteurs natifs — résultats potentiellement non transférables au français spontané ou à des langues tonales
- **Effet du resampling 16→8 kHz** : perte d'information dans les hautes fréquences avant même Codec2, à isoler dans l'analyse

## Estimation de coût

- Compute : 1 GPU consumer (RTX 3060 ou équivalent) suffit, ~2 jours d'entraînement par condition
- Stockage : <50 GB pour TIMIT + variantes encodées
- Temps humain : 2-4 semaines (préparation données, entraînement, analyse, rédaction)
