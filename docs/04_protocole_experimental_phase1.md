# 04 — Protocole expérimental Phase 1

## Question de recherche

> Les frames Codec2 (mode 1200 bps) préservent-elles assez d'information phonétique discriminante pour qu'un classifieur supervisé puisse identifier le phonème prononcé en **français** avec une précision comparable à celle obtenue depuis des spectrogrammes mel ?

C'est la question fondatrice. Si la réponse est non, toute la suite de la recherche est invalidée. Si la réponse est oui, le passage aux Phases 2-6 est justifié.

## Choix du français comme langue d'évaluation

DeepVox cible prioritairement les communautés francophones (alignement avec SmsVox), notamment dans les régions à faible connectivité. La validation phonétique doit donc se faire sur du **français standard** dès la Phase 1, et non sur l'anglais (TIMIT) comme dans la littérature classique.

**Inventaire phonétique du français standard :** 36 phonèmes (16 voyelles dont 4 nasales, 17 consonnes, 3 semi-voyelles), encodés en SAMPA-FR ou X-SAMPA.

## Hypothèse à tester

H1 : la précision phonétique sur Common Voice French depuis frames Codec2 est ≥85 %, à comparer à ~88 % attendus avec spectrogrammes mel (estimation par analogie avec les baselines anglaises).

H0 (à rejeter) : Codec2 a éliminé trop d'information phonétique pour permettre une classification compétitive.

## Dataset principal

**Common Voice French v21.0 (mars 2025)** — version la plus récente du corpus francophone Mozilla, publiée sur Kaggle :
- Source officielle : https://commonvoice.mozilla.org/fr/datasets
- Mirror Kaggle (téléchargement sans compte Mozilla) : https://www.kaggle.com/datasets/fredrelec/common-voice-french-21-0-2025
- Volume : ~1000 h de parole francophone validée
- Locuteurs : milliers, accents variés (France, Belgique, Suisse, Canada, Afrique francophone)
- Format : MP3 16 kHz + transcriptions CSV
- Licence : CC0 (domaine public)

**Pas d'annotations phonétiques manuelles fournies.** Les annotations seront générées automatiquement via :

### Outil d'alignement : Montreal Forced Aligner (MFA)

- https://montreal-forced-aligner.readthedocs.io/
- Modèle acoustique français pré-entraîné disponible (`french_mfa`)
- Lexique de prononciation français : `french_mfa.dict` (variantes Académie + variantes courantes)
- Sortie : alignements TextGrid avec bornes phonétiques ±20 ms
- Gratuit, open-source, qualité d'alignement reconnue par la communauté

**Pipeline de préparation :**

```
Common Voice FR (MP3 16 kHz + transcription)
    ↓ resampling
WAV 8 kHz
    ↓ Montreal Forced Aligner (lexique français)
Annotations phonétiques (TextGrid)
    ↓ encodage Codec2 1200 bps
Frames .c2 (48 bits / 40 ms)
    ↓ alignement MFA → grille Codec2
Couples (frame_codec2, phonème) prêts pour entraînement
```

## Datasets complémentaires francophones

| Corpus | Source | Heures | Usage |
|---|---|---|---|
| **CSS10 French** | https://www.kaggle.com/datasets/bryanpark/french-single-speaker-speech-dataset | ~10 h | Voix unique propre, validation dev |
| **MLS French** | https://www.openslr.org/94 | ~1100 h | Audiobooks, qualité studio, fine-tuning |
| **African Accented French** | OpenSLR 57 | ~22 h | Variantes Sénégal, Maroc, RDC — robustesse régionale |
| **French Speech Recognition** | https://www.kaggle.com/datasets/unidpro/french-speech-recognition-dataset | Variable | Validation supplémentaire |
| **French Spontaneous Dialogue** | https://www.kaggle.com/datasets/nexdatafrank/french-spontaneous-dialogue-speech-dataset | Variable | Test sur parole spontanée (≠ parole lue) |
| **VoxForge French** | http://www.voxforge.org/fr | ~30 h | Historique, gratuit |

## Pré-traitement obligatoire

1. **Resampling** 16 kHz → 8 kHz (Codec2 1200 bps n'accepte que 8 kHz)
2. **Alignement forcé** via MFA pour générer les annotations phonétiques
3. **Encodage Codec2** 1200 bps de chaque énoncé via `c2enc`
4. **Mapping temporel** : chaque frame Codec2 = 40 ms ; re-quantification des bornes phonétiques sur la grille 40 ms

## Représentations à comparer

Quatre conditions expérimentales, mêmes données, mêmes splits, mêmes hyperparamètres de classifieur :

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
Linear(256 → 36)         ← 36 phonèmes français
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
| **Matrice de confusion** | 36×36 | Pour analyse qualitative |
| **Confusions saillantes** | Top 10 paires confondues | Pour identifier ce que Codec2 perd |

## Splits de données

| Split | Source | Usage |
|---|---|---|
| **train** | Common Voice FR train + MLS French train | Entraînement |
| **dev** | Common Voice FR dev + CSS10 (10 % en holdout) | Sélection de modèle, early stopping |
| **test** | Common Voice FR test | Évaluation finale (rapportée) |
| **test-OOD** | African Accented French + French Spontaneous Dialogue | Robustesse hors-distribution |

## Analyses complémentaires

1. **Ablation par champ Codec2** : entraîner un classifieur en masquant successivement LSP, pitch, énergie, voicing → quel champ porte quelle proportion d'information phonétique en français ?

2. **Robustesse au bruit** : ajouter du bruit blanc ou rose au PCM avant encodage Codec2, mesurer la chute de PER.

3. **Comparaison mode 1200 vs 700C** : refaire l'expérience avec Codec2 700C (encore plus comprimé, 28 bits/frame). Hypothèse : la chute est significative mais peut rester acceptable pour certaines tâches.

4. **Robustesse aux variantes francophones** : évaluer séparément sur African Accented French (accents sénégalais, marocains, congolais) pour vérifier qu'on ne sur-apprend pas l'accent métropolitain.

5. **Voyelles nasales** : analyse spécifique des 4 voyelles nasales du français (/ɑ̃/, /ɛ̃/, /ɔ̃/, /œ̃/) — particularité linguistique qui peut être un point de fragilité de Codec2 (bande de fréquences nasales).

6. **Comparaison transversale TIMIT (anglais)** : optionnellement, refaire l'expérience sur TIMIT pour positionner les résultats par rapport à la littérature anglophone existante.

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

- **Qualité de l'alignement MFA** : moins précis que des annotations manuelles (TIMIT), peut introduire jusqu'à ±20 ms d'erreur sur les bornes phonétiques. Atténuation : marges plus généreuses dans le calcul du PER, ou évaluation au niveau séquence plutôt que frame-par-frame
- **Biais Common Voice** : crowdsourcé, qualité audio variable, accents très divers (peut aussi être un atout pour la robustesse). Atténuation : validation supplémentaire sur MLS French (qualité studio)
- **Désalignement temporel** entre la grille MFA (variable, ~10 ms) et la grille Codec2 (40 ms fixe) — re-quantification systématique
- **Effet du resampling 16→8 kHz** : perte d'information dans les hautes fréquences avant même Codec2, particulièrement critique pour les fricatives /s/ /ʃ/ /f/. À isoler dans l'analyse
- **Voyelles nasales françaises** : Codec2 a été développé pour l'anglais ; performance possible-ment sub-optimale sur les nasales (à mesurer)

## Estimation de coût

- Compute : 1 GPU consumer (RTX 3060 ou équivalent) suffit, ~2 jours d'entraînement par condition
- Stockage : ~100 GB (Common Voice FR + MLS French + African Accented + variantes encodées)
- Temps humain : 3-5 semaines (préparation données + alignement MFA + entraînement + analyse + rédaction)
- Coût datasets : **0 €** (tout est gratuit, accès direct via Mozilla / Kaggle / OpenSLR)
