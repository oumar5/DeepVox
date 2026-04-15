# 07 — Risques et questions ouvertes

Cette page recense honnêtement ce qui peut échouer, ce qui n'est pas garanti, et ce qui reste à valider expérimentalement. À relire avant chaque phase pour ajuster les attentes.

## Risques fondamentaux (peuvent invalider toute la piste)

### R1 — Codec2 a éliminé trop d'information phonétique

Codec2 a été optimisé pour la **reconstruction perceptuelle de la parole humaine**, pas pour préserver l'information phonétique discriminante au sens machine. Les compressions agressives à 700-1200 bps lissent les transitoires et quantifient brutalement les LSP, ce qui peut effacer des distinctions phonétiques importantes.

**Indicateur de risque :** Phase 1 PER > 25 %.

**Mitigation :** ajout de delta features, test de Codec2 mode 3200 bps (moins compressé), ou abandon de la piste pure Codec2 au profit d'une représentation hybride (Codec2 + features MFCC légères).

### R2 — Pas d'information prosodique fine pour les langues tonales

Codec2 encode le pitch sur 7 bits par frame (~128 valeurs). Pour les langues tonales (mandarin, vietnamien, yoruba, langues africaines à tons), cette résolution peut être insuffisante pour distinguer les tons (qui portent du sens lexical).

**Indicateur :** dégradation marquée des résultats sur paires impliquant une langue tonale (FR↔ZH par exemple) en Phase 5.

**Mitigation :** augmenter la résolution du pitch via post-traitement, ou pour ces langues spécifiques, conserver le pitch en pleine résolution en parallèle des frames Codec2.

### R3 — Codec2 8 kHz inadapté pour certaines langues

Codec2 1200/700C n'opère qu'en 8 kHz. Cela coupe les fréquences au-dessus de 4 kHz (Nyquist), où vivent par exemple les fricatives sifflantes (/s/, /ʃ/, /f/) qui peuvent être confondues sans énergie haute fréquence.

**Indicateur :** confusions systématiques sur fricatives dans la matrice de confusion Phase 1.

**Mitigation :** Codec2 propose des modes 16 kHz (700C-16k expérimental). À évaluer si le bottleneck est confirmé.

## Risques techniques (gérables mais à surveiller)

### R4 — Quantité de données parallèles insuffisante pour les paires hors anglais

CVSS et MuST-C couvrent essentiellement des paires impliquant l'anglais. Pour FR↔AR, FR↔WO (wolof), ou autres paires à forte valeur pour SmsVox en Afrique, les données sont rares.

**Mitigation :** pivot via anglais (FR→EN→AR), ou augmentation par TTS multilingue (génération de paires synthétiques).

### R5 — Coût computationnel des Phases 5-6

Les modèles end-to-end S2TT/S2ST nécessitent traditionnellement 8-16 GPU pendant des semaines. Si le budget compute reste à 1-2 GPU consumer, il faudra accepter des modèles plus petits que la SOTA et compenser par des techniques d'efficacité (LoRA, quantization-aware training, knowledge distillation depuis Whisper).

### R6 — Évaluation S2ST sans baseline directe sur Codec2

Comme aucun système publié n'utilise Codec2, il n'y a pas de leaderboard où nous positionner. Nous devons construire notre propre pipeline d'évaluation (encodage Codec2 → modèle existant Whisper pour ASR-BLEU sur la sortie audio reconstruite).

**Mitigation :** publier les scripts d'évaluation, viser la reproductibilité plus que la SOTA absolue.

## Risques produit / éthique

### R7 — Voice cloning et usage malveillant

La Phase 6 (préservation du timbre) ouvre la porte au clonage vocal non consenti. Risque réel pour SmsVox si un utilisateur peut faire dire à la voix d'un autre des choses qu'il n'a pas dites.

**Mitigation :**
- Watermarking audio (AudioSeal de Meta) systématique
- Limitation produit : voice cloning uniquement entre messages **du même expéditeur**, pas inter-locuteurs
- Documentation éthique dans tout artefact publié

### R8 — Biais de qualité par langue

Les modèles multilingues sous-performent systématiquement sur les langues sous-représentées dans les données d'entraînement. Si SmsVox cible des régions africaines, livrer un modèle qui marche bien en français mais mal en wolof reproduit une iniquité d'accès.

**Mitigation :** évaluation explicite par langue, communication transparente des limites, priorisation des langues sous-ressourcées dans la collecte de données complémentaires.

## Questions ouvertes à trancher expérimentalement

### Q1 — Quel mode Codec2 choisir comme défaut ?

| Mode | Bitrate | Frame | Hypothèse |
|---|---|---|---|
| 3200 | 3200 bps | 20 ms | Plus de qualité, mais 2× plus de tokens |
| 1200 | 1200 bps | 40 ms | Compromis SmsVox actuel |
| 700C | 700 bps | 40 ms | Maximalement compact |

À évaluer en Phase 1 sur le critère PER vs. compacité.

### Q2 — Faut-il un tokenizer en plus des frames brutes ?

Deux options :
- Utiliser les 48 bits de chaque frame comme features continues (entrée linéaire 48-dim au transformer)
- Quantifier via k-means les frames en N clusters (typiquement 4096 ou 8192) pour avoir des tokens discrets utilisables avec embeddings appris

À tester comparativement en Phase 2.

### Q3 — Le transfert entre tâches justifie-t-il un backbone partagé ?

L'architecture C parie sur un transfert positif. Mais si les tâches divergent trop (ASR ≠ TTS dans leur structure), un backbone partagé peut sous-performer des modèles spécialisés.

À mesurer en Phase 6 par comparaison avec les modèles indépendants des Phases 2/3/5.

### Q4 — Préservation prosodique en S2ST

La traduction parole-à-parole devrait-elle préserver le rythme et l'intonation du locuteur source, ou produire une prosodie naturelle pour la langue cible ? Question ouverte dans la littérature (cf. SeamlessExpressive vs. Translatotron 2).

À expliciter comme choix de design dès le début de la Phase 5.

### Q5 — Latence inférence sur mobile

Cible utilisateur : <2 secondes pour traduire 5 secondes de parole sur smartphone bas de gamme. À mesurer dès la Phase 2 sur device réel, pas seulement en théorie sur GPU.

## Plan de mitigation global

| Risque | Probabilité | Impact | Action préventive |
|---|---|---|---|
| R1 (info phonétique) | Moyenne | Critique | Phase 1 prioritaire avec critère GO/NO-GO clair |
| R2 (langues tonales) | Élevée | Modéré | Inclure une langue tonale dès Phase 5 |
| R3 (8 kHz fricatives) | Moyenne | Modéré | Analyse confusion Phase 1, évaluer Codec2 16k |
| R4 (données) | Élevée | Modéré | Identifier paires alternatives dès maintenant |
| R5 (compute) | Élevée | Modéré | Architecture compacte par design, distillation |
| R6 (évaluation) | Certaine | Faible | Construire pipeline éval reproductible Phase 4 |
| R7 (clonage) | Moyenne | Critique (réputation) | Watermarking dès Phase 6 |
| R8 (biais langues) | Élevée | Modéré | Évaluation par langue, transparence |

## Critères d'arrêt de la recherche

La piste doit être abandonnée si :
- Phase 1 : PER > 35 % même avec delta features et ablations exhaustives
- Phase 2 : WER > 50 % sur LibriSpeech test-clean (ASR inutilisable)
- Phase 5 : BLEU end-to-end systématiquement <50 % de la baseline cascade

Tout résultat intermédiaire négatif **doit être documenté et publié** s'il est instructif (par exemple : "Codec2 préserve l'information segmentale mais pas suprasegmentale" est un résultat négatif publiable et utile à la communauté).
