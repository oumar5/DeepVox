# Documentation DeepVox

Ce dossier contient la documentation complète du projet de recherche DeepVox.

## Lecture recommandée

Lire dans l'ordre :

1. [01_vision_ecosysteme.md](01_vision_ecosysteme.md) — Vision globale et concept de représentation pivot
2. [02_matrice_taches_et_architectures.md](02_matrice_taches_et_architectures.md) — Les 7 tâches cibles et 3 architectures candidates
3. [03_feuille_de_route.md](03_feuille_de_route.md) — Roadmap en 6 phases avec critères GO/NO-GO
4. [04_protocole_experimental_phase1.md](04_protocole_experimental_phase1.md) — Premier protocole expérimental à exécuter
5. [05_sources_donnees.md](05_sources_donnees.md) — Datasets francophones et corpus parallèles
6. [06_etat_de_lart.md](06_etat_de_lart.md) — Travaux liés et positionnement
7. [07_risques_et_questions_ouvertes.md](07_risques_et_questions_ouvertes.md) — Limites identifiées

## Retours d'expérience (runs d'entraînement)

- [08_retour_experience_phase1_run1.md](08_retour_experience_phase1_run1.md) — Run #1 : Condition A sur 5 000 fichiers (PER=69 %, surapprentissage, piste données)
- [09_retour_experience_phase1_run2.md](09_retour_experience_phase1_run2.md) — Run #2 : Condition A sur 20 000 fichiers (PER=62.8 %, −6.4 pp, voisement capté à 91 %)
- [10_retour_experience_phase1_run3_conditionB.md](10_retour_experience_phase1_run3_conditionB.md) — Run #3 : Condition B (delta features) sur 20k — échec (PER=65 %, pire que A), Condition B abandonnée
- [11_retour_experience_phase1_run4.md](11_retour_experience_phase1_run4.md) — Run #4 : Condition A sur 80k (PER=56 %, Top-5=80.7 %, voisement 93 %) — Phase 1 validée, passage Phase 2

## Phase 2 — ASR directe (Codec2 → texte)

- [12_retour_experience_phase2_run1.md](12_retour_experience_phase2_run1.md) — Run #1 : BiLSTM CTC sur 20k (CER=71.2 %), baseline établi, scaling données prioritaire

## Article de recherche associé

[../paper/article_fr.md](../paper/article_fr.md) — Brouillon de soumission Interspeech / ICASSP 2027.

## Origine

Cette recherche est issue du projet [SmsVox](https://github.com/oumar5/SmsVox), qui a démontré la faisabilité de Codec2 (1200 bps) pour la transmission vocale ultra-bas-débit par SMS. DeepVox explore la suite logique : utiliser cette même représentation comme entrée de modèles transformers multilingues.

## Statut

- **2026-04-16** — Création du dépôt DeepVox, migration depuis SmsVox
- **2026-04-16** — Phase 1 run #1 exécuté (Condition A, 5 k fichiers, PER=69 % — voir doc 08)
- **À venir** — Phase 1 runs #2-#5 (plus de données, Condition B, contexte élargi)
