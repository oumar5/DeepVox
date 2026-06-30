---
project_name: DeepVox
user_name: Benloloumar
date: '2026-06-30'
sections_completed:
  - technology_stack
  - language_rules
  - framework_rules
  - testing_rules
  - code_quality
  - project_architecture
  - critical_rules
existing_patterns_found: 28
---

# Project Context for AI Agents

_Règles critiques et patterns que tout agent IA doit respecter lors de l'implémentation de code dans DeepVox. Focalisé sur les détails non-évidents._

---

## Technology Stack & Versions

### Core

| Technologie | Version | Rôle |
|---|---|---|
| Python | >=3.10 (cible 3.12) | Langage principal |
| PyTorch | >=2.2 | Framework ML |
| torchaudio | >=2.2 | Audio processing |
| Codec2 | 1200 bps mode | Encodage vocal déterministe |

### Dépendances clés

| Package | Version | Usage |
|---|---|---|
| transformers | >=4.40 | Modèles pré-entraînés (optionnel) |
| librosa | >=0.10 | Chargement/analyse audio |
| numpy | >=1.26 | Tenseurs numériques |
| scipy | >=1.12 | Traitement signal |
| praatio | - | Parsing TextGrid (MFA) |
| pycodec2 | - | Binding Codec2 Python |

### Outils de développement

| Outil | Version | Config |
|---|---|---|
| ruff | >=0.4 | line-length=100, target-version="py310" |
| mypy | >=1.10 | Type checking strict |
| pytest | >=8.0 | Tests unitaires, --strict-markers |
| pytest-cov | >=4.1 | Couverture de code |

### Contraintes critiques

- **CTC Loss ne fonctionne PAS sur MPS (Apple Silicon)** — toujours utiliser CUDA/CPU
- **Codec2 requiert un backend** : pycodec2 OU les binaires CLI (c2enc/c2dec)
- **Audio DOIT être 8 kHz mono int16** pour Codec2 1200 bps
- **num_workers=0 obligatoire** sur Kaggle (multiprocessing incompatible)

---

## Règles Python — Spécifiques au projet

### Import obligatoire en tête de TOUT fichier

```python
from __future__ import annotations
```

### Syntaxe Python 3.10+

- Utiliser `X | None` au lieu de `Optional[X]`
- Utiliser `list[Path]` au lieu de `List[Path]`
- Utiliser `str | Path` au lieu de `Union[str, Path]`
- `zip(..., strict=True)` quand les longueurs doivent correspondre

### Organisation des imports (ordre strict)

1. Standard library (`pathlib`, `subprocess`, `tempfile`)
2. Third-party (`numpy`, `torch`, `librosa`)
3. Local (`from deepvox.codec2.encoder import ...`)

### Type annotations

- **TOUJOURS** annoter les arguments et retours de fonctions publiques
- Retour `-> None` explicite pour les fonctions sans retour
- Types tensoriels : `torch.Tensor`, `np.ndarray` (jamais `Any`)

### Docstrings — Style Google

```python
def encode_pcm(pcm: np.ndarray) -> np.ndarray:
    """Encode raw PCM samples to Codec2 1200 bps frames.

    Args:
        pcm: int16 PCM audio at 8 kHz, shape (n_samples,).

    Returns:
        Raw frame bytes as uint8 array, shape (n_frames, 6).
    """
```

---

## Architecture & Conventions de code

### Structure des modules

```
src/deepvox/
├── codec2/       → Wrappers Codec2 (encode/decode)
├── data/         → Datasets PyTorch, preprocessing, tokenizers
├── models/       → Architectures (classifiers, ASR)
├── training/     → Boucles d'entraînement par phase
└── eval/         → Métriques (PER, WER, CER, F1)
```

### Naming conventions

| Élément | Convention | Exemple |
|---|---|---|
| Fichiers | snake_case | `phoneme_classifier.py` |
| Classes | PascalCase | `PhonemeClassifier` |
| Fonctions | snake_case | `encode_pcm()` |
| Constantes | UPPER_SNAKE_CASE | `SAMPLE_RATE = 8000` |
| Privé | _leading_underscore | `_encode_pycodec2()` |
| Tests | test_*.py | `test_model.py` |

### Constantes du domaine (NE PAS modifier)

```python
SAMPLE_RATE = 8000          # Codec2 1200 bps = 8 kHz obligatoire
FRAME_DURATION_MS = 40      # Trame Codec2 = 40 ms
FRAME_BYTES = 6             # 48 bits par trame (1200 bps × 40 ms)
NUM_PHONEMES = 45           # Phonèmes IPA français (MFA french_mfa)
VOCAB_SIZE = 49             # Tokenizer caractères (blank + unk + 47 chars)
```

### Pattern Dataset

- Hériter de `torch.utils.data.Dataset`
- `__getitem__` retourne un tuple `(features, labels)` ou `(features, labels, lengths)`
- Gestion silencieuse des fichiers corrompus (return None + filtrage)
- Enum `Condition` pour les conditions expérimentales

### Pattern modèle

- Hériter de `nn.Module`
- Méthode `count_parameters() -> int` obligatoire
- Forward signature : `(x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor`
- Greedy decode dans le modèle pour l'inférence ASR

---

## Règles de test

### Framework : pytest

- Fichiers dans `tests/test_*.py`
- Classes de test : `TestNomModule`
- Pas de fixtures complexes — données synthétiques inline
- `@pytest.mark.parametrize` pour les conditions multiples
- Assertions de forme : `assert tensor.shape == (batch, seq, dim)`
- Assertions de plage pour les paramètres : `assert 500_000 < params < 10_000_000`

### Ce qu'on teste

- Formes de sortie des modèles pour toutes les conditions d'entrée
- Nombre de paramètres (borne réaliste)
- Fonctions de métriques avec des cas connus
- Tokenizer : encodage/décodage roundtrip

### Ce qu'on ne teste PAS

- Convergence du training (trop long, fait dans les notebooks)
- Intégration Codec2 CLI (dépendance système)
- Données réelles (Common Voice non inclus dans le repo)

---

## Workflow de développement

### Commandes Makefile

```bash
make install    # Setup venv + dépendances
make test       # pytest
make test-cov   # pytest avec couverture
make lint       # ruff check + fix
make train      # Entraînement Phase 1
```

### Convention de commit

- Messages en français ou anglais
- Préfixes : `feat:`, `fix:`, `docs:`, `refactor:`, `test:`

### Expérimentation

- Chaque run produit un document dans `docs/XX_retour_experience_*.md`
- Notebooks Kaggle pour le GPU (pas d'entraînement local sur MPS pour CTC)
- Checkpoints sauvés dans `checkpoints/` avec schéma : `best_{task}_run{N}_{samples}.pt`

---

## Règles critiques — NE PAS OUBLIER

### ❌ Anti-patterns à ÉVITER

1. **Ne JAMAIS utiliser MPS pour CTC** — la loss CTC n'est pas implémentée sur Apple Silicon
2. **Ne JAMAIS changer SAMPLE_RATE** — Codec2 1200 bps est fixé à 8 kHz
3. **Ne JAMAIS utiliser SAMPA** — le projet utilise IPA (migration effectuée)
4. **Ne JAMAIS utiliser Adam** — utiliser AdamW avec weight_decay=1e-2
5. **Ne JAMAIS parser les TextGrid manuellement** — utiliser praatio
6. **Ne JAMAIS utiliser tqdm classique** — utiliser `tqdm.auto` (détection notebook/terminal)
7. **Ne JAMAIS inclure /data/ dans git** — les données sont locales (>50 Go)

### ✅ Patterns OBLIGATOIRES

1. **Gradient clipping = 5.0** pour tout entraînement CTC
2. **Early stopping basé sur CER** (pas sur la loss) pour l'ASR
3. **gc.collect() + torch.cuda.empty_cache()** à chaque fin d'epoch sur Kaggle
4. **ReduceLROnPlateau** comme scheduler (patience=3-5)
5. **Résumé du modèle** (nombre de params) affiché au début de chaque run
6. **Logging structuré** : epoch, loss, métriques à chaque epoch

### 🔬 Contexte de recherche

- **Hypothèse centrale** : les frames Codec2 préservent assez d'information phonétique pour l'ASR
- **Résultat clé Phase 1** : voisement 93%, top-5 80.7% — confirme l'hypothèse
- **Résultat clé Phase 2** : CER 32.3% (300k samples, BiLSTM CTC) — amélioration log-linéaire avec les données
- **Prochaine étape** : fine-tune sur corpus complet (586k) + KenLM beam search
- **Question ouverte** : BiLSTM vs Conformer (étude comparative dans docs/16)

---

## Relation avec SmsVox

DeepVox produit des modèles `.tflite` que [SmsVox](https://github.com/oumar5/SmsVox) embarque. **Aucun couplage code-source** — juste des artefacts binaires téléchargeables. Ne jamais introduire de dépendance directe vers SmsVox.
