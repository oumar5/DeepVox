.PHONY: install test prepare mfa train train-all ablation clean

VENV = .venv/bin
PYTHON = $(VENV)/python
CONDA_MFA = eval "$$(~/miniconda3/bin/conda shell.zsh hook)" && conda activate mfa

DATA_DIR = data/prepared
MFA_DIR = data/mfa-output
CV_DIR = data
OUTPUT_DIR = outputs/phase1
MAX_SAMPLES ?= 5000

# ──────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────

install:
	python3.12 -m venv .venv
	$(VENV)/pip install --upgrade pip setuptools wheel
	$(VENV)/pip install --only-binary=:all: llvmlite numba librosa
	$(VENV)/pip install -e ".[dev]"
	$(VENV)/pip install pycodec2
	@echo "✅ Installation terminée"

# ──────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────

test:
	$(PYTHON) -m pytest tests/ -v

test-cov:
	$(PYTHON) -m pytest tests/ -v --cov=deepvox --cov-report=term-missing

# ──────────────────────────────────────────────
# Data preparation
# ──────────────────────────────────────────────

prepare:
	$(PYTHON) scripts/prepare_data.py \
		--cv-dir $(CV_DIR) \
		--output-dir $(DATA_DIR) \
		--max-samples $(MAX_SAMPLES) \
		--skip-mfa

mfa:
	$(CONDA_MFA) && mfa align \
		$(DATA_DIR) \
		french_mfa \
		french_mfa \
		$(MFA_DIR) \
		--num_jobs 4 \
		--clean \
		--overwrite

# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────

train:
	$(PYTHON) scripts/phase1_phoneme_classification.py \
		--data-dir $(DATA_DIR) \
		--textgrid-dir $(MFA_DIR) \
		--condition A \
		--output-dir $(OUTPUT_DIR)

train-all:
	$(PYTHON) scripts/run_all_conditions.py \
		--data-dir $(DATA_DIR) \
		--textgrid-dir $(MFA_DIR) \
		--output-dir $(OUTPUT_DIR)

ablation:
	$(PYTHON) scripts/ablation_codec2_fields.py \
		--data-dir $(DATA_DIR) \
		--textgrid-dir $(MFA_DIR) \
		--output-dir $(OUTPUT_DIR)/ablation

# ──────────────────────────────────────────────
# Pipeline complet
# ──────────────────────────────────────────────

pipeline: prepare mfa train-all ablation
	@echo "✅ Pipeline Phase 1 terminé"

# ──────────────────────────────────────────────
# Cleanup
# ──────────────────────────────────────────────

clean:
	rm -rf outputs/ .pytest_cache/ .mypy_cache/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
