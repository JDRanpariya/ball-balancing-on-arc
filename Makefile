# Ball-on-Arc Benchmark - Makefile
# Common commands for development, training, evaluation, and reproduction

.PHONY: install figures eval-sim train docker clean help figures-docker eval-sim-docker table stats check smoke train-worldmodel train-worldmodel-v12 train-iql train-ppo

help:
	@echo "Ball-on-Arc Benchmark"
	@echo "====================="
	@echo ""
	@echo "Setup:"
	@echo "  make install         Install balancer package + deps"
	@echo "  make install-full    Install with all optional deps (RL, MPC, MPPI)"
	@echo ""
	@echo "Paper Reproduction:"
	@echo "  make figures         Generate all paper figures"
	@echo "  make figures-main    Main paper figures only"
	@echo "  make figures-supp    Supplementary figures only"
	@echo "  make table           Print results tables"
	@echo ""
	@echo "Evaluation:"
	@echo "  make eval-sim        Run simulation evaluation (all controllers)"
	@echo "  make eval-hw         Run hardware evaluation (all controllers)"
	@echo ""
	@echo "Training:"
	@echo "  make train-ppo       Train PPO (BZ+DR)"
	@echo "  make train-iql       Train IQL offline"
	@echo "  make train-worldmodel Train the LSTM world model"
	@echo ""
	@echo "Docker:"
	@echo "  make docker          Build + run figure generation in Docker"
	@echo "  make figures-docker  Run inside container: figures + copy to /output"
	@echo "  make eval-sim-docker Run inside container: eval-sim + copy to /output"
	@echo ""
	@echo "Misc:"
	@echo "  make clean           Remove generated artifacts"
	@echo "  make check           Run reproducibility checklist"
	@echo "  make smoke           Run unit + benchmark smoke tests (all 13 controllers load)"

# ============================================================
# SETUP
# ============================================================

install:
	pip install -e balancer/

install-full:
	pip install -e "balancer/[all]"

# ============================================================
# PAPER FIGURES
# ============================================================

FIG_SCRIPT = paper/ram/regenerate_all_figures.py
MAIN_FIG_SCRIPT = paper/ram/regenerate_figures.py
SUPP_FIG_SCRIPT = paper/ram/regenerate_extra_figures.py

# All figures (main + supplementary). Thin wrapper runs the two below.
figures:
	python $(FIG_SCRIPT)

# Main-paper figures only (Table I boxplot, SR ranking, difficulty breakdown).
figures-main:
	python $(MAIN_FIG_SCRIPT)

# Supplementary figures only (Appendices A-I).
figures-supp:
	python $(SUPP_FIG_SCRIPT)

# Verify every Table I cell matches the committed hardware data (13/13 rows).
table:
	python paper/verify_table1.py

# Reproduce Table I + Section V statistics (Wilson CIs, Mann-Whitney, Fisher).
stats:
	python paper/scripts/compute_statistics.py

# ============================================================
# EVALUATION
# ============================================================

eval-sim:
	cd evaluation && PYTHONPATH=. python scripts/eval_sim.py --exp 1 --action-type cont --trials 50

eval-hw:
	cd evaluation && PYTHONPATH=. python scripts/eval.py --exp 1 --action-type cont --trials 50

# ============================================================
# TRAINING
# ============================================================

train-ppo:
	cd training/scripts && python train_all_cont.py --algos ppo --rewards balanced --dr --blind-zone --timesteps 3000000 --tag ppo_cont_balanced_dr_bz

train-iql:
	cd training/offline_rl && python offline_train_cont.py --algo iql --n-steps 500000 --gpu

# v1 = PPO-WM's world model, trained on the pre_neg 1M collection (script default)
train-worldmodel:
	cd training/scripts && python train_world_model.py --epochs 50 --export --export-version v1 --no-wandb

# v12 = MPPI's world model, trained on the calib196_160 1M collection
train-worldmodel-v12:
	cd training/scripts && python train_world_model.py --epochs 50 --data-path ../../data/dataset/arcball_cont_1M_calib196_160.h5 --export --export-version v12 --no-wandb

# ============================================================
# DOCKER
# ============================================================

docker:
	docker build -t balancer-repro .
	mkdir -p output
	docker run -v $(PWD)/output:/output balancer-repro

# `docker run ... balancer-repro make <target>` overrides the image's default
# CMD, so the plain `figures`/`eval-sim` targets never reach the "copy to
# /output" step and results die with the container. These *-docker wrappers
# run the real target and then copy results to /output when it exists (i.e.
# when run inside the container with -v "$(pwd)/output:/output" mounted),
# mirroring what the default CMD already does for `make figures`.
figures-docker: figures
	@if [ -d /output ]; then \
		mkdir -p /output && \
		cp -r paper/ram/figures/* /output/ && \
		cp -r paper/ram/supplementary/figures /output/supplementary_figures && \
		echo "Figures copied to /output"; \
	fi

eval-sim-docker: eval-sim
	@if [ -d /output ]; then \
		mkdir -p /output/eval_sim_results && \
		cp -r evaluation/results/sim/* /output/eval_sim_results/ && \
		echo "Sim results copied to /output/eval_sim_results"; \
	fi

# ============================================================
# MISC
# ============================================================

clean:
	find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.pyc' -delete 2>/dev/null || true
	rm -rf output/

smoke:
	@echo "Smoke tests: unit tests + benchmark controller integration (all 13 load)."
	python -m pytest balancer/tests -q
	cd evaluation && PYTHONPATH=. python -m pytest tests/ -q

check:
	@echo "Reproducibility Check - verifies data integrity and key dependencies."
	@echo "See reproducibility_guide.md for the full reproduction pipeline."
	@echo "=========================================================="
	@echo ""
	@python -c "import numpy; print('  [OK] numpy', numpy.__version__)"
	@python -c "import scipy; print('  [OK] scipy', scipy.__version__)"
	@python -c "import matplotlib; print('  [OK] matplotlib', matplotlib.__version__)"
	@python -c "import balancer; print('  [OK] balancer installed')"
	@test -f paper/data/exp1_hardware/consolidated.json && echo "  [OK] consolidated.json exists" || { echo "  [FAIL] consolidated.json missing"; exit 1; }
	@# 21 entries = 13 main controllers + 8 ablation variants (paper §III, Table I).
	@python -c "import json; d=json.load(open('paper/data/exp1_hardware/consolidated.json')); assert len(d['controllers'])==21; print('  [OK] 21 controller entries (13 main + 8 ablation)')"
	@python -c "import json; d=json.load(open('paper/data/exp1_hardware/consolidated.json')); assert all(len(v)==50 for v in d['controllers'].values()); print('  [OK] all have 50 trials')"
	@python paper/verify_table1.py >/dev/null 2>&1 && echo "  [OK] Table I matches data (SR, Wilson CI, settling, effort)" || { echo "  [FAIL] Table I mismatch"; exit 1; }
	@echo ""
	@echo "All checks passed!"
