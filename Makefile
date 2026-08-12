# StatXI — shortcuts so every script is `make <name>` (no venv activation needed).
#
# Why this exists: the model/eval scripts import `from src...`, so they must run as
# `python -m src.models.<x>` FROM THE REPO ROOT; the webapp runs as `python -m
# webapp.app`; the data scripts run directly. This file remembers all of that for
# you. Each recipe calls the venv's python by its path ($(PY)), so you never have to
# `source venv/bin/activate` first.
#
# Usage:  make            -> shows this menu
#         make montecarlo -> runs one script
# (Run from the StatXI folder. If `make webapp` says "port in use", see `make kill`.)

PY := venv/bin/python           # the interpreter inside the project's virtualenv

# .PHONY tells make these targets are command names, not files to build — so it
# always runs them and never gets confused by a same-named file appearing.
.PHONY: help webapp kill \
        poisson poisson-eval poisson-rating \
        montecarlo montecarlo-eval \
        train evaluate broader-eval walk-forward build-matrix odds \
        tune-weights tune-hyper tune-combined tune-joint \
        migrate scrape-squads build-schema build-ratings build-odds link-ratings

# The first target is the default, so a bare `make` prints the menu.
help:
	@echo ""
	@echo "  StatXI — run any script with:  make <name>"
	@echo ""
	@echo "  WEB"
	@echo "    make webapp            start the Flask app  -> http://127.0.0.1:5001/"
	@echo "    make kill              free port 5001 if a stale server is stuck on it"
	@echo ""
	@echo "  SCORELINE (Poisson / Dixon-Coles)"
	@echo "    make poisson           fit strengths + example predictions"
	@echo "    make poisson-eval      Poisson vs XGBoost vs bookmaker (Phase 4)"
	@echo "    make poisson-rating    rating_gap covariate ablation (Phase 5)"
	@echo ""
	@echo "  MONTE CARLO tournament simulator"
	@echo "    make montecarlo        Phase-A cross-check + seeded WC2022 simulation"
	@echo "    make montecarlo-eval   WC2022 pre-tournament backtest (Phase E)"
	@echo ""
	@echo "  WDL (XGBoost) model + evaluation"
	@echo "    make train             train the WDL classifier -> wdl_xgb.json"
	@echo "    make evaluate          evaluate the saved WDL model"
	@echo "    make broader-eval      broad odds-covered block eval + ablation"
	@echo "    make walk-forward      honest per-tournament walk-forward backtest"
	@echo "    make build-matrix      rebuild the feature matrix"
	@echo "    make odds              odds parsing / de-vig checks"
	@echo ""
	@echo "  TUNERS (all honest, leakage-safe; none promote a model)"
	@echo "    make tune-weights  |  tune-hyper  |  tune-combined  |  tune-joint"
	@echo ""
	@echo "  DATA REBUILD (slow / network / rewrites the DB — run deliberately)"
	@echo "    make migrate           CSV -> SQLite matches table"
	@echo "    make scrape-squads     scrape squads from Wikipedia (network)"
	@echo "    make build-schema      build tournaments/squads/players schema"
	@echo "    make build-ratings     build player_ratings table from the FC24 CSV"
	@echo "    make build-odds        parse odds exports -> match_odds table"
	@echo "    make link-ratings      link players -> FIFA ratings (blocking+score)"
	@echo ""

# ---- Web -------------------------------------------------------------------
webapp:
	$(PY) -m webapp.app
# Free port 5001 if a previous server didn't shut down (the ToDo notes this happens).
kill:
	@lsof -ti tcp:5001 | xargs kill -9 2>/dev/null && echo "freed port 5001" || echo "port 5001 already free"

# ---- Scoreline: Poisson / Dixon-Coles --------------------------------------
poisson:          ; $(PY) -m src.models.poisson
poisson-eval:     ; $(PY) -m src.models.poisson_eval
poisson-rating:   ; $(PY) -m src.models.poisson_rating_ablation

# ---- Monte Carlo tournament simulator --------------------------------------
montecarlo:       ; $(PY) -m src.models.montecarlo
montecarlo-eval:  ; $(PY) -m src.models.montecarlo_eval

# ---- WDL (XGBoost) model + evaluation --------------------------------------
train:            ; $(PY) -m src.models.train_wdl
evaluate:         ; $(PY) -m src.models.evaluate_wdl
broader-eval:     ; $(PY) -m src.models.broader_eval
walk-forward:     ; $(PY) -m src.models.walk_forward
build-matrix:     ; $(PY) -m src.models.build_matrix
odds:             ; $(PY) -m src.odds

# ---- Tuners ----------------------------------------------------------------
tune-weights:     ; $(PY) -m src.models.tune_weights
tune-hyper:       ; $(PY) -m src.models.tune_hyperparams
tune-combined:    ; $(PY) -m src.models.tune_combined
tune-joint:       ; $(PY) -m src.models.tune_joint

# ---- Data rebuild (root scripts, run directly) -----------------------------
migrate:          ; $(PY) migrate_to_db.py
scrape-squads:    ; $(PY) scrape_all_squads.py
build-schema:     ; $(PY) build_squad_schema.py
build-ratings:    ; $(PY) build_ratings_schema.py
build-odds:       ; $(PY) build_odds_schema.py
link-ratings:     ; $(PY) link_player_ratings.py
