#!/usr/bin/env bash
set -euo pipefail
docker info >/dev/null 2>&1 || { echo "Docker is not running: start Docker Desktop"; exit 1; }
python -m thevillage.ping
python -m thevillage.run commons   --seed 99 --agents 3 --set n_rounds=2  --runs-dir runs_smoke --overwrite
python -m thevillage.run receipts  --seed 99 --agents 2 --set max_steps=6 --runs-dir runs_smoke --overwrite
python -m thevillage.run principal --seed 99 --agents 2 --set max_steps=6 --runs-dir runs_smoke --overwrite
echo
echo "smoke ok"
echo "run: python -m analysis.show runs_smoke/commons/n_rounds=2/seed99"
