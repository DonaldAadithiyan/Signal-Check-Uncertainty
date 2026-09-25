#!/usr/bin/env bash
# Rebuild all six paper figures into paper/figures/ from outputs/ results files.
set -euo pipefail
cd "$(dirname "$0")"
for i in 1 2 3 4 5 6; do python3.11 "make_fig$i.py"; done
