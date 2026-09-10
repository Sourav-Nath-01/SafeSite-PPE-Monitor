#!/usr/bin/env bash
# Run the compliance-logic test suite. No model, GPU or camera required --
# the association and state-machine layers are deliberately dependency-free.
set -euo pipefail

cd "$(dirname "$0")/Model-Deployment"
python -m unittest discover -s tests -p 'test_*.py' -v
