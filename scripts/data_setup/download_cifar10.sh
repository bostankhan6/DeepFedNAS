#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
"${PYTHON_BIN:-python3}" "$ROOT/scripts/data_setup/prepare_cifar.py" cifar10
