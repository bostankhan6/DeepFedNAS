#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESUME_TRAINING=1 exec bash "$ROOT/experiments/02_deepfednas/cifar10.sh"
