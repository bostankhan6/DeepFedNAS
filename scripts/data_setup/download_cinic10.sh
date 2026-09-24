#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="$ROOT/data/cinic10"
mkdir -p "$DEST"
wget -c https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz -O "$DEST/CINIC-10.tar.gz"
tar -xzf "$DEST/CINIC-10.tar.gz" -C "$DEST"
if [[ -d "$DEST/CINIC-10/train" ]]; then
    for split in train valid val test; do
        if [[ -d "$DEST/CINIC-10/$split" ]]; then
            target="$split"
            [[ "$split" == valid ]] && target=val
            mv "$DEST/CINIC-10/$split" "$DEST/$target"
        fi
    done
fi
[[ -d "$DEST/train" && -d "$DEST/val" && -d "$DEST/test" ]] || { echo "Expected train/val/test folders under $DEST" >&2; exit 1; }
