#!/bin/bash
set -e

# 1. Robustly find the project root (2 levels up from scripts/data_setup/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# 2. Define the target directory (root/data/cinic10)
TARGET_DIR="$PROJECT_ROOT/data/cinic10"
mkdir -p "$TARGET_DIR"

echo "Data will be downloaded to: $TARGET_DIR"
cd "$TARGET_DIR"

# 3. Download from Google Drive
#    (This complicated wget command handles the large file confirmation from Google)
echo "Downloading CINIC-10.tar.gz..."

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1WimEOXYdCdtry4cZQrJl3DzrAKcEJuyA' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1WimEOXYdCdtry4cZQrJl3DzrAKcEJuyA" -O "CINIC-10.tar.gz" 

# 4. Cleanup cookies
rm -rf /tmp/cookies.txt

# 5. Extract
echo "Extracting CINIC-10..."
tar -xvzf CINIC-10.tar.gz

# Optional: cleanup the tar file to save space
# rm CINIC-10.tar.gz

echo "Done. CINIC-10 is ready at $TARGET_DIR"