#!/usr/bin/env bash
# Downloads the official Stockfish Linux binary and stages it at engine/bin/stockfish.
# engine/bin/ is gitignored (113MB binary) -- run this once after cloning.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VARIANT="${1:-bmi2}"  # bmi2 covers most x86-64 CPUs since ~2013; pass e.g. "avx2" to override
TAG="sf_18"
ASSET="stockfish-ubuntu-x86-64-${VARIANT}.tar"
URL="https://github.com/official-stockfish/Stockfish/releases/download/${TAG}/${ASSET}"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

curl -sL -o "$TMP/sf.tar" "$URL"
tar -xf "$TMP/sf.tar" -C "$TMP" "stockfish/stockfish-ubuntu-x86-64-${VARIANT}"

mkdir -p "$HERE/bin"
cp "$TMP/stockfish/stockfish-ubuntu-x86-64-${VARIANT}" "$HERE/bin/stockfish"
chmod +x "$HERE/bin/stockfish"

echo "stockfish staged at $HERE/bin/stockfish"
echo "quit" | "$HERE/bin/stockfish" | head -1
