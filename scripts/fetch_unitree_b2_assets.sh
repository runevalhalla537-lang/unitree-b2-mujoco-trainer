#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ASSET_DIR="$ROOT/assets/b2"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

git clone --depth 1 https://github.com/unitreerobotics/unitree_mujoco.git "$TMP/unitree_mujoco"

mkdir -p "$ASSET_DIR"
cp -a "$TMP/unitree_mujoco/unitree_robots/b2/." "$ASSET_DIR/"

echo "Copied Unitree B2 MuJoCo assets to: $ASSET_DIR"
echo "Main XML: $ASSET_DIR/b2.xml"
