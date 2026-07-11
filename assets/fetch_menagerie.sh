#!/usr/bin/env bash
# Fetches the UR5e + Robotiq 2F-85 MJCF models from mujoco_menagerie and stages
# their mesh files where the generated scene XML expects them (meshdir="assets/"
# relative to assets/ur5e_chess_scene.xml, i.e. assets/assets/).
#
# Both assets/menagerie/ and assets/assets/ are gitignored (vendored, regenerable)
# -- run this once after cloning, then `python assets/build_scene.py`.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -d "$HERE/menagerie" ]; then
    git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git "$HERE/menagerie"
fi

mkdir -p "$HERE/assets"
cp "$HERE"/menagerie/universal_robots_ur5e/assets/* "$HERE/assets/"
cp "$HERE"/menagerie/robotiq_2f85/assets/* "$HERE/assets/"

echo "menagerie models + meshes staged under $HERE/menagerie and $HERE/assets"
