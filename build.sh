#!/usr/bin/env bash
set -euo pipefail
cd simulation
cargo build --release
cd ..
cp --reflink=auto \
    simulation/target/release/liballiance_evolution_simulation.so \
    python/alliance_evolution_simulation.so
strip python/alliance_evolution_simulation.so --strip-unneeded
chmod -x python/alliance_evolution_simulation.so
