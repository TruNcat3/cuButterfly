#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DESTINATION=${TURBOFFT_ROOT:-"$ROOT/external/TurboFFT"}
TURBOFFT_REPOSITORY=${TURBOFFT_REPOSITORY:-https://github.com/shixun404/TurboFFT.git}
TURBOFFT_COMMIT=${TURBOFFT_COMMIT:-918179c826f332967097fb977094ae7f54d36ab6}

if [[ ! -d "$DESTINATION/.git" ]]; then
    git clone --filter=blob:none "$TURBOFFT_REPOSITORY" "$DESTINATION"
fi

git -C "$DESTINATION" fetch --depth 1 origin "$TURBOFFT_COMMIT"
git -C "$DESTINATION" checkout --detach "$TURBOFFT_COMMIT"
printf 'TurboFFT installed at %s (%s)\n' "$DESTINATION" "$(git -C "$DESTINATION" rev-parse HEAD)"
