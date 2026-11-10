#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DESTINATION=${VKFFT_ROOT:-"$ROOT/external/VkFFT"}
VKFFT_REPOSITORY=${VKFFT_REPOSITORY:-https://github.com/DTolm/VkFFT.git}
VKFFT_COMMIT=${VKFFT_COMMIT:-066a17c17068c0f11c9298d848c2976c71fad1c1}

if [[ ! -d "$DESTINATION/.git" ]]; then
    git clone --filter=blob:none "$VKFFT_REPOSITORY" "$DESTINATION"
fi

git -C "$DESTINATION" fetch --depth 1 origin "$VKFFT_COMMIT"
git -C "$DESTINATION" checkout --detach "$VKFFT_COMMIT"
printf 'VkFFT installed at %s (%s)\n' "$DESTINATION" "$(git -C "$DESTINATION" rev-parse HEAD)"
