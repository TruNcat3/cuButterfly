#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
SOURCE_TEX="${SOURCE_TEX:-main.tex}"
case "$SOURCE_TEX" in
  main.tex|main-evidence.tex) ;;
  *) printf 'Unsupported manuscript source: %s\n' "$SOURCE_TEX" >&2; exit 1 ;;
esac
OUTPUT_NAME="${SOURCE_TEX%.tex}"
BUILD_DIR="${BUILD_DIR:-$SCRIPT_DIR/build}"
case "$BUILD_DIR" in
  /*) ;;
  *) BUILD_DIR="$SCRIPT_DIR/$BUILD_DIR" ;;
esac

resolve_compiler() {
  if [[ -n "${TECTONIC:-}" ]]; then
    if [[ -x "$TECTONIC" ]]; then
      printf '%s\n' "$TECTONIC"
      return
    fi
    if command -v "$TECTONIC" >/dev/null 2>&1; then
      command -v "$TECTONIC"
      return
    fi
    printf 'TECTONIC does not point to an executable: %s\n' "$TECTONIC" >&2
    return 1
  fi
  local cache_base candidate
  cache_base="${XDG_CACHE_HOME:-${HOME:-$SCRIPT_DIR}/.cache}"
  for candidate in \
    "$SCRIPT_DIR/.tools/tectonic" \
    "$cache_base/cuButterfly/tools/tectonic-0.17.0/tectonic" \
    "$cache_base/cuButterfly/tools/tectonic/tectonic"; do
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return
    fi
  done
  if command -v tectonic >/dev/null 2>&1; then
    command -v tectonic
    return
  fi
  printf '%s\n' 'Tectonic was not found. See paper/BUILD.md for a user-level install.' >&2
  return 1
}

TECTONIC_BIN="$(resolve_compiler)"
mkdir -p "$BUILD_DIR"
cd "$SCRIPT_DIR"
"$TECTONIC_BIN" --chatter minimal --keep-logs --keep-intermediates \
  --outdir "$BUILD_DIR" "$SOURCE_TEX"
# Reject an incomplete bibliography before updating the convenient reader copy.
if [[ ! -s "$BUILD_DIR/$OUTPUT_NAME.bbl" ]] || \
   ! grep -q '\\bibitem' "$BUILD_DIR/$OUTPUT_NAME.bbl"; then
  printf 'Bibliography is empty; inspect %s/%s.blg.\n' "$BUILD_DIR" "$OUTPUT_NAME" >&2
  exit 1
fi
if grep -Eq 'LaTeX Warning: (Citation .* undefined|Reference .* undefined|There were undefined references)' "$BUILD_DIR/$OUTPUT_NAME.log"; then
  printf 'Unresolved citations or cross-references; inspect %s/%s.log.\n' "$BUILD_DIR" "$OUTPUT_NAME" >&2
  exit 1
fi
if [[ "$BUILD_DIR/$OUTPUT_NAME.pdf" != "$SCRIPT_DIR/$OUTPUT_NAME.pdf" ]]; then
  # An editor may compile directly to main.pdf while this build runs. Publish
  # a complete copy by rename rather than interleave two writers on one inode.
  reader_copy="$(mktemp "$SCRIPT_DIR/.$OUTPUT_NAME.pdf.XXXXXX")"
  cp "$BUILD_DIR/$OUTPUT_NAME.pdf" "$reader_copy"
  mv -f "$reader_copy" "$SCRIPT_DIR/$OUTPUT_NAME.pdf"
fi
printf 'Wrote %s/%s.pdf\n' "$BUILD_DIR" "$OUTPUT_NAME"
