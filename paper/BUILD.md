# Build the paper

From the repository root, run `make -C paper`. The output is
`paper/build/main.pdf`, with an identical reader copy at `paper/main.pdf`.
`paper/build.sh` also works directly; `BUILD_DIR` selects
an alternative output directory, and `TECTONIC` selects a compiler executable.
The Make target tracks all section, generated-table, bibliography, and PDF
figure inputs. It also refreshes the reader copy when the canonical PDF is
already current.
It does not invoke CUDA, CMake, hardware detection, or benchmarks.

The current manuscript includes analytical schematics and measured figures;
neither build currently includes the retained synthetic layout examples.
Build the evidence-wrapper entry with `make -C paper evidence`. Its canonical output is
`paper/build/main-evidence.pdf`, with reader copy `paper/main-evidence.pdf`.
The `main-evidence.tex` wrapper defines `\cBNoDraftFigures`; `main.tex`
disables any draft-only floats guarded by that switch.
Both builds retain analytical schematics, measured figures, measured tables,
and the same bibliography checks. This is an evidence-selection switch, not
a declaration of submission readiness.

`SOURCE_TEX=main-evidence.tex paper/build.sh` is the equivalent direct build.
The build accepts only the two declared source entries and never overwrites
the main PDF with the evidence-wrapper PDF. Historical synthetic source rows, metadata,
and replacement requirements are in `generated/draft/` and
`figures/DRAFT_LAYOUTS.md`. Regenerate them explicitly with
`python paper/tools/render_draft_evaluation.py --overwrite`; do not relabel
their values as observations. The renderer runs on the CPU.
The reader copy is published by atomic rename so its bytes cannot be
interleaved with an editor's concurrent PDF writer. An editor recipe may
subsequently replace that copy with its own build. Open `paper/build/main.pdf`
to view the validated Tectonic output in that situation.

The tested compiler is the statically linked Tectonic 0.17.0 Linux x86_64 musl
release. It is installed in the user cache, without sudo or changes to the
shared cuButterfly conda environment. Lookup order is an explicit `TECTONIC`,
`paper/.tools/tectonic`, the versioned user-cache location below, the unversioned
user-cache location, then `PATH`.

On another Linux x86_64 host:

```bash
tool_dir="${XDG_CACHE_HOME:-$HOME/.cache}/cuButterfly/tools/tectonic-0.17.0"
download_dir="$(mktemp -d)"
mkdir -p "$tool_dir"
curl -fL --retry 2 \
  'https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.17.0/tectonic-0.17.0-x86_64-unknown-linux-musl.tar.gz' \
  -o "$download_dir/tectonic.tar.gz"
printf '%s  %s\n' \
  '8533d07f9ccbd7a65824b9e0459041bca34af1eb33daba48f59215593753a3b7' \
  "$download_dir/tectonic.tar.gz" | sha256sum -c -
tar -xzf "$download_dir/tectonic.tar.gz" -C "$tool_dir"
make -C paper
```

The first build downloads TeX support files into Tectonic's cache. Later builds
reuse them. The validated bundle content identifier is
`6ffe055852f8faf66c0acbe1a7fb27f87b869a90bad1204f3bf4d9683f597c7c`;
an offline build requires the compiler and this populated cache. Paper sources,
including generated tables and bibliography, are self-contained below `paper/`.

`IEEEtran.cls` is copied byte-for-byte from the provided ISCA2026 template:
SHA256 `da751920a317ed318b7b5cd7fa585a6cc7d28502d457856382e9be24b10a3bd7`.
The class geometry and font sizes are not overridden. `T1` font encoding keeps
the class's Times families available under Tectonic, avoiding its default TU
font substitutions. `references.bib` and the vendored `IEEEtran.bst` are
required. The unmodified style is IEEEtran 1.14, copied from the local TeX Live
distribution with its license header intact (SHA256
`314f0ece704568faf827011bac498650691b2b5ee06320720830e782416d5a5f`).
Keeping it beside `main.tex` also supports conventional BibTeX installations
that lack the system IEEE style package.

The reference question marks observed in the older `paper/main.pdf` came
from an empty `main.bbl`: its BibTeX log reported `I couldn't open style file
IEEEtran.bst`. A single LaTeX pass cannot complete references. Use the build
entry above, which runs bibliography/reference resolution and refuses to
publish a reader copy with an empty bibliography or unresolved references.
The bibliography remains ordinary `\cite{key}` plus `references.bib`; there
are no hand-written citation numbers.

To regenerate the evidence table or check the theoretical examples:

```bash
python paper/tools/export_evidence.py
python paper/tools/check_theory.py --output paper/generated/theory_checks.json
make -C paper
```

These commands use only the frozen paper inputs and CPU calculations. They do
not extend the GPU experiment matrix. The `build/` directory, TeX intermediates,
and tool binaries are ignored by Git; the generated evidence table and mathematical-check summary
are retained as part of the working draft.

## Figures and the current result snapshot

The vector figures are retained under `figures/` and do not require Python
or draw.io to compile the paper. The eight method diagrams use native draw.io
sources. After editing them, export with the desktop draw.io renderer and
rebuild. The exporter preserves edits made in the editor:

```bash
python paper/tools/export_drawio_figures.py
make -C paper
```

The export requires Python 3.9+, draw.io Desktop, and Xvfb on headless Linux.
Set `DRAWIO` and `XVFB` to their executable paths if they are not on `PATH`.
On the current host the exporter also recognizes the prepared user-cache
installation, with no system installation or conda changes.
`make -C paper drawio-figures PYTHON=/path/to/python` is the equivalent export
target; it is deliberately separate from the default PDF build. The editable
eight-page bundle, single-figure updates, and bundle import are documented in
[`figures/drawio/README.md`](figures/drawio/README.md).

For the current manuscript's operator, historical fixed-core, image-filter,
and Poisson graphs, redraw their accepted evidence populations with:

```bash
/home/wt/yes/envs/cubutterfly/bin/python paper/tools/render_principle_evaluation.py
/home/wt/yes/envs/cubutterfly/bin/python paper/tools/render_factor_organization.py
make -C paper
```

The second command reads the independent frozen October 25 nine-case FP64
organization diagnostic in `evidence/factor_organization_20261025.json`.
Its CSVs and hash metadata are separate from `generated/principle_*` and do
not change the 36-request screen or independent 12-request NTT statistics.
Both renderers run on the CPU; accepting later measurements requires an
explicit evidence update and review of the prose and aggregates.

The older `render_experiment_figures.py` maintains `current_snapshot.json`
and historical `results_*` assets. Its `--refresh-snapshot` option explicitly
accepts later input for that older population; it is not an update switch
for the current manuscript figures. `export_evidence.py` maintains the
historical table. Missing results remain pending, never synthetic zero
measurements. Figure lineage and scope are documented in `figures/README.md`.

## FFT-L precision columns in the unified comparison

The historical November 4 low-precision evidence has a separate frozen export and does not
alter main66 or cyclic-NTT denominators. Its public-winner replay is loaded into
the ordinary operator figure/table. The November 6 nine-request FP16 cohort
subsequently replaced that replay. Further BF16 experiments are deferred by the
user's FP16-only scope decision; cuFFT Xt supports BF16 storage. Fixed controls are retained separately and never used as AUTO results.

```bash
/home/wt/yes/envs/cubutterfly/bin/python paper/tools/load_fft_l_comparison.py
/home/wt/yes/envs/cubutterfly/bin/python paper/tools/render_current_refresh.py
make -C paper all evidence
```

These commands use frozen paper assets and need no live result directory or GPU.
`generated/fft_l_20261104/generate.py` can separately re-audit the original local
raw records; changes to frozen evidence require intentional loader/source review.
The old `render_fft_l_boundary.py` figure is archived and excluded from the
manuscript. Profile evidence now supports the common generated-kernel analysis.

The [November 7 closeout](evidence/A100_CLOSEOUT_20261107.md) records the frozen
A100 cohorts and the separate eleven-request 40GB installed-default supplement.
The latter is excluded from all 80GB figure/table aggregates. PRO6000 Server
96GB installation and matched comparison remain pending.
