# Editable draw.io method figures

Open [cubutterfly-methods.drawio](cubutterfly-methods.drawio) in draw.io Desktop
or diagrams.net. Its eight named pages correspond to the manuscript figures.
All diagrams use native shapes, text, and connectors; they are not images
wrapped inside an editable file. Individual `.drawio` files are the canonical
sources used by the exporter. The combined document is an editing convenience.

| Page | Question answered |
| --- | --- |
| `overview` | How do common dependencies connect local arithmetic, closed regions, and hierarchical execution? |
| `organization_tradeoffs` | What changes when the same graph is grouped stage-wise, in bands, or as a whole transform? |
| `roofline` | Why does a higher intensity ceiling still require enough ready parallel work? |
| `hierarchy` | How do row producers feed column consumers, and how can a larger domain be refined? |
| `semantic_transport` | Which structural correspondences carry across order, tensor axes, and inverse local arithmetic? |
| `bank_layout` | How does a row-dependent XOR offset distribute a column across banks? |
| `pipeline` | When may consumers start and buffers be reused? |
| `system_architecture` | How does preparation produce a checked plan for complete runtime execution? |

## Edit and export

From the repository root, after editing an individual source:

```bash
python paper/tools/export_drawio_figures.py hierarchy
make -C paper
make -C paper evidence
```

To adopt edits from the eight-page bundle, explicitly import it. Keep its page
names unchanged. Import replaces the eight individual sources with those pages:

```bash
python paper/tools/export_drawio_figures.py \
  --import-bundle paper/figures/drawio/cubutterfly-methods.drawio
make -C paper
make -C paper evidence
```

To export all individual sources and refresh the combined document:

```bash
python paper/tools/export_drawio_figures.py
```

Export creates vector PDF, SVG, and a 2x PNG preview per figure under `exports/`.
LaTeX uses the PDF files. SVG also embeds the editable diagram. `manifest.json`
records source and output SHA-256 hashes, native cell counts, page dimensions,
and the actual renderer version. Every export refreshes the combined document
from the canonical individual sources. Import any unsaved-to-source bundle edits
before exporting individual figures.

Python 3.9+, draw.io Desktop, and Xvfb are required for headless Linux export.
The exporter uses `DRAWIO`/`XVFB`, executables on `PATH`, or the current host's
user-cache installation. Example for another installation:

```bash
DRAWIO=/path/to/drawio XVFB=/path/to/Xvfb \
  python paper/tools/export_drawio_figures.py
```

The current host uses the official
[draw.io Desktop v32.4.1](https://github.com/jgraph/drawio-desktop/releases/tag/v32.4.1)
under `~/.cache/cuButterfly/tools/drawio-32.4.1/root/opt/drawio/drawio`, with
Xvfb extracted under the same `root/usr/bin/`. The tool is separate from the
research conda environment. Export uses a private display and software rendering;
it does not run CUDA or change experiment queues. Committed/generated PDF
exports are sufficient to compile the paper without an installed editor.

## Layout and evidence

Three overview/comparison figures use a 1000-unit double-column canvas. The
five worked examples use a 480-unit single-column canvas. A common muted palette,
white background, and labels of at least 14 units keep their typography consistent
at manuscript size. Color is supplemented by panel titles, group boxes, logical
indices, and lane labels.

The transfer counts, index mappings, bank example, and buffer schedule are
analytical constructions. The roofline has symbolic axes, and the timeline has
illustrative durations. None of these diagrams is a measured performance result.
Measured experimental plots remain separate data-driven assets.

`tools/drawio_common.py` and `tools/drawio_*_figures.py` produced the initial
native diagrams and include checks for the analytical examples. They are
bootstrap generators, not the normal export path. Rerunning a generator replaces
its `.drawio` outputs and can overwrite later editor changes. The export command
only reads sources unless `--import-bundle` is explicitly requested.
