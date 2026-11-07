# Editable draw.io method figures

Open [cubutterfly-methods.drawio](cubutterfly-methods.drawio) in draw.io Desktop
or diagrams.net. Seven of its eight named pages are included in the manuscript;
the standalone `bank_layout` page remains as an editable source after its
example was merged into `hierarchy`.
All diagrams use native shapes, text, and connectors; they are not images
wrapped inside an editable file. Individual `.drawio` files are the canonical
sources used by the exporter. The combined document is an editing convenience.

| Page | Question answered |
| --- | --- |
| `overview` | How do common dependencies connect local arithmetic, closed regions, and hierarchical execution? |
| `organization_tradeoffs` | How do stage groups change local values, independent components, and transfers in the same graph? |
| `roofline` | Why can legal organizations attain different rates below the same resource roof? |
| `hierarchy` | How do row/column handoffs, capacity refinement, and an independent bank-layout example constrain mappings? |
| `semantic_transport` | Which structural correspondences carry across order, tensor axes, and inverse local arithmetic? |
| `bank_layout` | Retained standalone version of the four-bank XOR example now shown within `hierarchy` |
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

The 2026-11-04 redraw uses one 1000 × 310 overview and six single-column
figures. The selected-plan figure is now 480 × 292 instead of a second wide
architecture diagram. The overview owns the common input/construction/resource
story; the selected-plan figure owns preparation versus execution and a concrete
boundary binding. The two figures no longer repeat the same architecture boxes.

| Source | Canvas | Visual evidence |
| --- | --- | --- |
| overview | 1000 × 310 | N=8 dependency edges; stage/data × space/time choices; bounded workers |
| organization_tradeoffs | 480 × 414 | Full N=16 dependency network, stage cuts, local values and transfers |
| roofline | 480 × 292 | Seven frozen A100 40GB points; symbolic roof; categorical gap mechanisms |
| hierarchy | 480 × 436 | Row/column writeback; bounded child bands; bank contention versus XOR |
| semantic_transport | 480 × 274 | Directed adjacent/strided pair graphs and carried arithmetic labels |
| pipeline | 480 × 414 | Cross-band readiness plus within-band load/compute/store with two slots |
| system_architecture | 480 × 292 | Static preparation, reusable execution, concrete successor pairs |

Transfer counts, index maps, bank examples, and buffer schedules are analytical
constructions. Timeline durations are illustrative. Roofline execution points
come from `generated/roofline_cases_20261029.csv`; its roof is symbolic, not a
measured hardware ceiling. The three causal boxes have no quantitative encoding.
Neither worker lanes nor the symbolic roof establish actual hardware overlap,
occupancy, or an attainable optimum. In particular, the row/column transpose and
four-bank XOR examples are separate maps, not a composition proven conflict-free.

Use [the redesign review](../../evidence/FIGURE_REDESIGN_20261104.md) for reference
paper provenance, the purpose of each change, and remaining measurement gaps.

`tools/drawio_common.py` and `tools/drawio_*_figures.py` produced the initial
native diagrams and include checks for the analytical examples. They are
bootstrap generators, not the normal export path. Rerunning a generator replaces
its `.drawio` outputs and can overwrite later editor changes. The export command
only reads sources unless `--import-bundle` is explicitly requested.
