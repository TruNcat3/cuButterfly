# Editable draw.io method figures

Open [cubutterfly-methods.drawio](cubutterfly-methods.drawio) in draw.io Desktop
or diagrams.net. Seven of its eight named pages are included in the manuscript;
the standalone `bank_layout` page remains an archived analytical example.
The manuscript's `hierarchy` page now shows the complete boundary I/O path.
All diagrams use native shapes, text, and connectors; they are not images
wrapped inside an editable file. Individual `.drawio` files are the canonical
sources used by the exporter. The combined document is an editing convenience.

| Page | Question answered |
| --- | --- |
| `overview` | How do five abstract construction steps compose one complete Hadamard organization? |
| `organization_tradeoffs` | How do stage groups change local values, independent components, and transfers in the same graph? |
| `roofline` | Why can legal organizations attain different rates below the same resource roof? |
| `hierarchy` | How do successor-major placement and bank remapping jointly serve the same producer/consumer boundary? |
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

The 2026-11-08 walkthrough pairs every abstract step in the full-width
`overview` with the same 16-point Hadamard example. The five steps derive
regions, derive handoffs, match resources, place values, and execute subgraphs.
The 2026-11-09 full-width `hierarchy` figure expands the fourth step into
producer outputs, mapped stores, boundary storage, consumer lanes, local
core inputs, and requested outputs, using exactly the same index roles and
physical map. The overview now contains actual butterfly edges, local-state
capacity bars, and worker storage slots; `semantic_transport` shows both
complete three-stage graphs. The separate selected-plan figure
continues to explain preparation versus reusable execution.

| Source | Canvas | Visual evidence |
| --- | --- | --- |
| overview | 1000 × 392 | Complete N=16 network, five rules, demand/capacity bars, worker-local slots, and execution |
| organization_tradeoffs | 480 × 414 | Full N=16 dependency network, stage cuts, local values and transfers |
| roofline | 480 × 292 | Seven frozen points; symbolic roof without a quantitative headroom annotation |
| hierarchy | 1000 × 414 | Producer cores → mapped stores → banks → lane reads → consumer core → requested output, with separate segment/bank views |
| semantic_transport | 480 × 328 | Complete N=8 three-stage graphs and a transported closed prefix, with carried labels |
| pipeline | 480 × 414 | Cross-band readiness plus within-band load/compute/store with two slots |
| system_architecture | 480 × 292 | Static preparation, reusable execution, concrete successor pairs |

Transfer counts, index maps, bank examples, and buffer schedules are analytical
constructions. Timeline widths illustrate order, not measured duration. Roofline
points come from `generated/roofline_cases_20261029.csv`; the roof is symbolic,
so distances to it are not measured headroom. Its categorical resource boxes
encode no quantitative attribution.

The connected example has four values of local storage **per worker** after
auxiliary reservations and a **separate** 16-word, four-bank boundary. It does
not fit all intermediate state in a four-value budget. Successor offset `c`
and bank `r XOR c` give address `4c + (r XOR c)`. Every stated row or column
access group visits all four banks, while concurrent groups may still contend.
Two workers reuse their local slots; all producers publish before any column
consumer starts. The detailed figure expands C1. Its lane r reads bank r XOR 1
at offset 1, preserving the ordered four-point core input; final values use
the requested output indices. The lower external-storage panel is an alternative
scope illustration with four-word segments, not the physical bank-map encoding.
It shows both the reduction in successor read segments and the increase in
unstaged producer store segments. Neither is an actual GPU transaction count. Larger boundaries may be materialized at an outer scope.
The example proves neither a measured rate nor the optimality of this choice.

`tools/walkthrough_example.py` checks closure, intersections, bijectivity, bank
coverage, publication order, and a complete integer Hadamard result. See
[the abstract-to-case walkthrough](../../evidence/ABSTRACT_TO_CASE_WALKTHROUGH_20261108.md)
for assumptions and review findings. The standalone `bank_layout` page retains
its older row-major example and is not used as the running-example map.
The [boundary I/O review](../../evidence/METHOD_IO_FIGURE_REVIEW_20261109.md)
records the complete input/output mapping and final manuscript layout checks.

Use [the redesign review](../../evidence/FIGURE_REDESIGN_20261104.md) for reference
paper provenance, the purpose of each change, and remaining measurement gaps.

`tools/drawio_common.py` and `tools/drawio_*_figures.py` produced the initial
native diagrams and include checks for the analytical examples. They are
bootstrap generators, not the normal export path. Rerunning a generator replaces
its `.drawio` outputs and can overwrite later editor changes. The export command
only reads sources unless `--import-bundle` is explicitly requested.
