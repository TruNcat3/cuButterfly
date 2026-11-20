"""Physical-size plotting defaults for the two-column manuscript.

Render at the intended column width: LaTeX scaling must not enlarge the text.
This module controls presentation only; it does not transform measurements.
"""

SINGLE_COLUMN = 3.45
DOUBLE_COLUMN = 7.05
RC = {
    'font.family': 'serif',
    'font.serif': ['Liberation Serif', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 7.0,
    'axes.labelsize': 7.0,
    'axes.titlesize': 7.2,
    'axes.titleweight': 'normal',
    'axes.titlepad': 4.0,
    'xtick.labelsize': 6.5,
    'ytick.labelsize': 6.5,
    'legend.fontsize': 6.5,
    'axes.linewidth': 0.55,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 2.0,
    'ytick.major.size': 2.0,
    'xtick.major.pad': 2.0,
    'ytick.major.pad': 2.0,
    'lines.linewidth': 0.85,
    'lines.markersize': 3.0,
    'grid.linewidth': 0.4,
    'grid.color': '#DDDDDD',
    'legend.frameon': False,
    'legend.handlelength': 1.3,
    'legend.handletextpad': 0.35,
    'legend.columnspacing': 0.9,
    'legend.borderpad': 0.1,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
    'savefig.facecolor': 'white',
    'axes.axisbelow': True,
}

BLUE = '#0072B2'
ORANGE = '#D55E00'
GREEN = '#007B59'
PURPLE = '#7B3294'
