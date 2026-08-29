"""Stage-2 CASA awp2 visibility oracle. Do not run until Stage 1 is frozen.

This script is the declared Bacchus recipe, not an implemented generator.
It refuses so a default VLA prediction cannot be frozen by accident.
"""

from __future__ import annotations

raise RuntimeError(
    "the full-polarisation CASA awp2 oracle is not implemented; "
    "freeze the Stage-1 power-beam comparison first, then predict "
    "I / I+Q / I+U / I+V point sources with gridder='awp2' and "
    "savemodel='modelcolumn'. Do not invert CASA visibilities back "
    "into antenna Jones."
)
