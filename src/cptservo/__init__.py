"""CPTServo: learned closed-loop servo for a chip-scale CPT-Rb87 atomic clock.

A digital twin of a Knappe-lab-lineage chip-scale CPT-Rb87 atomic clock plus a
learned servo trained via differentiable-physics analytic policy gradient (APG)
through the twin, benchmarked against PI and receding-horizon LQR baselines,
deployed on a Jetson Orin Nano.
"""

from __future__ import annotations

__version__ = "0.1.0"
