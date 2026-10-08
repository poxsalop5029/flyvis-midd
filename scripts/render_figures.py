#!/usr/bin/env python3
"""Compatibility CLI; canonical entry points are the Figure notebooks.

The single maintained plotting implementation is flyvis_midd.figure_rendering.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flyvis_midd.figure_rendering import render, add_s1_calibration

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    for key in sys.argv[1:] or ["1", "2", "3", "4", "5", "6", "S3", "S5", "S6", "S7", "S8", "S9", "S10"]:
        render(key)
        plt.close("all")
