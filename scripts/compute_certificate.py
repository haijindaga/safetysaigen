"""Compute the probabilistic safe-traversal certificate from measured values.

Feed in the measured perception latency (VLM + segmentation + grounding)
and the calibrated unsafe-detection rate; get back kappa* and the maximum
safe traversal speed for a target risk delta.

Usage:
    python scripts/compute_certificate.py --latency 5.0 --detection 0.75
    python scripts/compute_certificate.py --latency 5.0 --detection 0.75 \
        --delta 0.1 --sensing-radius 4.0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core_safety.theory.certificate import default_m, find_certificate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latency", type=float, required=True,
                    help="perception cycle time [s] (VLM + seg + grounding)")
    ap.add_argument("--detection", type=float, default=0.75,
                    help="high-probability lower bound of the unsafe "
                         "detection rate (paper: 0.75)")
    ap.add_argument("--delta", type=float, default=0.1,
                    help="target failure probability")
    ap.add_argument("--gamma", type=float, default=0.0,
                    help="risk fraction for worst-case contexts")
    ap.add_argument("--sensing-radius", type=float, default=4.0,
                    help="max distance D at which unsafe regions are "
                         "detectable [m]")
    args = ap.parse_args()

    m = lambda r: default_m(r, D=args.sensing_radius, p=args.detection)
    cert = find_certificate(delta=args.delta, gamma=args.gamma,
                            D=args.sensing_radius,
                            t_perception=args.latency, m=m)
    if cert is None:
        print("No certificate found — detection rate too low for this delta.")
        return
    print(f"kappa*          = {cert.kappa} observations per {args.sensing_radius} m")
    print(f"(c, l)          = ({cert.c}, {cert.ell:.4f})  residual {cert.residual:.4f}")
    print(f"max safe speed  = {cert.max_speed:.3f} m/s "
          f"(guarantee: P[unsafe entry] <= {args.delta})")
    print(f"-> set v_max <= {cert.max_speed:.3f} in CoreConfig / the demo scripts")


if __name__ == "__main__":
    main()
