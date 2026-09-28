"""Scan once, save evidence, then render with the shared staff-decision rules.

Usage: python src/visualize_staff.py [video] [reference] [output.mp4]
All identify_staff.py scan options, including --reid-tracker, are supported.
"""
from pathlib import Path
import math

from identify_staff import make_parser, parse_args as parse_scan_args, run
from render_evidence_video import parse_args as parse_render_args, render

ROOT = Path(__file__).resolve().parent.parent


def parse_args(argv=None):
    parser = make_parser()
    parser.description = __doc__
    parser.add_argument("output_path", nargs="?")
    parser.add_argument("--display-seconds", type=float, default=0.6)
    parser.add_argument("--min-display-score", type=float, default=0.65)
    parser.set_defaults(batch_size=4)
    args = parse_scan_args(argv, parser)
    if not math.isfinite(args.display_seconds) or args.display_seconds < 0:
        parser.error("--display-seconds must be finite and non-negative")
    if not 0 <= args.min_display_score <= 1:
        parser.error("--min-display-score must be in [0, 1]")
    return args


def main():
    args = parse_args()
    report = run(args)
    output = args.output_path or str(ROOT / "output" / "runs" / f"{Path(report['output_dir']).name}.mp4")
    render_args = parse_render_args([report["output_dir"], args.video_path, output,
                                    "--display-seconds", str(args.display_seconds),
                                    "--min-display-score", str(args.min_display_score)])
    render(render_args)


if __name__ == "__main__":
    main()
