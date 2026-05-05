#!/usr/bin/env python3
"""
Build summary.json for attractor_analysis.py.

- For chexgen_long: scans every study dir, keeps only fully-finished K=100 runs
  (metrics.json + 101 img embeds + 101 text embeds + 101 findings.txt), and
  writes summary.json with {"n_studies": N, "per_study": [<metrics.json>, ...]}.
  Each metrics.json already contains 'study_id' and 'findings' — the only
  fields attractor_analysis.py reads from per_study entries.

- For chexgen_lyapunov_long: writes a tiny sentinel summary.json. The Lyapunov
  loader only checks existence; it walks anchor_*/seed_* dirs itself.

Usage:
  python make_summary.py <chexgen_long_dir> <chexgen_lyapunov_long_dir>
"""
import json
import os
import sys
from glob import glob


def build_long_summary(long_dir, K=100):
    expected = K + 1  # iter 0 .. iter K inclusive
    sids = sorted(d for d in os.listdir(long_dir)
                  if os.path.isdir(os.path.join(long_dir, d)) and d.isdigit())
    print(f"[chexgen_long] Found {len(sids)} candidate study dirs in {long_dir}")

    finished, skipped = [], []
    for sid in sids:
        sd = os.path.join(long_dir, sid)
        n_img = len(glob(os.path.join(sd, "img_embed_iter_*.npy")))
        n_txt = len(glob(os.path.join(sd, "text_embed_iter_*.npy")))
        n_fnd = len(glob(os.path.join(sd, "findings_iter_*.txt")))
        m_path = os.path.join(sd, "metrics.json")
        ok = (n_img == expected and n_txt == expected and n_fnd == expected
              and os.path.exists(m_path))
        if not ok:
            skipped.append((sid, n_img, n_txt, n_fnd, os.path.exists(m_path)))
            continue
        try:
            with open(m_path) as f:
                finished.append(json.load(f))
        except Exception as e:
            skipped.append((sid, n_img, n_txt, n_fnd, f"json_error:{e}"))

    summary = {"n_studies": len(finished), "per_study": finished}
    out_path = os.path.join(long_dir, "summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f)
    print(f"[chexgen_long] Wrote {out_path} with {len(finished)} fully-finished studies "
          f"(skipped {len(skipped)}).")
    if skipped:
        print(f"[chexgen_long] First {min(5, len(skipped))} skipped:")
        for row in skipped[:5]:
            print(f"    sid={row[0]} img={row[1]} txt={row[2]} fnd={row[3]} metrics={row[4]}")


def build_lyap_sentinel(lyap_dir):
    if not os.path.isdir(lyap_dir):
        print(f"[lyapunov]   WARNING: dir not found: {lyap_dir}")
        return
    out_path = os.path.join(lyap_dir, "summary.json")
    with open(out_path, "w") as f:
        json.dump({"note": "sentinel — attractor_analysis.py walks anchor_*/seed_* at runtime"}, f)
    n_anchors = len(glob(os.path.join(lyap_dir, "anchor_*")))
    print(f"[lyapunov]   Wrote sentinel {out_path}  (found {n_anchors} anchor_* dirs)")


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    build_long_summary(sys.argv[1])
    build_lyap_sentinel(sys.argv[2])


if __name__ == "__main__":
    main()