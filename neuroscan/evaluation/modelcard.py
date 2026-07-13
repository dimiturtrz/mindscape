"""Auto model card — a per-run markdown summary (the siblings' modelcard pattern).

Reads a harness aggregate dict and writes a rigorous one-page card: the headline numbers, the number
against the published ceiling, the per-subject spread, and explicitly **where it fails** (worst subject).
"""
from __future__ import annotations

from pathlib import Path

from core.reference import Reference


class ModelCard:
    @staticmethod
    def card(res: dict, dataset: str, regime: str) -> str:
        fm, pooled, sp = res["fold_mean"], res["pooled"], res["acc_spread"]
        ref_regime = "within_subject" if regime == "within" else regime
        per = sorted(res["per_fold"], key=lambda r: r["acc"])
        worst, best = per[0], per[-1]
        lines = [
            f"# Model card — {res['method']} · {regime} · {dataset}",
            "",
            f"- **fold-mean accuracy**: {fm['acc']:.3f}  ·  kappa {fm['kappa']:.3f}  ·  ECE {fm['ece']:.3f}",
            f"- **pooled** (per-epoch): acc {pooled['acc']:.3f}  ·  kappa {pooled['kappa']:.3f}  ·  "
            f"ECE {pooled['ece']:.3f}",
            f"- **per-subject spread**: {sp['min']:.3f} – {sp['max']:.3f}  (std {sp['std']:.3f})",
            f"- **vs reference**: {Reference.compare(fm['acc'], dataset, ref_regime, res['method'])}",
            "",
            "## Where it fails",
            f"- worst: subject {worst['fold']} — acc {worst['acc']:.3f} "
            f"(kappa {worst['kappa']:.3f}, ECE {worst['ece']:.3f})",
            f"- best:  subject {best['fold']} — acc {best['acc']:.3f}",
            "",
            "## Per-subject",
            "| subject | acc | kappa | ece | n |",
            "|---|---|---|---|---|",
        ]
        lines += [f"| {r['fold']} | {r['acc']:.3f} | {r['kappa']:.3f} | {r['ece']:.3f} | {r['n']} |"
                  for r in res["per_fold"]]
        lines += ["", "_Honest eval: the model is commodity; the contribution is the measured spread + "
                  "calibration, not the headline number._", ""]
        return "\n".join(lines)

    @staticmethod
    def write(res: dict, dataset: str, regime: str, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(ModelCard.card(res, dataset, regime))
        return path
