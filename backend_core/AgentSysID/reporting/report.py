"""
PDF report generation and result packaging.

Currently uses fpdf2 (ported from the original pdf_generator.py).
When packages/labcd_pdfmaker is available in the monorepo it should be
preferred – see the migration note in GUIDE.md.
"""

from __future__ import annotations

import datetime
import os
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fpdf import FPDF


def _clean_text(text: str) -> str:
    if not isinstance(text, str):
        return str(text)
    replacements = {
        "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-",
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u2022": "-", "\u00a0": " ",
    }
    for char, rep in replacements.items():
        text = text.replace(char, rep)
    return text.encode("latin-1", "ignore").decode("latin-1")


class SystemIDReport(FPDF):
    def header(self):
        self.set_font("Times", "B", 14)
        self.cell(0, 8, _clean_text("System Identification and Dynamic Stability Report"), ln=True)
        self.set_font("Times", "B", 12)
        self.cell(0, 6, _clean_text("Automated Core Engineering Analysis Engine"), ln=True)
        self.set_font("Times", "I", 11)
        self.cell(0, 6, _clean_text("LabCD.ai – AgentSysID"), ln=True)
        self.set_font("Times", "", 10)
        date_str = f"Evaluation Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        self.cell(0, 6, _clean_text(date_str), ln=True)
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Times", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def generate_final_pdf(
    env_name: str,
    abstract: str,
    best_config: Dict[str, Any],
    best_mse: float,
    best_rmse: float,
    latency_ms: float,
    state_dim: int,
    action_dim: int,
    use_pinn: bool = False,
    output_dir: str | Path = ".",
    filename: Optional[str] = None,
) -> str:
    """Generate a concise PDF report and return the file path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if filename is None:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"SysID_Report_{env_name}_{stamp}.pdf"
    path = output_dir / filename

    pdf = SystemIDReport()
    pdf.add_page()
    pdf.set_font("Times", "B", 12)
    pdf.cell(0, 8, _clean_text("1. Executive Abstract"), ln=True)
    pdf.set_font("Times", "", 11)
    pdf.multi_cell(0, 6, _clean_text(abstract))
    pdf.ln(4)

    pdf.set_font("Times", "B", 12)
    pdf.cell(0, 8, _clean_text("2. Identification Summary"), ln=True)
    pdf.set_font("Times", "", 11)
    lines = [
        f"Dataset / Environment : {env_name}",
        f"State dimension       : {state_dim}",
        f"Action dimension      : {action_dim}",
        f"Best Validation MSE   : {best_mse:.6e}",
        f"Best Validation RMSE  : {best_rmse:.6e}",
        f"Inference latency     : {latency_ms:.3f} ms",
        f"Physics-informed (PINN): {'Yes' if use_pinn else 'No'}",
        f"Final architecture    : {best_config}",
    ]
    for line in lines:
        pdf.cell(0, 6, _clean_text(line), ln=True)

    pdf.output(str(path))
    print(f"   📄 PDF report written → {path}")
    return str(path)


def package_final_results_to_zip(
    pdf_filename: str,
    extra_files: Optional[List[str]] = None,
    output_dir: str | Path = ".",
    env_name: str = "system",
) -> str:
    """Bundle PDF + optional artefacts into a ZIP and return the zip path."""
    output_dir = Path(output_dir)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = output_dir / f"SysID_Delivery_{env_name}_{stamp}.zip"

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        if os.path.isfile(pdf_filename):
            zf.write(pdf_filename, arcname=os.path.basename(pdf_filename))
        for f in extra_files or []:
            if os.path.isfile(f):
                zf.write(f, arcname=os.path.basename(f))

    print(f"   📦 Delivery ZIP written → {zip_path}")
    return str(zip_path)
