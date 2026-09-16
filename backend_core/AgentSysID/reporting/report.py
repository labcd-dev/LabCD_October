"""
PDF report generation and structured result packaging.

ZIP layout matches the original backend delivery:

    SystemID_RunResults_<timestamp>.zip
    ├── README.md
    ├── Agents_log/
    ├── figures/
    ├── deployment/
    └── report/
"""

from __future__ import annotations

import datetime
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

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
    """Generate PDF under output_dir/report/ and return the file path."""
    output_dir = Path(output_dir)
    report_dir = output_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    if filename is None:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"System_Report_{env_name}_{stamp}.pdf"
    path = report_dir / filename

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
    run_dir: str | Path,
    env_name: str,
    timestamp: str,
    pdf_filename: Optional[str] = None,
) -> str:
    """
    Package run_dir into SystemID_RunResults_<timestamp>.zip:

        README.md, figures/, deployment/, report/, Agents_log/
    """
    run_dir = Path(run_dir)
    zip_filename = run_dir / f"SystemID_RunResults_{timestamp}.zip"

    print("\n" + "=" * 80)
    print("📦 PACKAGING FINAL PIPELINE RESULTS")
    print("=" * 80)

    readme_content = f"""# System Identification & Neural Controller Package
**Run Timestamp:** {timestamp}
**Framework:** AgentSysID (LabCD multi-agent system identification)

---

## Directory Structure

### 1. `figures/`
Diagnostic plots (MSE/RMSE convergence, latency, hyperparameters, architecture contour).

### 2. `deployment/`
* **Model weights (`*.pth`)**
* **`deployed_controller_{env_name}.py`** – standalone NeuralController
* **`NN.py`** – inference / integration snippet

### 3. `report/`
PDF engineering report.

### 4. `Agents_log/`
LLM conversation history (prompts, responses, token usage).
"""
    readme_path = run_dir / "README.md"
    readme_path.write_text(readme_content, encoding="utf-8")

    with zipfile.ZipFile(zip_filename, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(readme_path, arcname="README.md")
        print("  ├── Added root README.md")

        fig_dir = run_dir / "figures"
        n_fig = 0
        if fig_dir.is_dir():
            for img in sorted(fig_dir.glob("*.png")):
                zipf.write(img, arcname=f"figures/{img.name}")
                n_fig += 1
        print(f"  ├── [figures/] {n_fig} plot(s)")

        dep_dir = run_dir / "deployment"
        n_dep = 0
        if dep_dir.is_dir():
            for f in sorted(dep_dir.iterdir()):
                if f.is_file():
                    zipf.write(f, arcname=f"deployment/{f.name}")
                    n_dep += 1
        for m in sorted(run_dir.glob("*.pth")):
            zipf.write(m, arcname=f"deployment/{m.name}")
            n_dep += 1
        print(f"  ├── [deployment/] {n_dep} file(s)")

        report_dir = run_dir / "report"
        n_rep = 0
        if report_dir.is_dir():
            for f in sorted(report_dir.glob("*.pdf")):
                zipf.write(f, arcname=f"report/{f.name}")
                n_rep += 1
        if pdf_filename and Path(pdf_filename).is_file():
            p = Path(pdf_filename)
            if p.parent.resolve() != report_dir.resolve():
                zipf.write(p, arcname=f"report/{p.name}")
                n_rep += 1
        print(f"  ├── [report/] {n_rep} PDF(s)")

        n_log = 0
        for pattern in ("*.txt", "*.log"):
            for logf in sorted(run_dir.glob(pattern)):
                if logf.name.lower().startswith("readme"):
                    continue
                zipf.write(logf, arcname=f"Agents_log/{logf.name}")
                n_log += 1
        agents_dir = run_dir / "Agents_log"
        if agents_dir.is_dir():
            for logf in sorted(agents_dir.iterdir()):
                if logf.is_file():
                    zipf.write(logf, arcname=f"Agents_log/{logf.name}")
                    n_log += 1
        print(f"  ├── [Agents_log/] {n_log} log file(s)")

    print("-" * 80)
    print(f"  ✅ ZIP: {zip_filename}")
    print(f"  📂 {zip_filename.resolve()}")
    print("=" * 80 + "\n")
    return str(zip_filename)
