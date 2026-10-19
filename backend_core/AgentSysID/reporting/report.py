"""
PDF report generation and structured result packaging.

The document follows the original nine-section engineering manuscript:

    Abstract
    1. System Structure & Parameterization
    2. Optimal Neural Architecture & Performance
    3. System Identification Convergence (MSE)
    4. Absolute Error Magnitude (RMSE)
    5. Architecture Topology Space
    6. Hyperparameter Mutation Trajectory
    7. Inference Timing Comparison
    8. Data-Driven Held-Out Verification
    9. Concluding Remarks & Deployment Synthesis

The client ZIP contains the concise delivery package::

    SystemID_RunResults_<timestamp>.zip
    ├── README.md
    ├── deployment/
    └── report/
"""

from __future__ import annotations

import datetime
import glob
import os
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

from fpdf import FPDF
from fpdf.enums import XPos, YPos


def _clean_text(text: Any) -> str:
    """
    Bulletproof text sanitizer.

    Replaces typographical characters the PDF latin-1 core fonts cannot encode
    and strips anything else that would crash fpdf.
    """
    if not isinstance(text, str):
        return str(text)

    replacements = {
        "\u2011": "-",  # non-breaking hyphen
        "\u2012": "-",  # figure dash
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
        "\u2018": "'",  # left single quote
        "\u2019": "'",  # right single quote
        "\u201c": '"',  # left double quote
        "\u201d": '"',  # right double quote
        "\u2022": "-",  # bullet
        "\u00a0": " ",  # non-breaking space
    }
    for char, rep in replacements.items():
        text = text.replace(char, rep)

    return text.encode("latin-1", "ignore").decode("latin-1")


# Public alias - the legacy module exported this name.
clean_text = _clean_text


class SystemIDReport(FPDF):
    """Report template: running header, footer, section titles, table rows."""

    def header(self) -> None:
        self.set_font("Times", "B", 14)
        self.cell(
            0, 8,
            _clean_text("System Identification Results and Validation Report"),
            border=False, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="L",
        )

        self.set_font("Times", "B", 12)
        self.cell(
            0, 6,
            _clean_text("Automated Core Engineering Analysis Engine"),
            border=False, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="L",
        )

        self.set_font("Times", "I", 11)
        self.cell(0, 6, _clean_text("LabCD.ai"), border=False, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="L")

        self.set_font("Times", "", 10)
        date_str = f"Evaluation Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        self.cell(0, 6, _clean_text(date_str), border=False, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="L")

        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)

    def footer(self) -> None:
        self.set_y(-15)
        self.set_font("Times", "I", 8)
        self.cell(
            0, 10,
            _clean_text(
                f"Confidential - Automated Computational Engineering Core | Page {self.page_no()}"
            ),
            align="C",
        )

    def add_section_title(self, title: str) -> None:
        self.ln(5)
        self.set_font("Times", "B", 12)
        self.set_fill_color(220, 220, 220)
        self.cell(0, 8, _clean_text(title), border=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="L", fill=True)
        self.ln(3)

    def add_abstract(self, abstract_text: str) -> None:
        self.set_font("Times", "B", 11)
        self.cell(20, 6, _clean_text("Abstract: "), new_x=XPos.RIGHT, new_y=YPos.TOP)
        self.set_font("Times", "", 11)
        self.multi_cell(0, 6, _clean_text(abstract_text))

    def add_table_row(self, col1: Any, col2: Any) -> None:
        self.set_font("Times", "B", 10)
        self.cell(80, 8, _clean_text(str(col1)), border=1)
        self.set_font("Times", "", 10)
        self.cell(110, 8, _clean_text(str(col2)), border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def add_plot_with_explanation(
        self, image_path: str | Path, title: str, explanation: str, image_width: int = 170
    ) -> None:
        self.add_section_title(title)

        if image_path and os.path.exists(str(image_path)):
            if self.get_y() > 180:
                self.add_page()

            self.image(str(image_path), x=20, w=image_width)
            self.ln(5)
            self.set_font("Times", "", 11)
            self.multi_cell(0, 6, _clean_text(explanation))
            self.ln(5)
        else:
            self.set_font("Times", "I", 11)
            self.set_text_color(200, 0, 0)
            self.cell(
                0, 6,
                _clean_text(f"[Error: Plot file '{image_path}' not found.]"),
                new_x=XPos.LMARGIN, new_y=YPos.NEXT,
            )
            self.set_text_color(0, 0, 0)


def generate_final_pdf(
    env_name: str,
    state_dim: int,
    action_dim: int,
    best_config: Dict[str, Any],
    best_mse: float,
    best_rmse: float,
    latency: float,
    use_pinn: bool,
    timestamp: str,
    abstract_text: str,
    conclusion_text: str,
    success_score: Optional[float],
    model_status: str,
    complexity_label: str,
    output_dir: str | Path = ".",
    figures_dir: Optional[str | Path] = None,
    architecture: str = "MLP",
    physics_mode: str = "pinn",
    pcnn_hard_constraint_states: Optional[list[str]] = None,
    rollout_horizon: int = 1,
    lstm_seq_length: int = 10,
    prefix: str = "system_id",
    filename: Optional[str] = None,
) -> str:
    """
    Build the engineering manuscript and return the written PDF path.

    ``figures_dir`` defaults to ``output_dir/figures`` - the location
    ``reporting.plots`` writes to.
    """
    output_dir = Path(output_dir)
    report_dir = output_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(figures_dir) if figures_dir else (output_dir / "figures")

    if filename is None:
        filename = f"System_Report_{env_name}_{timestamp}.pdf"
    path = report_dir / filename

    pdf = SystemIDReport()
    pdf.add_page()

    # --- ABSTRACT & SPECS ------------------------------------------------
    pdf.add_abstract(abstract_text)

    # --- SECTION 1: SYSTEM STRUCTURE -------------------------------------
    pdf.add_section_title("1. System Structure & Parameterization")
    pdf.set_font("Times", "", 11)
    pdf.multi_cell(
        0, 6,
        _clean_text(
            "The system represents a dynamic plant mapping state and action vectors to their "
            "respective state derivatives. The model structure captures strong cross-coupling "
            "among the coordinates, characteristic of complex mechanical systems."
        ),
    )
    pdf.ln(3)

    base_arch = (architecture or "MLP").strip().upper()
    if base_arch == "LSTM":
        arch_display = f"LSTM (Memory Window: {lstm_seq_length} steps)"
    else:
        arch_display = "MLP (Memoryless State)"

    mode = str(physics_mode or "pinn").strip().lower()
    if use_pinn and mode == "pcnn":
        arch_mode = f"PCNN (equation-constrained) + {arch_display}"
    elif use_pinn:
        arch_mode = f"PINN (soft physics loss) + {arch_display}"
    else:
        arch_mode = f"Data-Driven {arch_display}"
    loss_profile = (
        f"Autoregressive Rollout (Horizon = {rollout_horizon} steps)"
        if rollout_horizon > 1
        else "Single-Step Supervised Loss"
    )

    pdf.add_table_row("Number of States (n_states)", state_dim)
    pdf.add_table_row("Number of Inputs (n_inputs)", action_dim)
    pdf.add_table_row("Target Prediction", "State Derivatives (X_dot)")
    pdf.add_table_row("Loss Optimization", loss_profile)
    pdf.add_table_row("Dataset Complexity", complexity_label)
    pdf.add_table_row("Architecture Mode", arch_mode)
    if use_pinn and mode == "pcnn":
        constraints = ", ".join(pcnn_hard_constraint_states or []) or "None (learned correction for every state)"
        pdf.add_table_row("Exact PCNN State Equations", constraints)

    # --- SECTION 2: OPTIMAL ARCHITECTURE & SCORES ------------------------
    hidden_layers = (best_config or {}).get("hidden_layers", []) or []
    pdf.add_section_title("2. Optimal Neural Architecture & Performance")
    pdf.add_table_row("Hidden Layers Topology", str(hidden_layers))
    pdf.add_table_row("Total Network Depth", f"{len(hidden_layers)} layers")
    pdf.add_table_row("Validation MSE", f"{best_mse:.6f}")
    pdf.add_table_row("Validation RMSE", f"{best_rmse:.6f}")
    pdf.add_table_row("Inference Latency", f"{latency:.3f} ms")
    pdf.add_table_row("Fit + Held-out Rollout Score", f"{success_score:.1f} / 100" if success_score is not None else "Unavailable")
    pdf.add_table_row("Run Evidence Rating", model_status)

    # --- PLOT PATHS -------------------------------------------------------
    target = "Xdot"
    mse_plot = fig_dir / f"{prefix}_{env_name}_{target}_mse_convergence_{timestamp}.png"
    rmse_plot = fig_dir / f"{prefix}_{env_name}_{target}_rmse_convergence_{timestamp}.png"
    contour_plot = fig_dir / f"{prefix}_{env_name}_{target}_layers_neurons_contour_{timestamp}.png"
    hyper_plot = fig_dir / f"{prefix}_{env_name}_{target}_hyperparameters_{timestamp}.png"
    latency_plot = fig_dir / f"{prefix}_{env_name}_latency_evolution_{timestamp}.png"

    # --- 3. MSE CONVERGENCE ----------------------------------------------
    pdf.add_page()
    mse_desc = (
        "Figure 1: Mean Squared Error (MSE) Convergence. The plot tracks the validation loss "
        "across tuning cycles. Sharp spikes represent 'Radical Escapes' initiated by the "
        "Explorer Agent to break out of local minima."
    )
    pdf.add_plot_with_explanation(mse_plot, "3. System Identification Convergence (MSE)", mse_desc)

    # --- 4. RMSE CONVERGENCE ---------------------------------------------
    rmse_desc = (
        "Figure 2: Root Mean Squared Error (RMSE) Convergence. RMSE provides a performance "
        "metric in the exact same physical units as the state derivatives, offering a more "
        "intuitive engineering grasp of the absolute prediction error magnitude across the "
        "multi-agent tuning cycles."
    )
    pdf.add_plot_with_explanation(rmse_plot, "4. Absolute Error Magnitude (RMSE)", rmse_desc)

    # --- 5. TOPOLOGY CONTOUR ---------------------------------------------
    contour_desc = (
        "Figure 3: Architecture Search Space Contour. This heat map visualizes the performance "
        "of various topologies tested during the run. The red star indicates the global minimum "
        "(Best Config) discovered by the agents."
    )
    pdf.add_plot_with_explanation(
        contour_plot, "5. Architecture Topology Space", contour_desc, image_width=140
    )

    # --- 6. HYPERPARAMETERS ----------------------------------------------
    pdf.add_page()
    hyper_desc = (
        "Figure 4: Hyperparameter Evolution. Tracking the adjustments to Learning Rate, Total "
        "Neurons, and Layer Depth. This validates the progression from broad exploration phases "
        "into precise local fine-tuning."
    )
    pdf.add_plot_with_explanation(hyper_plot, "6. Hyperparameter Mutation Trajectory", hyper_desc)

    # --- 7. LATENCY -------------------------------------------------------
    latency_desc = (
        "Figure 5: Measured model inference latency across candidate topologies. The final "
        "measurement is compared with the configured timing target; it does not establish "
        "end-to-end real-time behavior on the controller hardware."
    )
    pdf.add_plot_with_explanation(latency_plot, "7. Inference Timing Comparison", latency_desc)

    # --- 8. VERIFICATION PLOTS -------------------------------------------
    pdf.add_page()
    pdf.add_section_title("8. Data-Driven Held-Out Verification")
    pdf.set_font("Times", "", 11)
    pdf.multi_cell(
        0, 6,
        _clean_text(
            "To estimate prediction error beyond the training data, the neural network was rolled "
            "forward on each saved held-out test trajectory. Each trajectory is evaluated separately "
            "with its own initial state and recurrent history. These plots describe this split only; "
            "they do not prove generalization to other operating conditions or closed-loop stability."
        ),
    )
    pdf.ln(5)

    plots_added = 0
    # Find every verification chunk regardless of the exact naming.
    found_plots = sorted(glob.glob(str(fig_dir / f"*{env_name}*test_verification*.png")))
    for test_plot in found_plots:
        if os.path.exists(test_plot):
            if pdf.get_y() > 180:
                pdf.add_page()
            pdf.image(test_plot, x=20, w=170)
            pdf.ln(5)
            pdf.set_font("Times", "I", 10)
            pdf.cell(
                0, 6,
                _clean_text("Verification Sequence: True Dataset vs. NN Prediction"),
                align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT,
            )
            pdf.ln(10)
            plots_added += 1

    if plots_added == 0:
        pdf.set_font("Times", "I", 11)
        pdf.set_text_color(200, 0, 0)
        pdf.cell(0, 6, _clean_text("[No verification plots generated or found.]"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(0, 0, 0)

    # --- 9. CONCLUSION ----------------------------------------------------
    if pdf.get_y() > 190:
        pdf.add_page()
    pdf.add_section_title("9. Conclusion & Validation Limits")
    pdf.set_font("Times", "", 11)
    pdf.multi_cell(0, 6, _clean_text(conclusion_text))

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
    Package the run folder into ``SystemID_RunResults_<timestamp>.zip`` with the
    client delivery: README.md, model code and weights, and the PDF report.
    Internal plots and agent logs remain in the run folder, outside the ZIP.
    """
    run_dir = Path(run_dir)
    zip_filename = run_dir / f"SystemID_RunResults_{timestamp}.zip"
    controller_script_name = f"deployed_controller_{env_name}.py"

    print("\n" + "=" * 80)
    print("📦 PACKAGING FINAL PIPELINE RESULTS")
    print("=" * 80)

    readme_content = f"""# System Identification Model Package

**Run Timestamp:** {timestamp}
**Framework:** AgentSysID - Automated Multi-Agent Deep Learning & Control Framework

---

## Directory Structure

### 1. `deployment/`
* **Model weights (`*.pth`):** the final optimized neural network weights.
* **`{controller_script_name}`:** the standalone engine class that loads and executes
  the network (no framework dependency).
* **`NN.py`:** a ready-to-use inference script that initializes the model and shows the
  kinematic integration loop.

### 2. `report/`
The PDF engineering report documents the model, validation metrics, and remaining test limits.
"""
    readme_path = run_dir / "README.md"
    readme_path.write_text(readme_content, encoding="utf-8")

    with zipfile.ZipFile(zip_filename, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(readme_path, arcname="README.md")

        # Keep the runnable code and its weights together. Do not package the
        # internal agent transcripts or diagnostic plots as client downloads.
        dep_dir = run_dir / "deployment"
        n_dep = 0
        if dep_dir.is_dir():
            for f in sorted(dep_dir.iterdir()):
                if f.is_file() and f.suffix.lower() in {".py", ".pth"}:
                    zipf.write(f, arcname=f"deployment/{f.name}")
                    n_dep += 1
        for m in sorted(run_dir.glob("*.pth")):
            zipf.write(m, arcname=f"deployment/{m.name}")
            n_dep += 1

        report_dir = run_dir / "report"
        pdf_path = Path(pdf_filename) if pdf_filename and Path(pdf_filename).is_file() else None
        if pdf_path is None and report_dir.is_dir():
            pdf_path = next(iter(sorted(report_dir.glob("*.pdf"))), None)
        if pdf_path is not None and pdf_path.is_file():
            zipf.write(pdf_path, arcname=f"report/{pdf_path.name}")
        print(f"  ├── Added {n_dep} code/weight file(s) and {int(pdf_path is not None)} report PDF")

    print("-" * 80)
    print(f"  ✅ ZIP FILE CREATED SUCCESSFULLY: {zip_filename.name}")
    print(f"  📂 SAVED EXACTLY TO: {zip_filename.resolve()}")
    print("=" * 80 + "\n")
    return str(zip_filename)
