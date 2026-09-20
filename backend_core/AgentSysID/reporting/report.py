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
    7. Real-Time Deployment Viability
    8. Data-Driven Held-Out Verification
    9. Concluding Remarks & Deployment Synthesis

ZIP layout matches the original backend delivery::

    SystemID_RunResults_<timestamp>.zip
    ├── README.md
    ├── Agents_log/
    ├── figures/
    ├── deployment/
    └── report/
"""

from __future__ import annotations

import datetime
import glob
import hashlib
import os
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

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
            _clean_text("System Identification and Dynamic Stability Report"),
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
    success_score: float,
    model_status: str,
    complexity_label: str,
    output_dir: str | Path = ".",
    figures_dir: Optional[str | Path] = None,
    architecture: str = "MLP",
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

    arch_mode = f"PINN + {arch_display}" if use_pinn else f"Data-Driven {arch_display}"
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

    # --- SECTION 2: OPTIMAL ARCHITECTURE & SCORES ------------------------
    hidden_layers = (best_config or {}).get("hidden_layers", []) or []
    pdf.add_section_title("2. Optimal Neural Architecture & Performance")
    pdf.add_table_row("Hidden Layers Topology", str(hidden_layers))
    pdf.add_table_row("Total Network Depth", f"{len(hidden_layers)} layers")
    pdf.add_table_row("Validation MSE", f"{best_mse:.6f}")
    pdf.add_table_row("Validation RMSE", f"{best_rmse:.6f}")
    pdf.add_table_row("Inference Latency", f"{latency:.3f} ms")
    pdf.add_table_row("Composite Success Score", f"{success_score:.1f} / 100")
    pdf.add_table_row("Deployment Status", model_status)

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
        "Figure 5: Computational Inference Latency. Tracks the forward-pass execution time "
        "across topologies. The active constraint ensures the final model remains mathematically "
        "viable for real-time optimal control."
    )
    pdf.add_plot_with_explanation(latency_plot, "7. Real-Time Deployment Viability", latency_desc)

    # --- 8. VERIFICATION PLOTS -------------------------------------------
    pdf.add_page()
    pdf.add_section_title("8. Data-Driven Held-Out Verification")
    pdf.set_font("Times", "", 11)
    pdf.multi_cell(
        0, 6,
        _clean_text(
            "To verify the system learned generalized physical dynamics rather than memorizing "
            "the training set, the neural network was evaluated on sequentially held-out test "
            "data. The plots below simulate the network's predictions rolling forward over time."
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
    pdf.add_section_title("9. Concluding Remarks & Deployment Synthesis")
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
    delivery layout: README.md, figures/, deployment/, report/, Agents_log/.
    """
    run_dir = Path(run_dir)
    zip_filename = run_dir / f"SystemID_RunResults_{timestamp}.zip"
    controller_script_name = f"deployed_controller_{env_name}.py"

    print("\n" + "=" * 80)
    print("📦 PACKAGING FINAL PIPELINE RESULTS")
    print("=" * 80)

    readme_content = f"""# System Identification & Neural Controller Package

**Run Timestamp:** {timestamp}
**Framework:** AgentSysID - Automated Multi-Agent Deep Learning & Control Framework

---

## Directory Structure

### 1. `figures/`
Contains all high-resolution diagnostic plots generated during THIS specific run:
MSE / RMSE convergence, hyperparameter evolution, architecture contour, latency
evolution and the held-out test verification.

### 2. `deployment/`
* **Model Weights (`*.pth`):** the final optimized PyTorch neural network weights.
* **`{controller_script_name}`:** the standalone engine class that parses and executes
  the network (no framework dependency).
* **`NN.py`:** a ready-to-use inference script that initializes the model and shows the
  kinematic integration loop.

### 3. `report/`
PDF engineering report documenting the model, its metrics and its deployment status.

### 4. `Agents_log/`
Full LLM conversation history: prompts, responses and token usage for every agent turn.
"""
    readme_path = run_dir / "README.md"
    readme_path.write_text(readme_content, encoding="utf-8")

    with zipfile.ZipFile(zip_filename, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        # --- Root README --------------------------------------------------
        zipf.write(readme_path, arcname="README.md")
        print("  ├── Added root 'README.md' documentation")

        # --- FOLDER 1: FIGURES --------------------------------------------
        fig_dir = run_dir / "figures"
        n_fig = 0
        if fig_dir.is_dir():
            for img in sorted(fig_dir.glob("*.png")):
                zipf.write(img, arcname=f"figures/{img.name}")
                n_fig += 1
        print(f"  ├── [Folder 1] Added {n_fig} plots to 'figures/'")

        # --- FOLDER 2: DEPLOYMENT -----------------------------------------
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
        print(f"  ├── [Folder 2] Added {n_dep} deployment files to 'deployment/'")

        # --- FOLDER 3: REPORT ---------------------------------------------
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
        print(f"  ├── [Folder 3] Added {n_rep} PDF(s) to 'report/'")

        # --- FOLDER 4: AGENTS LOG -----------------------------------------
        # Agents_log/ wins; a root-level log is only added when it is not the
        # same content under another name, so a log never ships twice.
        n_log = 0
        seen_names: List[str] = []
        seen_digests: List[str] = []

        def _add_log(path: Path) -> bool:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if path.name in seen_names or digest in seen_digests:
                return False
            zipf.write(path, arcname=f"Agents_log/{path.name}")
            seen_names.append(path.name)
            seen_digests.append(digest)
            return True

        agents_dir = run_dir / "Agents_log"
        if agents_dir.is_dir():
            for logf in sorted(agents_dir.iterdir()):
                if logf.is_file() and _add_log(logf):
                    n_log += 1

        for pattern in ("*.txt", "*.log"):
            for logf in sorted(run_dir.glob(pattern)):
                if logf.name.lower().startswith("readme"):
                    continue
                if _add_log(logf):
                    n_log += 1
        print(f"  ├── [Folder 4] Added {n_log} log file(s) to 'Agents_log/'")

    print("-" * 80)
    print(f"  ✅ ZIP FILE CREATED SUCCESSFULLY: {zip_filename.name}")
    print(f"  📂 SAVED EXACTLY TO: {zip_filename.resolve()}")
    print("=" * 80 + "\n")
    return str(zip_filename)
