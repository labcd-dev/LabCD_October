import os
import datetime
from fpdf import FPDF


def clean_text(text):
    """
    Bulletproof text sanitizer.
    Replaces special typographical characters and completely strips
    any remaining characters that the PDF 'latin-1' core fonts cannot handle.
    """
    if not isinstance(text, str):
        return str(text)

    # Replace common AI-generated typographical characters with ASCII
    replacements = {
        '\u2011': '-',  # Non-breaking hyphen
        '\u2012': '-',  # Figure dash
        '\u2013': '-',  # En-dash
        '\u2014': '-',  # Em-dash
        '\u2018': "'",  # Left single quote
        '\u2019': "'",  # Right single quote
        '\u201c': '"',  # Left double quote
        '\u201d': '"',  # Right double quote
        '\u2022': '-',  # Bullet
        '\u00a0': ' ',  # Non-breaking space
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)

    # Force encode to latin-1 and IGNORE anything else that won't fit
    return text.encode('latin-1', 'ignore').decode('latin-1')


class SystemIDReport(FPDF):
    def header(self):
        self.set_font("Times", "B", 14)
        self.cell(0, 8, clean_text("System Identification and Dynamic Stability Report"), border=False, ln=True,
                  align="L")

        self.set_font("Times", "B", 12)
        self.cell(0, 6, clean_text("Automated Core Engineering Analysis Engine"), border=False, ln=True, align="L")

        self.set_font("Times", "I", 11)
        self.cell(0, 6, clean_text("LabCD.ai"), border=False, ln=True, align="L")

        self.set_font("Times", "", 10)
        date_str = f"Evaluation Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        self.cell(0, 6, clean_text(date_str), border=False, ln=True, align="L")

        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Times", "I", 8)
        self.cell(0, 10, clean_text(f"Confidential - Automated Computational Engineering Core | Page {self.page_no()}"),
                  align="C")

    def add_section_title(self, title):
        self.ln(5)
        self.set_font("Times", "B", 12)
        self.set_fill_color(220, 220, 220)
        self.cell(0, 8, clean_text(title), border=True, ln=True, align="L", fill=True)
        self.ln(3)

    def add_abstract(self, abstract_text):
        self.set_font("Times", "B", 11)
        self.cell(20, 6, clean_text("Abstract: "), ln=False)
        self.set_font("Times", "", 11)
        # Apply clean_text explicitly here to prevent the crash!
        self.multi_cell(0, 6, clean_text(abstract_text))

    def add_table_row(self, col1, col2):
        self.set_font("Times", "B", 10)
        self.cell(80, 8, clean_text(str(col1)), border=1)
        self.set_font("Times", "", 10)
        self.cell(110, 8, clean_text(str(col2)), border=1, ln=True)

    def add_plot_with_explanation(self, image_path, title, explanation, image_width=170):
        self.add_section_title(title)

        if os.path.exists(image_path):
            if self.get_y() > 180:
                self.add_page()

            self.image(image_path, x=20, w=image_width)
            self.ln(5)
            self.set_font("Times", "", 11)
            self.multi_cell(0, 6, clean_text(explanation))
            self.ln(5)
        else:
            self.set_font("Times", "I", 11)
            self.set_text_color(200, 0, 0)
            self.cell(0, 6, clean_text(f"[Error: Plot file '{image_path}' not found.]"), ln=True)
            self.set_text_color(0, 0, 0)


def generate_final_pdf(env_name, state_dim, action_dim, best_config, best_mse, best_rmse, latency, use_pinn, timestamp,
                       abstract_text, conclusion_text, success_score, model_status, complexity_label):
    pdf = SystemIDReport()
    pdf.add_page()

    # --- ABSTRACT & SPECS ---
    pdf.add_abstract(abstract_text)

    # --- SECTION 1: SYSTEM STRUCTURE ---
    pdf.add_section_title("1. System Structure & Parameterization")
    pdf.set_font("Times", "", 11)
    pdf.multi_cell(0, 6, clean_text(
        "The system represents a dynamic plant mapping state and action vectors to their respective state derivatives. "
        "The model structure captures strong cross-coupling among the coordinates, characteristic of complex mechanical systems."))
    pdf.ln(3)

    # 🧠 DYNAMIC PDF ARCHITECTURE & LOSS CHECK
    try:
        from config import NETWORK_ARCHITECTURE, ROLLOUT_HORIZON, LSTM_SEQ_LENGTH
        base_arch = NETWORK_ARCHITECTURE.strip().upper()
        horizon = ROLLOUT_HORIZON
        seq_len = LSTM_SEQ_LENGTH
    except ImportError:
        base_arch = "MLP"
        horizon = 1
        seq_len = 10

    if base_arch == "LSTM":
        arch_display = f"LSTM (Memory Window: {seq_len} steps)"
    else:
        arch_display = "MLP (Memoryless State)"

    arch_mode = f"PINN + {arch_display}" if use_pinn else f"Data-Driven {arch_display}"
    loss_profile = f"Autoregressive Rollout (Horizon = {horizon} steps)" if horizon > 1 else "Single-Step Supervised Loss"

    pdf.add_table_row("Number of States (n_states)", state_dim)
    pdf.add_table_row("Number of Inputs (n_inputs)", action_dim)
    pdf.add_table_row("Target Prediction", "State Derivatives (X_dot)")
    pdf.add_table_row("Loss Optimization", loss_profile)
    pdf.add_table_row("Dataset Complexity", complexity_label)
    pdf.add_table_row("Architecture Mode", arch_mode)

    # --- SECTION 2: OPTIMAL ARCHITECTURE & SCORES ---
    pdf.add_section_title("2. Optimal Neural Architecture & Performance")
    pdf.add_table_row("Hidden Layers Topology", str(best_config['hidden_layers']))
    pdf.add_table_row("Total Network Depth", f"{len(best_config['hidden_layers'])} layers")
    pdf.add_table_row("Validation MSE", f"{best_mse:.6f}")
    pdf.add_table_row("Validation RMSE", f"{best_rmse:.6f}")
    pdf.add_table_row("Inference Latency", f"{latency:.3f} ms")
    pdf.add_table_row("Composite Success Score", f"{success_score:.1f} / 100")  # <--- NEW: Added 0-100 Score
    pdf.add_table_row("Deployment Status", model_status)                        # <--- NEW: Added STABLE/UNSTABLE tag

    # --- PLOTS CONFIGURATION ---
    prefix = "system_id"
    target = "Xdot"
    mse_plot = f"{prefix}_{env_name}_{target}_mse_convergence_{timestamp}.png"
    rmse_plot = f"{prefix}_{env_name}_{target}_rmse_convergence_{timestamp}.png"
    contour_plot = f"{prefix}_{env_name}_{target}_layers_neurons_contour_{timestamp}.png"
    hyper_plot = f"{prefix}_{env_name}_{target}_hyperparameters_{timestamp}.png"
    latency_plot = f"{prefix}_{env_name}_latency_evolution_{timestamp}.png"

    # --- 3. MSE CONVERGENCE ---
    pdf.add_page()
    mse_desc = (
        "Figure 1: Mean Squared Error (MSE) Convergence. The plot tracks the validation loss across tuning cycles. "
        "Sharp spikes represent 'Radical Escapes' initiated by the Explorer Agent to break out of local minima.")
    pdf.add_plot_with_explanation(mse_plot, "3. System Identification Convergence (MSE)", mse_desc)

    # --- 4. RMSE CONVERGENCE ---
    rmse_desc = (
        "Figure 2: Root Mean Squared Error (RMSE) Convergence. RMSE provides a performance metric in the exact "
        "same physical units as the state derivatives, offering a more intuitive engineering grasp of the absolute "
        "prediction error magnitude across the multi-agent tuning cycles.")
    pdf.add_plot_with_explanation(rmse_plot, "4. Absolute Error Magnitude (RMSE)", rmse_desc)

    # --- 5. TOPOLOGY CONTOUR ---
    contour_desc = ("Figure 3: Architecture Search Space Contour. This heat map visualizes the performance of various "
                    "topologies tested during the run. The red star indicates the global minimum (Best Config) discovered by the agents.")
    pdf.add_plot_with_explanation(contour_plot, "5. Architecture Topology Space", contour_desc, image_width=140)

    # --- 6. HYPERPARAMETERS ---
    pdf.add_page()
    hyper_desc = (
        "Figure 4: Hyperparameter Evolution. Tracking the adjustments to Learning Rate, Total Neurons, and Layer Depth. "
        "This validates the progression from broad exploration phases into precise local fine-tuning.")
    pdf.add_plot_with_explanation(hyper_plot, "6. Hyperparameter Mutation Trajectory", hyper_desc)

    # --- 7. LATENCY ---
    latency_desc = (
        "Figure 5: Computational Inference Latency. Tracks the forward-pass execution time across topologies. "
        "The active constraint ensures the final model remains mathematically viable for real-time optimal control.")
    pdf.add_plot_with_explanation(latency_plot, "7. Real-Time Deployment Viability", latency_desc)

    # --- 8. VERIFICATION PLOTS (MAX 3) ---
    pdf.add_page()
    pdf.add_section_title("8. Data-Driven Held-Out Verification")
    pdf.set_font("Times", "", 11)
    pdf.multi_cell(0, 6, clean_text(
        "To verify the system learned generalized physical dynamics rather than memorizing the training set, the neural network was evaluated on sequentially held-out test data. The plots below simulate the network's predictions rolling forward over time."))
    pdf.ln(5)

    plots_added = 0
    import glob

    # ⚠️ BULLETPROOF FIX: Use glob to automatically find all test verification chunks regardless of exact naming!
    found_plots = sorted(glob.glob(f"*{env_name}*test_verification*.png"))

    for test_plot in found_plots:
        if os.path.exists(test_plot):
            if pdf.get_y() > 180:
                pdf.add_page()
            pdf.image(test_plot, x=20, w=170)
            pdf.ln(5)
            pdf.set_font("Times", "I", 10)
            pdf.cell(0, 6, clean_text(f"Verification Sequence: True Dataset vs. NN Prediction"), align="C", ln=True)
            pdf.ln(10)
            plots_added += 1

    if plots_added == 0:
        pdf.set_font("Times", "I", 11)
        pdf.set_text_color(200, 0, 0)
        pdf.cell(0, 6, clean_text("[No verification plots generated or found.]"), ln=True)
        pdf.set_text_color(0, 0, 0)

    # --- 9. CONCLUSION ---
    if pdf.get_y() > 190:
        pdf.add_page()
    pdf.add_section_title("9. Concluding Remarks & Deployment Synthesis")
    pdf.set_font("Times", "", 11)

    # Apply clean_text explicitly here!
    pdf.multi_cell(0, 6, clean_text(conclusion_text))

    # Save the PDF to the hard drive
    pdf_filename = f"System_Report_{env_name}_{timestamp}.pdf"
    pdf.output(pdf_filename)

    # Explicitly return the string filename back to main.py
    return pdf_filename