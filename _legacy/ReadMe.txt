# Agentic System Identification Framework

A LangGraph-orchestrated, multi-agent deep learning framework for state-constrained system identification. Built to automate the hyperparameter tuning of physical dynamics models (MLP/LSTM) for complex tasks like autonomous F1 racing trajectory optimization and nonlinear control systems.

This framework entirely decouples deterministic PyTorch mathematics from OpenAI/Groq reasoning agents, establishing a fully modular, Human-in-the-Loop (HIL) pipeline for discovering optimal network architectures.

---

## 🚀 Key Features

* **Multi-Agent Orchestration (LangGraph):**
  * **Initializer:** Analyzes dataset complexity (Tier 1-5) and sets bounded search spaces.
  * **Data Inspector (HIL):** Identifies mathematical anomalies (e.g., dead sensors, high multicollinearity) and requests engineer intervention before training.
  * **Critic & Actor:** Evaluates validation loss, generalization gaps (overfitting ratios), and real-time inference latency to incrementally fine-tune the architecture.
  * **Explorer:** Detects stagnation (local minima) and forces radical topological inversions to escape gradient trenches.
  * **Report Agent:** Authors a final professional engineering manuscript summarizing the deployed model.
* **Physics-Informed Neural Networks (PINN):** Integrates known kinematic equations directly into the loss function to ensure dynamically viable rollout predictions.
* **Automated Post-Processing:** Automatically extracts the best PyTorch weights, generates a standalone Python inference script, compiles a PDF performance report, and packages everything into a clean deployment ZIP file.

---

## 💻 System Requirements

* **Python:** 3.10 or higher.
* **Hardware:** Optimized for CUDA-enabled GPUs paired with multi-core processors to handle rapid PyTorch tensor operations alongside asynchronous LLM API calls.
* **API Keys:** Requires active keys for OpenAI, Groq, or OpenRouter depending on your backend choice.

---

## 🛠️ Installation

1. **Clone the repository and navigate to the root directory.**
2. **Create and activate a virtual environment.**
3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt