### README: Learning Complex Quadrotor Maneuvers Using Intrinsic Curiosity and Multi-Agent LLM-Assisted Reward Optimization

---

## Overview

This repository contains the code, datasets, and simulation configurations for the research paper:

**Title**: Learning Complex Quadrotor Maneuvers Using Intrinsic Curiosity and Multi-Agent LLM-Assisted Reward Optimization  
**Authors**: Renzo Balcazar Tapia, Shengyang Wang, Wenli Xiao, Bing Luo  
**Published In**: AAMAS 2025 (Association for the Advancement of Artificial Intelligence)

The research introduces a novel framework that combines intrinsic curiosity modules, multi-agent reinforcement learning (MARL), and large language model (LLM)-assisted optimization to enable quadrotor drones to perform complex maneuvers in dynamic environments. Key contributions include:

- **Intrinsic Curiosity Modules**: Integration of Hash Neural Networks for enhanced exploration in high-dimensional quaternion-based state spaces.
- **Multi-Agent Interaction**: Training drones with inter-agent interaction to simulate crashes and improve swarm coordination.
- **LLM-Assisted Optimization**: Dynamic refinement of reward functions and neural network architectures via a multi-agent LLM pipeline.

---

## Repository Structure

```
.
├── README.md            # This README file
├── src/                 # Source code for experiments and simulations
│   ├── main.py          # Entry point for training and simulation
│   ├── environment/     # Environment setup and customization
│   │   ├── quad_sim.py  # Custom quadrotor simulation environment
│   │   ├── dynamics.py  # Drone dynamics and control modeling
│   │   ├── reward.py    # Reward functions
│   │   └── config/      # Configurations for simulation
│   ├── agents/          # RL agents and LLM pipeline
│   │   ├── ppo.py       # Proximal Policy Optimization (PPO) implementation
│   │   ├── curiosity.py # Intrinsic curiosity module (HashNN integration)
│   │   ├── llm_pipeline.py # Multi-agent LLM optimization
│   │   └── utils.py     # Helper functions
│   └── visualization/   # Visualization and logging tools
│       ├── plots.py     # Script for generating training metrics plots
│       ├── logs/        # Logs and output files
│       └── metrics/     # Training and evaluation metrics
├── data/                # Pre-processed datasets or generated simulation data
├── results/             # Results and experiment outputs
├── docs/                # Documentation for the project
│   ├── methodology.md   # Detailed methodology and experimental setup
│   └── architecture.md  # Explanation of model and framework architecture
├── images/              # Images for simulation visuals and documentation
├── requirements.txt     # Python dependencies
├── LICENSE              # License information
└── CONTRIBUTING.md      # Contribution guidelines
```

---

## Installation

### Prerequisites
- Python 3.8+
- NVIDIA Isaac Sim (OmniIsaacGym)
- PyTorch
- NVIDIA GPU with CUDA support

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/quadrotor-llm.git
   cd quadrotor-llm
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv env
   source env/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up NVIDIA Isaac Sim:
   - Download and install Isaac Sim from NVIDIA's website.
   - Follow their setup guide to ensure compatibility.

5. Run the training script:
   ```bash
   python src/main.py --config src/environment/config/simulation_config.yaml
   ```

---

## Features

- **Simulation Environment**: High-fidelity quadrotor simulation using NVIDIA Isaac Sim, including customizable drone dynamics.
- **Intrinsic Curiosity Module**: Implements Hash Neural Networks to enable efficient exploration.
- **LLM Pipeline**: Automates optimization of reward functions and network architectures via multi-agent LLM feedback.
- **Multi-Agent Interaction**: Supports collaborative drone swarm simulations with crash handling and inter-agent interactions.

---

## Usage

### Training
Run the training script to train quadrotor agents:
```bash
python src/main.py --train
```

### Evaluation
Evaluate trained models on predefined test cases:
```bash
python src/main.py --evaluate --checkpoint <path_to_checkpoint>
```

### Visualization
Generate plots and visualize simulation performance:
```bash
python src/visualization/plots.py
```

---

## Results

- Single drone maneuvers, including flips and hovering, achieve high stability and efficiency.
- Coordinated swarm behavior improves through interaction training.
- LLM-assisted optimization enhances reward structure and neural architecture.

See the `results/` directory for detailed metrics and simulation videos.

---

## Citation

If you use this work in your research, please cite the paper:

```
@article{balcazar2025quadrotor,
  title={Learning Complex Quadrotor Maneuvers Using Intrinsic Curiosity and Multi-Agent LLM-Assisted Reward Optimization},
  author={Renzo Balcazar Tapia, Shengyang Wang, Wenli Xiao, Bing Luo},
  journal={AAMAS 2025},
  year={2025}
}
```

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contributing

We welcome contributions! Please see the `CONTRIBUTING.md` file for details on how to get started.

---

## Contact

For questions or issues, contact the corresponding author:
**Bing Luo** (bl291@dukekunshan.edu.cn)
