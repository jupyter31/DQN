# DQN

## Overview

This project is a paper replication of DeepMind's DQN [[link to paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)] agent as well as an attempt to implement Q-Learning algorithms for reinforcement learning using OpenAI Gym environments. 


- Linear Approximation to Q-Function

        Suppose we represent the Q-function as:
        
        \[ Q_\theta(s, a) = \theta^\top \delta(s, a) \]
        
        where \( \theta \in \mathbb{R}^{|S||A|} \) and \( \delta: S \times A \to \mathbb{R}^{|S||A|} \) with:
        
        \[ [\delta(s, a)]_{s', a'} = 
        \begin{cases} 
        1 & \text{if } s' = s, a' = a \\ 
        0 & \text{otherwise} 
        \end{cases} \]

- Implementation of Deep Q-Network (DQN)
- Supports Atari games from the Arcade Learning Environment (ALE).
- Configuration management using YAML.
- Training and evaluation scripts.
- Logging and model checkpointing.

## Installation

### Prerequisites

- Python 3.8+
- Conda (recommended for environment management)

### Step-by-Step Installation

1. **Clone the Repository**

    ```sh
    git clone https://github.com/yourusername/dqn
    cd dqn
    ```

2. **Set Up the Conda Environment**

    Create a new conda environment using the provided `environment.yml` file:

    ```sh
    conda env create -f environment.yml
    conda activate rl_env
    ```

3. **Install the Package**

    Install the package in editable mode:

    ```sh
    pip install -e .
    ```

4. **Import Atari ROMs**

    Download the Atari ROMs and import them:

    ```sh
    pip install gym[accept-rom-license]
    ale-import-roms path/to/your/atari_roms
    ```

## Usage
```sh
python main.py --config_filename config.yml
```

### Configuration

The configuration for the project is managed using a YAML file. Below is an example configuration (`config.yml`):

```yaml
model: "dqn"
env:
  env_name: "ALE/Pong-v5"
  render_mode: "human"
output:
  output_path: "results/"
  log_path: "results/log.txt"
  plot_output: "results/scores.png"
model_training:
  num_episodes_test: 20
  grad_clip: True
  clip_val: 10
  saving_freq: 5000
  log_freq: 50
  eval_freq: 1000
  soft_epsilon: 0
  device: "cpu"  # cpu/gpu
  compile: False
  compile_mode: "default"
hyper_params:
  nsteps_train: 10000
  batch_size: 32
  buffer_size: 1000
  target_update_freq: 500
  gamma: 0.99
  learning_freq: 4
  state_history: 4
  lr_begin: 0.005
  lr_end: 0.001
  lr_nsteps: 5000
  eps_begin: 1
  eps_end: 0.01
  eps_nsteps: 5000
  learning_start: 200
