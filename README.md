# Deep Q-Learning for Atari Bank Heist

A comprehensive implementation of Deep Q-Network (DQN) reinforcement learning agent for the Atari game Bank Heist, including systematic hyperparameter experimentation and analysis.

## Project Overview

This project implements and evaluates a Deep Q-Learning agent for Bank Heist, exploring how different hyperparameters affect learning efficiency. Through systematic experimentation across 2,700 training episodes, I identified optimal configurations that improved learning rate by 158% compared to baseline.

## Key Results

| Configuration | Episodes | Avg Score | Improvement |
|---------------|----------|-----------|-------------|
| **Baseline** | 1000 | 493.0 | Reference |
| Baseline (400 eps) | 400 | 27.9 | Early baseline |
| **Exp 1: Higher Alpha** | 400 | 8.8 | -68% (Failed) |
| **Exp 2: Lower Gamma** | 400 | 23.0 | -18% (Worse) |
| **Exp 3: Slower Epsilon** | 400 | **71.9** | **+158% (Good)** |
| **Exp 4: Boltzmann Policy** | 500 | 38.7 | +39% (Moderate) |

### Key Finding
**Slower epsilon decay (0.99 vs 0.995) achieved 2.6x better learning efficiency**, demonstrating that extended exploration significantly improves early-stage performance in complex navigation tasks.

## Architecture

### DQN Network
- **Input:** 4 stacked grayscale frames (84×84 pixels)
- **Architecture:** 
  - Conv2D(4→32, kernel=8, stride=4)
  - Conv2D(32→64, kernel=4, stride=2)
  - Conv2D(64→64, kernel=3, stride=1)
  - Fully Connected(3136→512)
  - Fully Connected(512→18)
- **Output:** Q-values for 18 discrete actions

### Key Components
- **Experience Replay:** 30,000 capacity buffer
- **Target Network:** Updated every 10 episodes
- **Batch Size:** 32 experiences
- **Optimizer:** Adam with learning rate 0.00025

## Experiments Conducted

### Experiment 1: Learning Rate (Alpha)
- **Hypothesis:** Higher learning rate accelerates training
- **Configuration:** α = 0.0005 (2x baseline)
- **Result:** Failed (8.8 avg) - caused training instability
- **Conclusion:** Aggressive learning rates harm convergence

### Experiment 2: Discount Factor (Gamma)
- **Hypothesis:** Lower gamma focuses on immediate rewards
- **Configuration:** γ = 0.95 (vs 0.99 baseline)
- **Result:** Worse (23.0 avg) - reduced long-term planning
- **Conclusion:** Bank Heist requires future-reward valuation

### Experiment 3: Epsilon Decay 
- **Hypothesis:** Slower decay allows more exploration
- **Configuration:** decay = 0.99 (vs 0.995 baseline)
- **Result:** **Winner (71.9 avg)** - 158% improvement
- **Conclusion:** Extended exploration discovers better strategies

### Experiment 4: Exploration Policy
- **Hypothesis:** Boltzmann exploration outperforms ε-greedy
- **Configuration:** Temperature-based softmax selection
- **Result:** Moderate (38.7 avg) - 39% improvement
- **Conclusion:** ε-greedy better suited for discrete action spaces

## 📈 Training Details

### Baseline Configuration
```python
total_episodes = 1000
max_steps = 2000
learning_rate = 0.00025  # Alpha in Bellman equation
gamma = 0.99             # Discount factor
epsilon_start = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 32
memory_size = 30000
```

### Environment Details
- **Game:** Bank Heist (ALE/BankHeist-v5)
- **State Space:** Continuous (84×84×4 stacked frames)
- **Action Space:** Discrete (18 actions)
- **Reward Structure:** Native game scoring (points for robbing banks)

## Key Learnings

### Technical Insights
1. **Exploration-exploitation balance is critical** - Slower epsilon decay (0.99) dramatically improved learning
2. **Learning rate stability matters** - 2x increase caused complete failure
3. **Future planning required** - Lower gamma hurt navigation tasks
4. **Policy selection important** - ε-greedy outperformed Boltzmann for discrete actions

### Libraries Used
- PyTorch for neural networks
- Gymnasium/ALE for Atari environments
- OpenCV for image processing
- NumPy for numerical operations
