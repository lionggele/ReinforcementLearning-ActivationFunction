# Redefining Activation Dynamics: An Empirical Comparison of Oscillating, Mixture, and Adaptive Activation Functions in Deep Reinforcement Learning

Welcome to the Task 2 directory! This directory is part of the AdaAF (Adaptive Activation Function) project, which explores various activation functions in deep learning models. Below is a structured outline of the directory with descriptions for each component.

## Methodology

### A. Environment and Experiment Design
This study investigates the impact of various activation functions on Deep Q-Networks (DQNs) within the "CartPole-v1" environment from the Gymnasium. The task involves balancing a pole on a moving cart, with the success measured by the agent’s ability to maintain balance for 200 consecutive steps. We conducted 15 trials for each activation function, recording key metrics such as the number of episodes required to achieve 300 timesteps, cumulative pole balance duration (Duration Reward), and pole verticality (Auxiliary Reward).

### B. DQN Model
We based our experiments on a PyTorch implementation of DQN. We integrated different activation functions into the model, following consistent hyperparameters as outlined in the original works. The models were trained using the selected activation functions to evaluate their performance.

### C. Training Strategy
We explored several activation function strategies, drawing insights from comprehensive evaluations of various activation functions. The selected activation functions include ReLU, Tanh, GCU, SQU, PReLU, and ELU. Mixed activation functions were implemented by combining different activation functions within a single network, enhancing learning efficiency and adaptability. Additionally, we experimented with gated activations that adaptively learn a gating mask, offering dynamic modulation based on the input.

In another experiment, hierarchical activations utilized varied functions like ReLU, ELU, PReLU, Tanh, GCU, and SQU, applied layer-wise and integrated using a winner-take-all strategy. This hierarchical approach enhances adaptability by filtering through multiple activation responses and selecting the most effective ones.

### D. Evaluation and Ranking
Activation functions were ranked based on learning speed, reward accumulation, and consistency across trials. Metrics such as the number of steps to solve the task, cumulative rewards, and standard deviations were used to assign ranks. A composite score was calculated by summing ranks across all metrics, with the method having the lowest total score ranked highest.

## Directory Structure Overview

### AdaAF
- **Ada_learnableAF**
  - `graph`: Contains graphs related to Adaptive activation functions.
  - `DQN_PRELU-ELU.py`: Implements DQN with PReLU and ELU activation functions.

### Gated_Activation
- `graph`: Contains graphs related to Gated activation functions.
  - `Ada_Gated.py`: Implements Ada-gated activation functions.

### Hierarchical Structure
- `graph`: Contains graphs related to Hierarchical activation functions.
  - `DQN-AdaAFOAF-All.py`: Implements DQN with AdaAFOAF for all conditions.
  - `myplotgraphbefore.png`: Pre-run plot for the hierarchical structure.

### AF and OAF
#### AF
- `final graph`: Contains final graph AF.
- `graph`: Contains graphs related to activation functions.
  - `DQN-Tanh-RELU.py`: Implements DQN with Tanh and ReLU activation functions.

#### OAF
- `GCU`
  - `graph`: Contains graphs related to GCU activation functions.
- `SQU`
  - `graph`: Contains graphs related to SQU activation functions.
  - `DQN-GCU-SQU.py`: Implements DQN with GCU and SQU activation functions.

### MixAF
- `graph`: Contains graphs related to Mixed activation functions.
- `MixAF.py`: Implements mixed activation functions.

### PlotGraphBefore
- `plotgrph.py`: Script to plot graphs before the runs.
  - `myplotgraphbefore.png`: Pre-run plot for the hierarchical structure.

## Result Pictures
The resulting pictures are inside each of the graph folders.

## Usage

To execute the scripts or view the graphs, navigate to the respective subdirectory and follow the experimental design detailed in the conference paper.
