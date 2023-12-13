# Comparison between OAF - GCU & NMC


# === Comment out when necessary (Jupyter notebook or not)
# %matplotlib inline
# !pip3 install gymnasium[classic_control]

# === High-level parameters
# Mode
mode = 1  # 1 = experimental, 2 = rendering test
# Main experimental parameters
p_exper = {}  # experimental parameters
p_exper['num_trials'] = 1 # number of trials
p_exper['num_episodes'] = 10  # number of episodes per trial
p_exper['max_ep_len'] = 300  # maximum episode length
p_exper['leng_solved'] = 200  # length of episode considered solved
p_exper['batch_size'] = 128
p_exper['num_nodes'] = 128  # network layer width
p_exper['gamma'] = 0.99
p_exper['eps_start'] = 0.9
p_exper['eps_end'] = 0.05
p_exper['eps_decay'] = 1000
p_exper['tau'] = 0.005
p_exper['rl_lr'] = 1e-3  # 1e-4 this will be better.
# Main test parameters
p_test = {}
p_test['condition'] = 3  # which condition
p_test['net_name'] = 'c3-net--16112023_1122'
p_test['max_ep_len'] = 300

# Adapted from the Reinforcement Learning (DQN) Tutorial by
# [Adam Paszke](https://github.com/apaszke) and [Mark Towers](https://github.com/pseudo-rnd-thoughts)

# === Imports ============================

import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime
from scipy.stats import mannwhitneyu

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# === Basic setup =========================

def is_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False


if not (is_colab()):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # change dir. to the one of the running script

env = gym.make("CartPole-v1")

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

# plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Support Data Structures ===============

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# === Support functions =======================

# Generate date/time string
def get_date_time_str():
    now = datetime.now()  # current date and time
    return now.strftime("%d%m%Y_%H%M")


# Generate filename for a policy net
def create_model_filename():
    core_name = ''  # os.path.basename(__file__)[0:-3] # [0:-3] --> to remove '.py'
    filename = 'net-' + core_name + '-' + get_date_time_str()
    return filename


# Get index of first value exceeding a threshold
def get_index_exceed(vec, val):
    # Find indices where the vector is greater than the value
    indices = np.where(vec > val)

    # Check if there are any indices found and get the first one
    if indices[0].size > 0:
        first_index = indices[0][0]
    else:
        first_index = -1

    return first_index


# Define the Adaptive DQN model with hierarchical activation functions
class AdaDQNall1(nn.Module):
    def __init__(self, num_nodes, n_observations, n_actions):
        super(AdaDQNall1, self).__init__()
        self.layer1 = nn.Linear(n_observations, num_nodes)
        self.layer2 = nn.Linear(num_nodes, num_nodes)
        self.layer3 = nn.Linear(num_nodes, n_actions)

        # Define activation functions
        self.elu = nn.ELU()
        self.prelu = nn.PReLU(num_parameters=1, init=0.25)

        # Learnable weights for gating
        self.gate = nn.Parameter(torch.randn(num_nodes))

    def forward(self, x):
        z = self.layer1(x)

        # Apply activation functions
        elu_act = self.elu(z)
        prelu_act = self.prelu(z)

        # Gating mechanism
        gate = torch.sigmoid(self.gate * z)

        elu = gate * elu_act
        prelu = (1 - gate) * prelu_act

        # Stack the activations for winner-take-all selection
        stacked_acts = torch.stack([elu, prelu], dim=0)
        adaptive_acts, _ = torch.max(stacked_acts, dim=0)

        # Second layer
        z = self.layer2(adaptive_acts)
        gate = torch.sigmoid(self.gate * z)

        elu = gate * elu_act
        prelu = (1 - gate) * prelu_act

        # Stack the activations for winner-take-all selection
        stacked_acts = torch.stack([elu, prelu], dim=0)
        adaptive_acts, _ = torch.max(stacked_acts, dim=0)

        # Pass the combined activations to the output layer
        return self.layer3(adaptive_acts)


# GCU Activation Function
def gcu_activation(x):
    return x * torch.cos(x)


class AdaDQNall2(nn.Module):
    def __init__(self, num_nodes, n_observations, n_actions):
        super(AdaDQNall2, self).__init__()
        self.layer1 = nn.Linear(n_observations, num_nodes)
        self.layer2 = nn.Linear(num_nodes, num_nodes)
        self.layer3 = nn.Linear(num_nodes, n_actions)

        # Define activation functions
        self.tanh = nn.Tanh()

        # Learnable weights for gating
        self.gate = nn.Parameter(torch.randn(num_nodes))

    def forward(self, x):
        z = self.layer1(x)

        # Apply activation functions
        tanh_act = self.tanh(z)
        gcu_act = gcu_activation(z)

        # Gating mechanism
        gate = torch.sigmoid(self.gate * z)

        tanh = gate * tanh_act
        gcu = (1 - gate) * gcu_act

        # Stack the activations for winner-take-all selection
        stacked_acts = torch.stack([tanh, gcu], dim=0)
        adaptive_acts, _ = torch.max(stacked_acts, dim=0)

        # Second layer
        z = self.layer2(adaptive_acts)
        gate = torch.sigmoid(self.gate * z)

        tanh = gate * tanh_act
        gcu = (1 - gate) * gcu_act

        # Stack the activations for winner-take-all selection
        stacked_acts = torch.stack([tanh, gcu], dim=0)
        adaptive_acts, _ = torch.max(stacked_acts, dim=0)

        # Pass the combined activations to the output layer
        return self.layer3(adaptive_acts)


# Custom activation functions
def gcu_activation(x):
    return x * torch.cos(x)
def enhanced_squ_activation(x, alpha=0.1):
    return alpha * (x ** 2) + x


# The AdaDQNall model
class AdaDQNall3(nn.Module):
    def __init__(self, num_nodes, n_observations, n_actions):
        super(AdaDQNall3, self).__init__()
        self.layer1 = nn.Linear(n_observations, num_nodes)
        self.layer2 = nn.Linear(num_nodes, num_nodes)
        self.layer3 = nn.Linear(num_nodes, n_actions)

        # Define activation functions
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.prelu = nn.PReLU(num_parameters=1, init=0.25)
        self.tanh = nn.Tanh()

        # Learnable weights for gating
        self.gate = nn.Parameter(torch.randn(num_nodes))

    def forward(self, x):
        z = self.layer1(x)

        # Apply all activation functions
        activations = [
            self.relu(z),
            self.elu(z),
            self.prelu(z),
            self.tanh(z),
            gcu_activation(z),
            enhanced_squ_activation(z)
        ]

        # Stack the activations for winner-take-all selection
        stacked_acts = torch.stack(activations, dim=0)
        adaptive_acts, _ = torch.max(stacked_acts, dim=0)

        # Second layer with winner-take-all activations from the first layer
        z = self.layer2(adaptive_acts)
        adaptive_acts, _ = torch.max(torch.stack([self.relu(z), self.elu(z)], dim=0), dim=0)

        # Output layer
        return self.layer3(adaptive_acts)


# === Experimental support functions ==================
# Function for selecting actions
def select_action(state, steps_done, policy_net):
    sample = random.random()
    eps_threshold = p_exper['eps_end'] + (p_exper['eps_start'] - p_exper['eps_end']) * \
                    math.exp(-1. * steps_done / p_exper['eps_decay'])
    # steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


# Function for selecting actions - for testing
def select_action_test(state, policy_net):
    with torch.no_grad():
        # t.max(1) will return the largest column value of each row.
        # second column on max result is index of where max element was
        # found, so we pick action with the larger expected reward.
        return policy_net(state).max(1)[1].view(1, 1)


episode_durations = []


# Function for plotting durations - mean and standard deviations for a condition
# mean_durat -> dim = (num_episodes,)
# std_durat -> dim = (num_episodes,)
# mean_rewards -> dim = (num_episodes,)
# std_rewards -> dim = (num_episodes,)
# a_title --> plot title
def plot_durations_4(mean_durat, std_durat, mean_aux_rewards, std_aux_rewards, performs, a_title):
    num_steps = mean_durat.shape[0]
    plt.title(a_title)
    plt.xlabel('Episode')
    plt.ylabel('Duration/Reward')
    xs = np.arange(0, num_steps)
    # Plot standard deviation regions
    plt.fill_between(xs, mean_durat - std_durat, mean_durat + std_durat, color='blue', alpha=0.2)
    plt.fill_between(xs, mean_aux_rewards - std_aux_rewards, mean_aux_rewards + std_aux_rewards, color='red', alpha=0.2)
    # Plot mean curve
    plt.plot(xs, mean_durat, color='blue', label='DR')
    plt.plot(xs, mean_aux_rewards, color='red', label='AR')
    plt.legend()
    # Plot extra performance metrics
    res_string = 'Solved at: {0}\n'.format(performs['solved_at'])
    res_string += 'M. cumul. DR: {0}\n'.format(performs['mean_cumul_dr'])  # durat. reward
    res_string += 'M. cumul. AR: {0}\n'.format(performs['mean_cumul_ar'])  # auxil. reward
    res_string += 'M. cumul. TR: {0}\n'.format(performs['mean_cumul_tr'])  # total reward
    res_string += 'M. std. DR: {0}\n'.format(np.round(performs['mean_std_dr'], 2))
    res_string += 'M. std. AR.: {0}\n'.format(np.round(performs['mean_std_ar'], 2))
    res_string += 'M. std. TR: {0}\n'.format(np.round(performs['mean_std_tr'], 2))
    x_pos = int(0.6 * num_steps)
    plt.text(x_pos, 30, res_string, color='black', fontsize=12)  # , ha='center')


# y-axis limits
# max_y = np.max(mean_durat + std_durat)
# plt.ylim(0, max_y)

def optimize_model(optimizer, memory, policy_net, target_net):
    if len(memory) < p_exper['batch_size']:
        return
    transitions = memory.sample(p_exper['batch_size'])
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(p_exper['batch_size'], device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * p_exper['gamma']) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


# === Main experimental functions =====================
def run_exper_trial(policy_net, target_net, p_exper):
    # Initializations
    steps_done = 0
    episode_durations = []
    episode_rewards = []
    episode_aux_rewards = []

    # Models
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.AdamW(policy_net.parameters(), lr=p_exper['rl_lr'], amsgrad=True)
    memory = ReplayMemory(10000)

    # Scan through episodes
    for i_episode in tqdm(range(p_exper['num_episodes'])):
        # Initialize the environment and get it's state
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        # Scan throgh time steps
        tot_reward = 0
        tot_ang_reward = 0
        for t in count():
            action = select_action(state, steps_done, policy_net)
            steps_done += 1
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            # Reward shaping - inversely prop. to vertical angle - cont. varying reward
            abs_ang = np.abs(observation[2])
            if abs_ang <= 0.2:
                ang_prop = 1 * (1 - (abs_ang / 0.2))
                tot_ang_reward += ang_prop
                reward += ang_prop
            tot_reward += reward

            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model(optimizer, memory, policy_net, target_net)

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            # Tomas note: this could be made more efficient
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * p_exper['tau'] + target_net_state_dict[
                    key] * (1 - p_exper['tau'])
            target_net.load_state_dict(target_net_state_dict)

            if t >= p_exper['max_ep_len'] or done:
                episode_durations.append(t + 1)
                episode_rewards.append(tot_reward.item())
                episode_aux_rewards.append(tot_ang_reward)
                # plot_durations_1()
                break

    return episode_durations, episode_rewards, episode_aux_rewards


def run_experiments():
    # p_exper['batch_size'] is the number of transitions sampled from the replay buffer
    # p_exper['gamma'] is the discount factor as mentioned in the previous section
    # p_exper['eps_start'] is the starting value of epsilon
    # p_exper['eps_end'] is the final value of epsilon
    # p_exper['eps_decay'] controls the rate of exponential decay of epsilon, higher means a slower decay
    # p_exper['tau'] is the update rate of the target network
    # LR is the learning rate of the ``AdamW`` optimizer

    # --- Shared parameters and initializations

    num_cond = 3  # number of conditions

    # Get number of actions from gym action space
    n_actions = env.action_space.n
    # Get the number of state observations
    state, info = env.reset()
    n_observations = len(state)
    all_durations = np.zeros((num_cond, p_exper['num_trials'], p_exper['num_episodes']))
    all_rewards = np.zeros((num_cond, p_exper['num_trials'], p_exper['num_episodes']))
    all_aux_rewards = np.zeros((num_cond, p_exper['num_trials'], p_exper['num_episodes']))
    cumul_durations = np.zeros((num_cond, p_exper['num_trials']))
    cumul_tot_rewards = np.zeros((num_cond, p_exper['num_trials']))
    cumul_aux_rewards = np.zeros((num_cond, p_exper['num_trials']))
    best_policy_nets = []  # store best policy networks for each condition

    # --- Condition 1 -------------------------

    # Scan through trials
    print('Condition 1 ...')
    cond_i = 0  # experimental condition index
    max_cumul_tr = 0  # largest cumulative total reward so far (across trials)
    for trial_i in range(p_exper['num_trials']):

        print('Trial {0} ...'.format(trial_i + 1))

        policy_net = AdaDQNall1(p_exper['num_nodes'], n_observations, n_actions).to(device)
        target_net = AdaDQNall1(p_exper['num_nodes'], n_observations, n_actions).to(device)
        episode_durations, episode_rewards, episode_aux_rewards = run_exper_trial(policy_net, target_net, p_exper)

        # Keep track of results
        all_durations[cond_i, trial_i, :] = episode_durations
        all_rewards[cond_i, trial_i, :] = episode_rewards
        all_aux_rewards[cond_i, trial_i, :] = episode_aux_rewards
        a_cumul_dr = np.sum(episode_durations)
        a_cumul_tr = np.sum(episode_rewards)
        a_cumul_ar = np.sum(episode_aux_rewards)
        cumul_durations[cond_i, trial_i] = a_cumul_dr
        cumul_tot_rewards[cond_i, trial_i] = a_cumul_tr
        cumul_aux_rewards[cond_i, trial_i] = a_cumul_ar

        # Keep track of the best model
        if a_cumul_tr > max_cumul_tr:
            # Keep track of best model
            max_cumul_tr = a_cumul_tr
            best_policy_net = policy_net.state_dict()

    best_policy_nets.append(best_policy_net)

    # --- Condition 2 -------------------------

    # Scan through trials
    print('Condition 2 ...')
    cond_i +=1  # experimental condition index
    max_cumul_tr = 0  # largest cumulative total reward so far (across trials)
    for trial_i in range(p_exper['num_trials']):

        print('Trial {0} ...'.format(trial_i + 1))

        policy_net = AdaDQNall2(p_exper['num_nodes'], n_observations, n_actions).to(device)
        target_net = AdaDQNall2(p_exper['num_nodes'], n_observations, n_actions).to(device)
        episode_durations, episode_rewards, episode_aux_rewards = run_exper_trial(policy_net, target_net, p_exper)

        # Keep track of results
        all_durations[cond_i, trial_i, :] = episode_durations
        all_rewards[cond_i, trial_i, :] = episode_rewards
        all_aux_rewards[cond_i, trial_i, :] = episode_aux_rewards
        a_cumul_dr = np.sum(episode_durations)
        a_cumul_tr = np.sum(episode_rewards)
        a_cumul_ar = np.sum(episode_aux_rewards)
        cumul_durations[cond_i, trial_i] = a_cumul_dr
        cumul_tot_rewards[cond_i, trial_i] = a_cumul_tr
        cumul_aux_rewards[cond_i, trial_i] = a_cumul_ar

        # Keep track of the best model
        if a_cumul_tr > max_cumul_tr:
            # Keep track of best model
            max_cumul_tr = a_cumul_tr
            best_policy_net = policy_net.state_dict()

    best_policy_nets.append(best_policy_net)


    # --- Condition 2 -------------------------

    # Scan through trials
    print('Condition 2 ...')
    cond_i += 1  # experimental condition index
    max_cumul_tr = 0  # largest cumulative total reward so far (across trials)
    for trial_i in range(p_exper['num_trials']):

        print('Trial {0} ...'.format(trial_i + 1))

        policy_net = AdaDQNall3(p_exper['num_nodes'], n_observations, n_actions).to(device)
        target_net = AdaDQNall3(p_exper['num_nodes'], n_observations, n_actions).to(device)
        episode_durations, episode_rewards, episode_aux_rewards = run_exper_trial(policy_net, target_net, p_exper)

        # Keep track of results
        all_durations[cond_i, trial_i, :] = episode_durations
        all_rewards[cond_i, trial_i, :] = episode_rewards
        all_aux_rewards[cond_i, trial_i, :] = episode_aux_rewards
        a_cumul_dr = np.sum(episode_durations)
        a_cumul_tr = np.sum(episode_rewards)
        a_cumul_ar = np.sum(episode_aux_rewards)
        cumul_durations[cond_i, trial_i] = a_cumul_dr
        cumul_tot_rewards[cond_i, trial_i] = a_cumul_tr
        cumul_aux_rewards[cond_i, trial_i] = a_cumul_ar

        # Keep track of the best model
        if a_cumul_tr > max_cumul_tr:
            # Keep track of best model
            max_cumul_tr = a_cumul_tr
            best_policy_net = policy_net.state_dict()

    best_policy_nets.append(best_policy_net)

    # === Prepare results for statistics and visualization

    mean_durations = all_durations.mean(axis=1)  # across trials
    std_durations = all_durations.std(axis=1)  # across trials
    mean_std_durations = std_durations.mean(axis=1)  # across episodes
    mean_rewards = all_rewards.mean(axis=1)  # across trials
    mean_aux_rewards = all_aux_rewards.mean(axis=1)  # across trials
    std_tot_rewards = all_rewards.std(axis=1)  # across trials
    std_aux_rewards = all_aux_rewards.std(axis=1)  # across trials
    mean_std_tot_rewards = std_tot_rewards.mean(axis=1)  # across episodes
    mean_std_aux_rewards = std_aux_rewards.mean(axis=1)  # across episodes
    mean_cumul_durat = cumul_durations.mean(axis=1)
    mean_cumul_tot_rewards = cumul_tot_rewards.mean(axis=1)
    mean_cumul_aux_rewards = cumul_aux_rewards.mean(axis=1)


    # === Visualize results

    # Plot graphs for different conditions
    # --- condition 1
    fig1 = plt.figure(1)
    ci = 0
    a_mean_cumul_durat = round(mean_cumul_durat[ci])
    a_mean_cumul_tot_rewards = round(mean_cumul_tot_rewards[ci])
    a_mean_cumul_aux_rewards = round(mean_cumul_aux_rewards[ci])
    performs = {}
    performs['solved_at'] = get_index_exceed(mean_durations[ci, :], p_exper['leng_solved'])
    performs['mean_std_dr'] = mean_std_durations[ci]
    performs['mean_std_tr'] = mean_std_tot_rewards[ci]
    performs['mean_std_ar'] = mean_std_aux_rewards[ci]
    performs['mean_cumul_dr'] = a_mean_cumul_durat
    performs['mean_cumul_tr'] = a_mean_cumul_tot_rewards
    performs['mean_cumul_ar'] = a_mean_cumul_aux_rewards
    a_title = 'Condition {0} - AdaHierarchicalStructure-PRELU&ELU'.format(ci + 1)
    plot_durations_4(mean_durations[ci, :], std_durations[ci, :], mean_aux_rewards[ci, :], std_aux_rewards[ci, :],
                     performs, a_title)
    print(mean_durations[ci, :], std_durations[ci, :], mean_aux_rewards[ci, :], std_aux_rewards[ci, :],
          performs, a_title)
    # --- condition 2
    fig2 = plt.figure(2)
    ci += 1
    a_mean_cumul_durat = round(mean_cumul_durat[ci])
    a_mean_cumul_tot_rewards = round(mean_cumul_tot_rewards[ci])
    a_mean_cumul_aux_rewards = round(mean_cumul_aux_rewards[ci])
    performs['solved_at'] = get_index_exceed(mean_durations[ci, :], p_exper['leng_solved'])
    performs['mean_std_dr'] = mean_std_durations[ci]
    performs['mean_std_tr'] = mean_std_tot_rewards[ci]
    performs['mean_std_ar'] = mean_std_aux_rewards[ci]
    performs['mean_cumul_dr'] = a_mean_cumul_durat
    performs['mean_cumul_tr'] = a_mean_cumul_tot_rewards
    performs['mean_cumul_ar'] = a_mean_cumul_aux_rewards
    a_title = 'Condition {0} - AdaHierarchicalStructure-Tanh&GCU'.format(ci + 1)
    plot_durations_4(mean_durations[ci, :], std_durations[ci, :], mean_aux_rewards[ci, :], std_aux_rewards[ci, :],
                     performs, a_title)

    # --- condition 3
    fig3 = plt.figure(3)
    ci += 1
    a_mean_cumul_durat = round(mean_cumul_durat[ci])
    a_mean_cumul_tot_rewards = round(mean_cumul_tot_rewards[ci])
    a_mean_cumul_aux_rewards = round(mean_cumul_aux_rewards[ci])
    performs['solved_at'] = get_index_exceed(mean_durations[ci, :], p_exper['leng_solved'])
    performs['mean_std_dr'] = mean_std_durations[ci]
    performs['mean_std_tr'] = mean_std_tot_rewards[ci]
    performs['mean_std_ar'] = mean_std_aux_rewards[ci]
    performs['mean_cumul_dr'] = a_mean_cumul_durat
    performs['mean_cumul_tr'] = a_mean_cumul_tot_rewards
    performs['mean_cumul_ar'] = a_mean_cumul_aux_rewards
    a_title = 'Condition {0} - AdaHierarchicalStructureAllAF'.format(ci + 1)
    plot_durations_4(mean_durations[ci, :], std_durations[ci, :], mean_aux_rewards[ci, :], std_aux_rewards[ci, :],
                     performs, a_title)

    plt.show()




    # Save models
    fn1 = 'c1-' + create_model_filename()
    torch.save(best_policy_nets[0], fn1)
    fn2 = 'c2-' + create_model_filename()
    torch.save(best_policy_nets[0], fn2)
    fn3 = 'c3' + create_model_filename()
    torch.save(best_policy_nets[0], fn3)


# === Main rendering test =======================
def render_test_net():
    env = gym.make("CartPole-v1", render_mode='human')

    # Get number of actions from gym action space
    n_actions = env.action_space.n
    # Get the number of state observations
    state, info = env.reset()
    n_observations = len(state)
    # Create network depending on condition
    if p_test['condition'] == 1:
        policy_net = AdaDQNall1(p_exper['num_nodes'], n_observations, n_actions)
    elif p_test['condition'] == 2:
        policy_net = AdaDQNall2(p_exper['num_nodes'], n_observations, n_actions)
    elif p_test['condition'] == 3:
        policy_net = AdaDQNall3(p_exper['num_nodes'], n_observations, n_actions)


    # Load model
    policy_net.load_state_dict(torch.load(p_test['net_name']))

    # Initialize the environment and get it's state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    # Scan throgh time steps
    # tot_reward = 0
    # tot_ang_reward = 0
    for t in count():
        action = select_action_test(state, policy_net)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Move to the next state
        state = next_state

        if t >= p_test['max_ep_len'] or done:
            break


# ==== Main ========================================

if __name__ == "__main__":
    if mode == 1:  # experimental model
        run_experiments()
    if mode == 2:  # test model with rendering
        render_test_net()  # test model via rendering
