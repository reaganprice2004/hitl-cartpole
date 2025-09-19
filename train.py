"""
train.py - Training loop that trains DQNAgent on CartPole-v1 while ingesting human corrections
from human_corrections.csv (if present). Human transitions are added into the replay buffer and
prioritized by scaling their loss (see agent.train_step's human_priority argument).
"""

# imports
import gymnasium as gym                # environment toolkit
import numpy as np                     # numerical arrays
import time                             # timing utilities
import csv                              # reading human corrections
import os                               # file checks
from agent import DQNAgent             # our agent implementation

# path to human corrections CSV produced by human_recorder.py
HUMAN_CORRECTIONS_FILE = "human_corrections.csv"

def load_human_corrections(path):
    # safely read CSV and convert rows into transitions similar to agent storage format
    transitions = []
    if not os.path.exists(path):
        return transitions
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # parse state and next_state from string representation (list-like)
            try:
                s = np.array(eval(row["state"]), dtype=np.float32)
                a = int(row["action"])
                s2 = np.array(eval(row["next_state"]), dtype=np.float32)
                r = float(row["reward"])
                done = row["done"].lower() in ("true", "1", "t")
                # mark as human correction with is_human=True
                transitions.append((s, a, r, s2, done, True))
            except Exception as e:
                # skip malformed lines gracefully
                print("Skipping malformed human correction line:", e)
    return transitions

def main():
    # create environment and agent
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]    # state dimensionality
    act_dim = env.action_space.n                # number of discrete actions
    agent = DQNAgent(state_dim=obs_dim, action_dim=act_dim)
    # training hyperparameters
    episodes = 500                              # number of training episodes
    max_steps = 500                             # max steps per episode
    epsilon = 1.0                               # start epsilon for exploration
    eps_decay = 0.995                           # decay per episode
    min_epsilon = 0.02                          # final epsilon
    human_ingest_interval = 5                   # ingest human file every N episodes
    # training loop over episodes
    for ep in range(1, episodes+1):
        state, info = env.reset()               # reset env at episode start
        ep_reward = 0.0                         # cumulative reward for logging
        done = False
        for t in range(max_steps):
            # select action via epsilon-greedy policy
            action = agent.select_action(state, epsilon)
            # perform environment step with chosen action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            # store transition as agent-collected (is_human=False)
            agent.store_transition(state, action, reward, next_state, done, is_human=False)
            # run a training step (if buffer large enough)
            loss = agent.train_step(target_update_freq=500, human_priority=4.0)
            # accumulate reward and advance state
            ep_reward += reward
            state = next_state
            if done:
                break
        # end of episode: optionally ingest human corrections file
        if ep % human_ingest_interval == 0:
            human_trans = load_human_corrections(HUMAN_CORRECTIONS_FILE)
            # add human transitions into replay buffer (might duplicate if already present)
            for tr in human_trans:
                agent.replay_buffer.add(tr)
            print(f"Episode {ep}: ingested {len(human_trans)} human transitions (if any).")
        # decay epsilon for exploration-exploitation tradeoff
        epsilon = max(min_epsilon, epsilon * eps_decay)
        # logging
        print(f"Episode {ep} finished - Reward: {ep_reward:.2f} - Epsilon: {epsilon:.3f} - Buffer size: {len(agent.replay_buffer)}")
    # save final model weights
    import torch
    torch.save(agent.q_net.state_dict(), "dqn_agent_final.pth")
    print("Training complete. Saved dqn_agent_final.pth")

if __name__ == "__main__":
    main()
