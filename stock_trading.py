import sys
from stock_trading_sim import StockTradingSimEnvironment
from stock_trading_agent import StockTradingAgent
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import time
from sklearn.model_selection import train_test_split

def print_progress(current, total, label="", char_space=50):
    # Saving space for the start and end bars
    char_space -= 2

    # Calculating the completed characters in the progress bar
    completed_char = int((current/total) * char_space) if current != 0 else 0
    missing_char = char_space - completed_char
    progress_string = f"\r[{'='*completed_char}{' '*missing_char}] {current}/{total} {label}"

    # Filling the completed chars
    sys.stdout.write(progress_string)
    sys.stdout.flush()

def gather_samples(episodes):
    print(f"Collecting samples for {episodes} episodes")
    samples = []
    for episode in range(episodes):
        # Print the progress
        print_progress(episode+1, episodes, label="Episodes")

        state = env.reset()
        samples.append(state)
        done = False

        while not done:
            # Getting a random action in the action space and playing
            action = np.random.choice(env.action_space)
            next_state, reward, done, _ = env.step(action)

            # Shifting states in time and saving it
            state = next_state
            samples.append(state)

    # After all the episodes are finished, return the samples
    print(f"\n{len(samples)} samples collected...")
    return samples

def epsilon_greedy_action(state, epsilon):
    # Exploration
    if np.random.random() < epsilon:
        action = np.random.choice(env.action_space)
    # Exploitation
    else:
        action = agent.get_action(state)
    return action


def play_one_episode(train_mode, epsilon):
    state = env.reset()
    done = False

    while not done:
        # Print the progress
        print_progress(env.pointer+1, env.n_steps, label="Steps")

        # Getting epsilon-greedy action and playing it
        action = epsilon_greedy_action(state, epsilon)
        next_state, reward, done, info = env.step(action)

        # If training then update agent
        if train_mode:
            agent.train(state, action, next_state, reward, done)

        # Shifting states in time
        state = next_state

    # Returning the portfolio at the end of the episode
    return info["portfolio_value"]


if __name__ == "__main__":
    # Reading and splitting train and test data
    data = pd.read_csv("aapl_msi_sbux.csv")
    train_data, test_data = train_test_split(data, test_size=0.4, random_state=101)

    # Defining our starting budget
    budget = 10000

    # Creating out environment and collect samples for the agent
    env = StockTradingSimEnvironment(train_data, budget)

    sample_episodes = 10
    agent = StockTradingAgent(env.state_size, env.action_space, gather_samples(sample_episodes))

    # Then training for 10 episodes
    portfolio_per_episode = []

    episodes_to_train = 5
    epsilon = 0.9

    for episode in range(episodes_to_train):
        # To measure episode time
        start_time = datetime.now()
        final_portfolio = play_one_episode(train_mode=True, epsilon=epsilon)
        portfolio_per_episode.append(final_portfolio)
        
        # Getting the time after the episode and the printing status
        end_time = datetime.now()
        print(f"\nEpisode: {episode+1}/{episodes_to_train}, Final value: {final_portfolio}, Duration: {end_time - start_time}")

    # Plotting the portfolio over the episodes
    plt.plot(portfolio_per_episode)
    plt.grid()
    plt.title("Portfolio per episode")
    plt.xlabel("Episode")
    plt.ylabel("Portfolio")
    plt.show()
    