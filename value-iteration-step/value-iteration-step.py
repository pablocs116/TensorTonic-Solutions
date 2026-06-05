import numpy as np

def value_iteration_step(values, transitions, rewards, gamma):
    new_values = []
    values = np.array(values)

    for state_transitions, state_rewards in zip(transitions, rewards):
        q_values = [
            reward + gamma * np.dot(trans_prob, values)
            for reward, trans_prob in zip(state_rewards, state_transitions)
        ]
        new_values.append(max(q_values))

    return new_values