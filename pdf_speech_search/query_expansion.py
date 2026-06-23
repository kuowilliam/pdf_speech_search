from __future__ import annotations

import re


TERM_EXPANSIONS: dict[str, list[str]] = {
    "a2c": ["advantage actor critic", "policy gradient", "value function"],
    "a3c": ["asynchronous advantage actor critic", "policy gradient", "value function"],
    "backprop": [
        "backpropagation",
        "back propagation",
        "backward pass",
        "chain rule",
        "reverse mode automatic differentiation",
        "gradient calculation",
    ],
    "backpropagation": [
        "back propagation",
        "backward pass",
        "chain rule",
        "reverse mode automatic differentiation",
        "gradient calculation",
    ],
    "cnn": ["convolutional neural network", "convolution", "filters", "feature maps"],
    "dqn": [
        "deep q network",
        "deep q-network",
        "q learning",
        "q-learning",
        "action value function",
        "bellman equation",
        "experience replay",
        "target network",
        "epsilon greedy",
        "temporal difference",
    ],
    "gan": ["generative adversarial network", "generator", "discriminator", "minimax"],
    "gmm": ["gaussian mixture model", "expectation maximization"],
    "hmm": ["hidden markov model", "transition probability", "emission probability"],
    "lstm": ["long short term memory", "gates", "recurrent neural network"],
    "mdp": ["markov decision process", "states actions rewards transition policy"],
    "pca": ["principal component analysis", "eigenvectors", "variance"],
    "ppo": ["proximal policy optimization", "policy gradient", "clipped objective"],
    "rl": ["reinforcement learning", "agent environment reward policy value function"],
    "rnn": ["recurrent neural network", "sequence model", "hidden state"],
    "sgd": ["stochastic gradient descent", "learning rate", "optimization"],
    "svm": ["support vector machine", "margin", "kernel"],
    "vae": ["variational autoencoder", "encoder decoder latent variable"],
}


def expand_query(query: str) -> str:
    normalized = query.strip()
    if not normalized:
        return ""

    additions: list[str] = []
    lowered = normalized.lower()
    for term, expansions in TERM_EXPANSIONS.items():
        if re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", lowered):
            additions.extend(expansions)

    if "q network" in lowered or "q-network" in lowered:
        additions.extend(TERM_EXPANSIONS["dqn"])

    if "back propagation" in lowered or "back propagations" in lowered:
        additions.extend(TERM_EXPANSIONS["backpropagation"])

    if not additions:
        return normalized

    return normalized + "\n" + " ".join(dict.fromkeys(additions))
