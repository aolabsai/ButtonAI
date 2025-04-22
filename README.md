# ao_loop1
Open source reference design for implementing Loop-1, an API for real-time reinforcement learning

Maintainer: aolabsai, eng@aolabs.ai

This is a reference implementation showing how to use an AO weightless neural network agent (`ao_pyth`) for real-time inference and learning over user-specified tasks.

This repo includes a skeleton that can be easily modified to a wide range of classification problem—just transform your input features to binary form and feed them into the agent. It uses our python wrapper, [ao_pyth](https://pypi.org/project/ao-pyth/); for other languages, visit our [API reference and CURL playground](https://docs.aolabs.ai/reference/agentapi).

---

## What Does It Do?

This repo shows how to:
- Load or define input features (any data type—text, numbers, images, etc.)
- Convert features into binary vectors
- Feed binary vectors into an AO agent
- Receive a binary prediction as output
- Provide real-time feedback to re-train the agent on-the-fly

You can customize the feature extraction step to suit your task (e.g., extracting features from text, images, or structured data), and the agent will learn from user input using real-time **reinforcement learning**.

---

## Key Concepts

- **Binary Inputs Only:** AO agents only accept binary input vectors (e.g., `[0, 1, 1, 0, ...]`). You must convert all input features into binary format before inference.
- **Binary Output:** The agent outputs a binary vector, typically representing class predictions or a percentage scale.
- **Real-Time Enforcement Learning:** After each prediction, you can provide feedback (correct/incorrect) to instantly update the agent's internal state.
- **No Backpropagation:** AO agents are weightless and learn without gradient descent. They are interpretable and can adapt quickly with minimal data.

---

## Setup Instructions

### Requirements

- Python 3.8+
- `ao_pyth`
- `numpy`
- `requests`
- `ast`, `openai` (if using LLMs for feature extraction)


