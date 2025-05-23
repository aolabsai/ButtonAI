# Python standard libraries
import ast

# Third-party libraries
import requests
import numpy as np
from openai import OpenAI

# AO library
# import ao_pyth as ao # $ pip install ao_pyth - https://pypi.org/project/ao-pyth/
import ao_core as ao # private package, to run our code locally, useful for advanced debugging; ao_pyth is enough for most use cases


# Importing API keys
# from config import ao_apikey, openai_apikey, google_apikey 







                                    # ----------- Initialize AO Agent -----------#

# Initialize AO agent architecture, here with 30 input neurons and 5 output neurons. 
# Input consists of 3 features, each given on a intensity (or other) scale of 0-10 (10 neurons for each feature):
# Output consists of 5 neurons corresponding to a single scale of 1-5 (or whatever output(s) you want to associate with input).

# arch = ao.Arch(arch_i="[20, 7, 7, 21, 10, 9]", arch_z="[6]", api_key=ao_apikey, kennel_id="buttonAI_demo_01") # --> architecture setup
arch = ao.Arch(arch_i=[20, 7, 7, 21, 10, 9], arch_z=[6], arch_c=[], connector_function="full_conn", description="buttonAI_demo") # --> architecture setup
agent = ao.Agent(arch, save_meta=True)  # --> agent creation


# agent.api_reset_compatibility = True


                                    # ----------- Pre-train with Baseline Examples -----------#

# Optional - Use this to train the agent on a baseline (if the agent has no prior training, it would output random;
# if it only has 1 label/training event, it can only ever output that until trained on more examples)

training_data = [
    (np.random.randint(0, 2, size=74), np.random.randint(0, 2, size=6)),
    (np.random.randint(0, 2, size=74), np.random.randint(0, 2, size=6)),
    (np.random.randint(0, 2, size=74), np.random.randint(0, 2, size=6)),
    (np.random.randint(0, 2, size=74), np.random.randint(0, 2, size=6)),
    (np.random.randint(0, 2, size=74), np.random.randint(0, 2, size=6)),
]


# Loop through each training pair and feed it to the agent
for inp, label in training_data:
    print("Training Input:", inp)
    print("Training Label:", label)
    # Train the agent on this example
    agent.next_state(INPUT=inp, LABEL=label, unsequenced=True)  # unsequenced is default; you can set it to `False` to run on data that is sequential





                                    # ----------- Inference on Content -----------#

# This section tests how the agent responds to new, unseen data after pre-training.

testing_data = [
    np.random.randint(0, 2, size=74),  # Test Input 1
    np.random.randint(0, 2, size=74),  # Test Input 2
]

# Loop through each test input and get the agent’s response
for inp in testing_data:
    response = agent.next_state(INPUT=inp, unsequenced=True)
    print("Test Input:", inp)
    print("Agent Response:", response)




                                    # ----------- Feedback Loop -----------#

# Picking the best input pattern based on some external evaluation (e.g., real-world performance)
best_index = 1  # Python index of the best performing instance (you can modify this based on evaluation)
input_to_agent = testing_data[best_index][0]  # Input pattern from testing data

# Feedback label - suppose the system determined the correct output (ground truth) to reinforce
feedback_label = np.random.randint(0, 2, size=6)  # Replace with actual label
# Closing the Learning Loop - train agent with correct output on this input
agent.next_state(INPUT=input_to_agent, LABEL=feedback_label, unsequenced=True)

# Re-evaluate After Feedback to check if agent learned from the feedback
agent_response = agent.next_state(INPUT=input_to_agent, unsequenced=True)
print("Agent raw binary response: ", agent_response)




                                      # ----------- Additional Test Input -----------#

# Manually test the agent on a specific input (e.g., a new ad variant) to observe its response.

# Select an "ad" input from your extracted feature list
input_to_agent = np.random.randint(0, 2, size=74)  # replace it with actual input

# Run inference on the selected ad
agent_response = agent.next_state(INPUT=input_to_agent, unsequenced=True)
print("Agent Response to new ad feature:", agent_response)