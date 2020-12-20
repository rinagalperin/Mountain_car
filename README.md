# Reinforcement learning task: Mountain Car from openAI Gym

## :dart: Goal ([source](https://gym.openai.com/envs/MountainCar-v0/))
A car is on a one-dimensional track, positioned between two "mountains". The goal is to drive up the mountain on the right; however, the car's engine is not strong enough to scale the mountain in a single pass. Therefore, the only way to succeed is to drive back and forth to build up momentum.

## :bulb: Solution
We solve the Mountain Car problem using function approximations with gaussian radial basis functions. We also utilized the Sarsa lambda algorithm for the learning process.

## :clipboard: Code
At each run the code does the following: 

1. Computes the optimal policy using function approximations with gaussian radial basis functions, where the learner is based on the Sarsa lambda algorithm (via the ["learn" function](sarsa_lambda.py) to train the agent)
2. Displays a simulation of the best policy
3. Plots result of the learning progress: x axis represents the steps’ count, y axis represents the policy value.

## :email: Contact
- rinag@post.bgu.ac.il
- schnapp@post.bgu.ac.il
