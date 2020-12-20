from sarsa_lambda import SarsaLambda
from simulation_runner import SimulationRunner
import matplotlib.pyplot as plt

sarsa_lambda = SarsaLambda(alpha=0.02, lambda_p=0.5)
x, y = sarsa_lambda.learn()

SimulationRunner().run_simulation(sarsa_lambda)

plt.plot(x, y)
plt.xlabel("Steps Count")
plt.ylabel("Policy Value")
plt.legend()

plt.show()
