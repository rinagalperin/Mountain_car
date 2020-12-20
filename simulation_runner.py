import gym


class SimulationRunner:
    def __init__(self):
        self.problem = gym.make('MountainCar-v0')

    def run_simulation(self, agent):
        print('---- start simulation with {} ----'.format(agent.name))
        env = self.problem.env
        t = 0
        s_t = self.problem.reset()
        done = False
        score = 0  # reward sum
        try:
            env.render()
        except:
            pass

        while not done and t < 500:
            a_t = agent.get_action(s_t)
            s_t, r_t, done, _ = env.step(a_t)
            score += r_t
            t += 1
            env.render()
        env.close()
        return score
