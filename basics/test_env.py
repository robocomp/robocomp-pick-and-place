from EnvKinova_gym import EnvKinova_gym


env = EnvKinova_gym()
# print(env.observation_space)

print("BEGIN RESET")
obs = env.reset()

for k, v in obs.items():
    print(f"{k}: {v}")
print("END RESET")
# action = env.action_space.sample()
# next_state, reward, done, info = env.step(action)
env.close()