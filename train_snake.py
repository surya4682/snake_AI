from stable_baselines3 import PPO
from snake_env import SnakeEnv

def main():
    env = SnakeEnv()
    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./snake_ppo_tensorboard/")
    obs = env.reset()
    for _ in range(20000):  # Adjust the number as needed
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()  # Update the rendering on each step
    model.learn(total_timesteps=20000)
    model.save("ppo_snake")
    print("Training completed!")

if __name__ == "__main__":
    main()
