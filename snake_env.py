import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import random
import cv2  # Ensure OpenCV is installed with `pip install opencv-python`

class SnakeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(SnakeEnv, self).__init__()
        self.action_space = spaces.Discrete(4)  # 0: Up, 1: Down, 2: Left, 3: Right
        self.observation_space = spaces.Box(low=0, high=255, shape=(1, 120, 160), dtype=np.uint8)  # Channel-first format

        # Define constants
        self.display_width = 800
        self.display_height = 600
        self.snake_block = 10
        self.snake_speed = 15

        # Initialize pygame
        pygame.init()
        self.dis = pygame.display.set_mode((self.display_width, self.display_height))
        pygame.display.set_caption('Snake Game Environment')

        self.clock = pygame.time.Clock()

        # Initialize game state
        self.reset()

    def step(self, action):
        if action == 0:  # Up
            self.y1_change = -self.snake_block
            self.x1_change = 0
        elif action == 1:  # Down
            self.y1_change = self.snake_block
            self.x1_change = 0
        elif action == 2:  # Left
            self.x1_change = -self.snake_block
            self.y1_change = 0
        elif action == 3:  # Right
            self.x1_change = self.snake_block
            self.y1_change = 0

        self.x1 += self.x1_change
        self.y1 += self.y1_change

        if self.x1 >= self.display_width or self.x1 < 0 or self.y1 >= self.display_height or self.y1 < 0:
            return self.get_state(), -10, True, {}

        self.snake_list.append([self.x1, self.y1])
        if len(self.snake_list) > self.length_of_snake:
            del self.snake_list[0]

        reward = 0
        if self.x1 == self.foodx and self.y1 == self.foody:
            self.foodx = round(random.randrange(0, self.display_width - self.snake_block) / 10.0) * 10.0
            self.foody = round(random.randrange(0, self.display_height - self.snake_block) / 10.0) * 10.0
            self.length_of_snake += 1
            reward = 10

        return self.get_state(), reward, False, {}

    def reset(self):
        self.x1 = self.display_width / 2
        self.y1 = self.display_height / 2
        self.x1_change = 0
        self.y1_change = 0
        self.snake_list = []
        self.length_of_snake = 1

        self.foodx = round(random.randrange(0, self.display_width - self.snake_block) / 10.0) * 10.0
        self.foody = round(random.randrange(0, self.display_height - self.snake_block) / 10.0) * 10.0

        return self.get_state()

    def get_state(self):
        self.dis.fill((0, 0, 0))  # Clear screen
        for x in self.snake_list:
            pygame.draw.rect(self.dis, (0, 255, 0), [x[0], x[1], self.snake_block, self.snake_block])
        pygame.draw.rect(self.dis, (255, 0, 0), [self.foodx, self.foody, self.snake_block, self.snake_block])

        frame = pygame.surfarray.array3d(pygame.display.get_surface())
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
        resized_frame = cv2.resize(frame, (160, 120), interpolation=cv2.INTER_AREA)
        resized_frame = np.expand_dims(resized_frame, axis=0)  # Add channel dimension
        return np.array(resized_frame, dtype=np.uint8)

    def render(self, mode='human'):
        pygame.display.update()
        self.clock.tick(self.snake_speed)

    def close(self):
        pygame.quit()
