import random
import numpy as np
import pre
from PIL import Image
import matplotlib.pyplot as plt
import os


class Maze:
    def __init__(self, maze, start_position, final_position):
        self.maze = maze
        self.rows = maze.shape[0]
        self.cols = maze.shape[1]
        self.n_states = self.rows * self.cols
        self.n_actions = 4 # up down left right
        self.start_position = start_position
        self.final_position = final_position
    
    def __getitem__(self, index):
        return self.maze[index]

    def get_state(self, position):
        return position[0] * self.cols + position[1]
    
    def get_position(self, state):
        return (state // self.cols, state % self.cols)
    
    def is_valid_position(self, position): # 1 is the way and 0 is the wall
        return (0 <= position[0] < self.rows) and (0 <= position[1] < self.cols)
    
    def get_new_state(self, state, action):
        position = self.get_position(state)
        if action == 0: # up
            new_position = (position[0] - 1, position[1])
        elif action == 1: # down
            new_position = (position[0] + 1, position[1])
        elif action == 2: # left
            new_position = (position[0], position[1] - 1)
        else: # right
            new_position = (position[0], position[1] + 1)
    
        if self.is_valid_position(new_position):
            return self.get_state(new_position)
        else:
            return state
        
    def get_reward(self, state):
        position = self.get_position(state)
        x, y = position
        bound = int(self.n_states / 2) # estimation of path
        if not self.is_valid_position(position):
            return -1000
        if position == self.final_position:
            return 1000
        elif self.maze[x][y] == 0:
            return -bound
        else:
            return -1


def initialize_q_table(n_states, n_actions):
    return np.zeros((n_states, n_actions))

def choose_action_index(state, q_table, epsilon):
    if random.uniform(0,1) < epsilon:
        return random.randint(0, 3)
    else:
        return np.argmax(q_table[state, :])
    
def update_q_table(q_table, state, new_state, action_index, alpha, reward, gamma):
    best_new_action_index = np.argmax(q_table[new_state, :])
    q_table[state, action_index] += alpha * (reward + gamma * q_table[new_state, best_new_action_index] - q_table[state, action_index])

def train_loop(maze, episodes, alpha, gamma, epsilon, min_epsilon = 0.01, decay_rate = 0.995):
    q_table = initialize_q_table(maze.n_states, maze.n_actions)
    for _ in range(episodes):
        state = maze.get_state(maze.start_position)
        steps = 0
        while state != maze.get_state(maze.final_position):
            action_index = choose_action_index(state, q_table, epsilon)
            new_state = maze.get_new_state(state, action_index)
            reward = maze.get_reward(new_state)
            update_q_table(q_table, state, new_state, action_index, alpha, reward, gamma)
            state = new_state
            steps += 1
            if steps > maze.n_states * 2:
                break
        epsilon = max(min_epsilon, epsilon * decay_rate)
    return q_table

def find_path(maze, q_table):
    path = [maze.start_position]
    state = maze.get_state(maze.start_position)
    steps = 0
    while state != maze.get_state(maze.final_position):
        action_index = np.argmax(q_table[state, :])
        new_state = maze.get_new_state(state, action_index)
        new_position = maze.get_position(new_state)
        if new_state == state or maze[new_position[0]][new_position[1]] == 0:
            return None
        else:
            path.append(maze.get_position(new_state))
        state = new_state
        steps += 1
        if steps > maze.n_states * 2:
            return None
    return path

def display_maze_with_path(maze_matrix, path, start_point, final_point):
    fig, ax = plt.subplots(figsize=(8, 8))
    # Show the maze image, set origin = 'upper', makes sure (0,0) is in the upper left corner.
    ax.imshow(maze_matrix, cmap='gray', origin='upper')
    
    # Mark the starting point.
    ax.scatter(start_point[1], start_point[0], marker='o', color='green', s=100, label='Start')
    
    # Mark the final point.
    if final_point:
        ax.scatter(final_point[1], final_point[0], marker='x', color='red', s=100, label='End')
    
    # Draw paths.
    if path:
        path_y = [pos[1] for pos in path]
        path_x = [pos[0] for pos in path]
        ax.plot(path_y, path_x, color='blue', linewidth=2, label='Path')
    
    # Add legends.
    ax.legend(loc='upper right')
    
    # Hide axis scale.
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.title("Maze Path")
    plt.show()

def on_click(event, maze_matrix_block, maze, start_point):
    if event.inaxes:
        # Get the coordinates of the mouse click.
        x, y = int(round(event.ydata)), int(round(event.xdata))
        print(f"Clicked coordinates: ({x}, {y})")

        # Check to see if the click position is within the maze.
        if not maze.is_valid_position((x, y)):
            print("点击位置超出迷宫范围，请重新选择。")
            return

        # Check if the click position is a wall.
        if maze_matrix_block[x, y] == 0:
            print("点击位置是墙，请选择一个通路位置。")
            return

        final_point = (x, y)
        print(f"选择的终点位置: {final_point}")

        # Update the final point.
        maze.final_position = final_point

        # Retrain Q-table.
        print("正在训练代理，请稍候...")
        episodes = 20000
        alpha = 0.1
        gamma = 0.98
        epsilon = 1
        q_table = train_loop(maze, episodes, alpha, gamma, epsilon)
        print("训练完成。")

        # Find paths.
        path = find_path(maze, q_table)
        if path:
            print("找到最短路径：", path)
        else:
            print("无法到达终点。")

        # Show the maze and paths.
        display_maze_with_path(maze_matrix_block, path, start_point, final_point)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, "maze.jpg")
    maze_image = Image.open(image_path).convert("L")
    threshold = 128
    binary_maze = np.array(maze_image) < threshold  
    maze_matrix = np.where(binary_maze, 1, 0)  
    maze_matrix_cut = pre.cut_maze_matrix_arround(matrix=maze_matrix)
    maze_matrix_block = pre.pixel_to_block(maze_matrix_cut)
    rows, cols = maze_matrix_block.shape
    rows, cols = int(rows), int(cols)
    maze_matrix_block[44, 63] = 1  # start point
    # maze_matrix_block[45, 63] = 0  if we want to test the case that fails to search the paths.
    maze = maze_matrix_block
    start_point = (44, 63)
    final_point = (21, 43) #initial test

    print(maze)
    # Ensuring effective start and final points
    if not (0 <= start_point[0] < rows and 0 <= start_point[1] < cols):
        raise ValueError("起点位置无效！")
    if not (0 <= final_point[0] < rows and 0 <= final_point[1] < cols):
        raise ValueError("终点位置无效！")
    if maze[start_point[0], start_point[1]] == 0:
        raise ValueError("起点位置是墙！")
    if maze[final_point[0], final_point[1]] == 0:
        raise ValueError("终点位置是墙！")

    # # Initialize Maze
    maze_obj = Maze(maze, start_position=start_point, final_position=final_point)

    # # Initial training
    print("初始训练代理，请稍候...")
    episodes = 10000
    alpha = 0.1
    gamma = 0.95
    epsilon = 0.1
    q_table = train_loop(maze_obj, episodes, alpha, gamma, epsilon)
    print("初始训练完成。")

    # Find the initial paths
    path = find_path(maze_obj, q_table)
    if path:
        print("找到最短路径：", path)
    else:
        print("无法到达初始终点。")

    # Display initial path
    display_maze_with_path(maze_matrix_block, path, start_point, final_point)

    # Set up Matplotlib Graphics for interaction
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(maze_matrix_block, cmap='gray', origin='upper')
    ax.scatter(start_point[1], start_point[0], marker='o', color='green', s=100, label='Start')
    ax.scatter(final_point[1], final_point[0], marker='x', color='red', s=100, label='Initial End')
    ax.legend(loc='upper right')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.title("Click to select a new final location")

    cid = fig.canvas.mpl_connect('button_press_event', lambda event: on_click(event, maze_matrix_block, maze_obj, start_point))

    plt.show()

if __name__ == "__main__":
    main()