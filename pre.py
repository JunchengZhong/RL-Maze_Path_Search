import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def cut_maze_matrix_up(matrix):
    for i,row in enumerate(matrix):
        if np.any(row - 1):
            matrix_up = matrix[i:]
            break
    return matrix_up

def cut_maze_matrix_arround(matrix):
    matrix_up = cut_maze_matrix_up(matrix)
    matrix_up_down = cut_maze_matrix_up(matrix_up[::-1])
    matrix_up_down_left = cut_maze_matrix_up(matrix_up_down.T)
    matrix_up_down_left_right = cut_maze_matrix_up(matrix_up_down_left[::-1])
    return matrix_up_down_left_right[::-1].T[::-1]

# 0是黑的，1是白的

def pixel_to_block(matrix,scale = None):
    if scale is None:
        temp = (matrix[0,:]+ matrix[1,:]).tolist()
        index = temp.index(2)
        scale = index-2
    m,n = matrix.shape
    m,n = int(m/scale), int(n/scale)
    matrix_block = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            if np.all(matrix[scale*i:scale*(i+1),scale*j:scale*(j+1)]) == 1:
                matrix_block[i,j] = 1
    return matrix_block



if __name__ == '__main__':
    image_path = "maze.jpg"
    maze_image = Image.open(image_path).convert("L")
    threshold = 128
    binary_maze = np.array(maze_image) < threshold  # 黑色为 True（墙），白色为 False（通路）
    maze_matrix = np.where(binary_maze, 1, 0)  # 1 表示墙，0 表示通路
    maze_matrix_cut = cut_maze_matrix_arround(matrix = maze_matrix)
    maze_matrix_block = pixel_to_block(maze_matrix_cut)
    maze_matrix_block[-5,-2] = 0.5
    plt.imshow(maze_matrix_block,cmap="gray")
    plt.axis("off")
    plt.show()