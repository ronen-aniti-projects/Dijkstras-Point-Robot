from enum import Enum 
import numpy as np 
import matplotlib.pyplot as plt
from queue import PriorityQueue 
from matplotlib.animation import FuncAnimation
import time

import pdb

class Action(Enum):
    """
    This is the implementation of the action set. Each action is basically a translation vector, having a component-wise `delta` representation and a magnitude or `cost`. 
    """
    def __init__(self, delta, cost):
        self.delta = delta
        self.cost = cost
    UP = ((1, 0), 1)
    DOWN = ((-1, 0), 1)
    LEFT = ((0, -1), 1)
    RIGHT = ((0, 1), 1)
    UP_RIGHT = ((1, 1), np.sqrt(2)) 
    UP_LEFT = ((1, -1), np.sqrt(2))
    DOWN_RIGHT = ((-1, 1), np.sqrt(2))
    DOWN_LEFT = ((-1, -1), np.sqrt(2))


def in_bounds(x, y):
    return (0 <= x <= 180 and 0 <= y <= 50)

def collision(x, y, safety=1.0):
    """
    This function handles collision avoidance. It returns true if the query point colides with any of a long list of primitive regions. The regions are defined as per the geometric specifications provided for this project. 
    Note: All regions are inflated by `safety` units in unit normal direction. 
    """
    regions = []

    # E
    ##########################
    # E, vertical rectangle primitive
    regions.append(
        (x - 20 + safety >= 0) and
        (x - 25 - safety <= 0) and
        (y - 10 + safety >= 0) and   
        (y - 35 - safety <= 0)
    )
    
    
    # E, bottom rectangle primitive
    regions.append(
        (x - 20 + safety >= 0) and
        (x - 33 - safety <= 0) and
        (y - 10 + safety >= 0) and 
        (y - 15 - safety <= 0)

    )
    
    # E, middle rectangle primitive
    regions.append(
        (x - 20 + safety >= 0) and
        (x - 33 - safety <= 0) and
        (y - 20 + safety >= 0) and 
        (y - 25 - safety <= 0)

    )
    
    # E, top rectangle primitive
    regions.append(
        (x - 20 + safety >= 0) and
        (x - 33 - safety <= 0) and
        (y - 30 + safety >= 0) and 
        (y - 35 - safety <= 0)

    )
    
    # N 
    ##########################
    # N, left vertical rectangle primitive
    regions.append(
        (x - 43 + safety >= 0) and
        (x - 48 - safety <= 0) and
        (y - 10 + safety >= 0) and   
        (y - 35 - safety <= 0)      
    )
    
    # N, middle in between two segment region
    regions.append(
        (y + 3*x - 179 - safety * np.sqrt(10) <= 0) and
        (y + 3*x - 169 + safety * np.sqrt(10) >= 0) and 
        (x - 48 + safety >= 0) and
        (x - 53 - safety <= 0) and 
        (y - 10 + safety >= 0) and 
        (y - 35 - safety <= 0)      

    )

    # N, right vertical rectangle primitive
    regions.append(
        (x - 53 + safety >= 0) and
        (x - 58 - safety <= 0) and
        (y - 10 + safety >= 0) and   
        (y - 35 - safety <= 0)      
    )

    # P
    ###############
    # P, left vertical bar
    regions.append(
        (x - 68 + safety >= 0) and
        (x - 73 - safety <= 0) and
        (y - 10 + safety >= 0) and   
        (y - 35 - safety <= 0)      
    )

    # P, semi-circular region
    regions.append(
        ( (x - 73)**2 + (y - 28.75 )**2 - (6.25 + safety)**2 <= 0) and
        (x - 73 - safety >= 0)
    )
    
     # M
    ##################
    # M, first vertical bar (left vertical)
    regions.append(
        (x - 85 + safety >= 0) and
        (x - 90 - safety <= 0) and
        (y - 10 + safety >= 0) and   
        (y - 35 - safety <= 0)
    )
    # M, left diagonal region (from (90,35) to (95,10))
    regions.append(
        (5*x + y - 485 - safety * np.sqrt(26) <= 0) and
        (5*x + y - 485 + safety * np.sqrt(26) >= 0) and
        (x - 90 + safety >= 0) and (x - 95 - safety <= 0) and
        (y - 10 + safety >= 0) and (y - 35 - safety <= 0)
    )
    # M, right diagonal region (from (95,10) to (100,35))
    regions.append(
        (5*x - y - 465 - safety * np.sqrt(26) <= 0) and
        (5*x - y - 465 + safety * np.sqrt(26) >= 0) and
        (x - 95 + safety >= 0) and (x - 100 - safety <= 0) and
        (y - 10 + safety >= 0) and (y - 35 - safety <= 0)
    )
    # M, second vertical bar (right vertical)
    regions.append(
        (x - 100 + safety >= 0) and
        (x - 105 - safety <= 0) and
        (y - 10 + safety >= 0) and
        (y - 35 - safety <= 0)
    )
    
    # 6 
    ##################
    # 6, circle primitive
    regions.append(
        (((x - 120)**2 + (y - 17.5)**2) <= (7.5 + safety)**2)
    )
    # 6, vertical rectangle on top of the circle
    regions.append(
        (x - 112.5 + safety >= 0) and
        (x - 117.5 - safety <= 0) and
        (y - 17.5 + safety >= 0) and
        (y - 35 - safety <= 0)
    )
    # 6, horizontal rectangle attached to the right of the vertical rectangle
    regions.append(
        (x - 117.5 + safety >= 0) and
        (x - 123 - safety <= 0) and
        (y - 33 + safety >= 0) and
        (y - 35 - safety <= 0)
    )
    
    # 6 (second 6)
    ##################
    # 6 (second 6), circle primitive
    regions.append(
        (((x - 139.5)**2 + (y - 17.5)**2) <= (7.5 + safety)**2)
    )
    # 6 (second 6), vertical rectangle
    regions.append(
        (x - 132 + safety >= 0) and
        (x - 137 - safety <= 0) and
        (y - 17.5 + safety >= 0) and
        (y - 35 - safety <= 0)
    )
    # 6 (second 6), horizontal rectangle 
    regions.append(
        (x - 137 + safety >= 0) and
        (x - 143 - safety <= 0) and
        (y - 33 + safety >= 0) and
        (y - 35 - safety <= 0)
    )
    
    # 1
    ###############
    # 1, rectangle primitive
    regions.append(
        (x - 155 + safety >= 0) and
        (x - 160 - safety <= 0) and
        (y - 10 + safety >= 0) and
        (y - 35 - safety <= 0)
    )
    
    return any(regions)


def generate_occupancy_grid(workspace_shape):
    """
    Generates an occupancy grid based on specified bounds with a resolution of one unit. 
    """
    # Generate an empty occupancy grid having the required shape
    occupancy_grid = np.zeros(workspace_shape, dtype=bool)

    # Iterate through each location on the grid, placing a `True`` if the selected location collides with an obstacle and a `False` otherwise. 
    for x in range(workspace_shape[0]):
        for y in range(workspace_shape[1]):
            if collision(x, y):
                occupancy_grid[x, y] = True

    return occupancy_grid



def backtrack(predecessors, start, goal):
    """
    Simple backtracking routine to extract the path from the branching dictionary
    """
    path = [goal]
    current = goal
    while predecessors[current] != None:
        parent = predecessors[current]
        path.append(parent)
        current = parent
    return path[::-1]


def find_valid_neighbors(occupancy_grid, current):
    """
    Returns the valid neighbors and associated distances relative to a current node on the occupancy grid. 
    """

    # Define the list of v
    actions_list = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT, Action.UP_RIGHT, Action.UP_LEFT, Action.DOWN_RIGHT, Action.DOWN_LEFT]

    # Define lists for valid neighbors and corresponding costs
    valid_neighbors_list = []
    distances_list = []
    # Iterate through the action set to determine the valid neighbors and associated costs
    for action in actions_list:
        dx, dy = action.delta
        x, y = current 
        new_x, new_y = x + dx, y + dy 
        # If the action leads to being off grid, skip
        if new_x < 0 or new_x > occupancy_grid.shape[0]-1:
            continue
        if new_y < 0 or new_y > occupancy_grid.shape[1]-1:
            continue
        # If the action leads to a non-obstacle location, add that location to the list of valid neighbors and record its cost 
        if not occupancy_grid[new_x, new_y]:
            valid_neighbors_list.append((new_x, new_y))
            distances_list.append(action.cost)

    return valid_neighbors_list, distances_list 

def animate_search(occupancy_grid, closed_set_history, open_set_history, path,
                   filename="dijkstra_animation.mp4", skip_count=100):
    """
    This function generates a Matplotlib animation of a successful Dijkstra's search. The function saves the animation both to an MP4 file and also to a GIF file.  
    """
    
    # Create figure and plot background
    fig, ax = plt.subplots()
    ax.imshow(occupancy_grid.T, origin="lower", cmap="Greys", 
              extent=[0, occupancy_grid.shape[0], 0, occupancy_grid.shape[1]])
    path_line = ax.plot([], [], color="green", lw=3, label="Path")[0]
    start_scatter = ax.scatter([path[0][0]], [path[0][1]], label="Start", s=10, color="green")
    goal_scatter = ax.scatter([path[-1][0]], [path[-1][1]], label="Goal", s=10, color="red")

    ax.set_title("Dijkstra's Search Animation")
    ax.legend()

    # Create artist objects of the animation that will change
    open_set_scatter = ax.scatter([], [], color="blue", s=10, label="Open Set")
    closed_set_scatter = ax.scatter([], [], color="orange", s=10, label="Closed Set")




    ax.legend(loc="upper left")

    # Sample only a subset of frames
    num_frames = len(closed_set_history)
    sampled_frames = list(range(0, num_frames, skip_count))
    # Ensure the final frame is sampled
    if sampled_frames[-1] != num_frames-1: 
        sampled_frames.append(num_frames-1) 

    # Convert path to Numpy array
    path = np.array(path)

    # Define the update function
    def update(frame):
        
        
        print(f"{frame} of {num_frames-1}")

        # Extract the frame data
        open_set_frame = open_set_history[frame]
        closed_set_frame = closed_set_history[frame]
        
        # Convert tuple to NumPy array
        # If either set is empty at this frame, use an empty array
        # of the correct size
        if len(open_set_frame) > 0:
            open_set_frame = np.array(open_set_frame)
        else:
            open_set_frame = np.empty((0,2))
        if len(closed_set_frame) > 0:
            closed_set_frame = np.array(closed_set_frame)
        else:
            closed_set_frame = np.empty((0, 2))
        
        # Update the artist object states
        open_set_scatter.set_offsets(open_set_frame)
        closed_set_scatter.set_offsets(closed_set_frame)
        
        # On the last frame, overlay the path
        if frame == sampled_frames[-1]:
            path_line.set_data(path[:, 0], path[:, 1])

        # Set a title
        ax.set_title("Dijkstra's Pathfinding Animation")
        
        return open_set_scatter, closed_set_scatter, path_line
    
    # Save the animation and close
    ani = FuncAnimation(fig, update, frames=sampled_frames, blit=True)
    ani.save(filename, writer="ffmpeg", fps=20)
    ani.save("dijkstra_animation.gif", writer="imagemagick", fps=60)
    plt.close()


def dijkstras(occupancy_grid, start, goal, logging=False):
    """
    An implementation of Dijkstra's search that returns `path, cost` when a path exists and `None, None` otherwise. This function works together with other functions in this module, i.e. collision check functions, animation functions, the backtracking function, and more. 
    """
    if not in_bounds(start[0], start[1]):
        print("The start node is not in bounds")
        return None, None
    if not in_bounds(goal[0], goal[1]):
        print("The goal node is not in bounds")
        return None, None
    if collision(start[0], start[1]):
        print("The start node is in collision")
        return None, None
    if collision(goal[0], goal[1]):
        print("The goal node is in collision")
        return None, None

    # Save data for later animation and analysis
    if logging:
        open_set_history = []
        closed_set_history = []

    # Initialize data structures
    parents = dict()
    open_set = set()
    closed_set = set()
    g_scores = dict() 
    queue = PriorityQueue() 
    goal_is_found = False

    # Handle the start node
    open_set.add(start)
    parents[start] = None
    g_scores[start] = 0.0
    queue.put((g_scores[start], start))

    # Begin the Dijkstra's main loop
    while not queue.empty(): 

        # Store open and closed set history for later analysis and animation 
        if logging:
            open_set_history.append(list(open_set))
            closed_set_history.append(list(closed_set))

        # Pop the most promising node
        g_current, current = queue.get() 

        # Assume the possibility that some queued nodes are visited nodes -- SKIP THEM
        if current in closed_set:
            continue 
        
        #
        # Only proceed to process unprocessed nodes:
        #
        open_set.remove(current)
        closed_set.add(current)

        # Stop searching if the goal is found
        if current == goal:
            print("Success")
            goal_is_found = True

        # Expand the neighbors
        valid_neighbors, edge_lengths = find_valid_neighbors(occupancy_grid, current)
        for i, neighbor in enumerate(valid_neighbors):
            
            # Assume the possibility of some neighbor nodes being in the closed set -- SKIP them
            if neighbor in closed_set:
                continue 

            # Assume the possibility of some neighbor nodes being in the open set -- PROCESS THEM, but IF AND ONLY IF a better partial plan would result.
            if neighbor in open_set:
                g_tentative = g_current + edge_lengths[i]
                if g_tentative < g_scores[neighbor]: # 
                    parents[neighbor] = current 
                    g_scores[neighbor] = g_tentative
                    queue.put((g_tentative, neighbor)) 

            # Assume the possibility of some neighbor nodes only now being opened for the first time -- PROCESS THEM
            if neighbor not in closed_set and neighbor not in open_set:
                open_set.add(neighbor)
                g_tentative = g_current +  edge_lengths[i]
                parents[neighbor] = current 
                g_scores[neighbor] = g_tentative
                queue.put((g_tentative, neighbor))

    # IF a PATH EXISTS, return it and its cost; otherwise, return `None` and `None`. 
    if goal_is_found:
        cost = g_scores[goal]
        path = backtrack(parents, start, goal)

        if logging:
            animate_search(occupancy_grid, closed_set_history, open_set_history, path)

        return path, cost
    
    print("Dijkstra's failed to find a valid path")
    return None, None


if __name__ == '__main__':

    # GITHUB LINK: https://github.com/ronen-aniti-projects/Dijkstras-Point-Robot
    
    # Define the workspace shape
    workspace_shape = (181, 51)

    # Generate an occupancy grid for pathfinding. The assumption is that 1 grid unit represents 1 mm. 
    # Note: To change the safety margin around obstacles, scroll up to the collision checking function (`collision`) and change its default safety distance (`safety`) parameter.  
    occupancy_grid = generate_occupancy_grid(workspace_shape)

    # Change these as you please:
    # My implementation of Dijsktra's should handle invalid (obstacle or off-grid) entries correctly
    start = (10,5)
    goal = (120,50)

    # Run search and time it. Only turn `logging` to `True` if you need to generate an animation.  
    start_time = time.perf_counter()
    path, cost = dijkstras(occupancy_grid, start, goal, logging=False)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    print(f"Completed search in {elapsed_time: .2f} seconds")

    # Generates a 2D plot of the resulting pathfinding operation    
    if path:
        plt.imshow(occupancy_grid.T, origin="lower", cmap="Greys", 
            extent=[0, occupancy_grid.shape[0], 0, occupancy_grid.shape[1]])
        plt.scatter([pt[0] for pt in path], [pt[1] for pt in path], c="green", s=1, label="Path")
        plt.scatter([path[0][0]], [path[0][1]], label="Start", s=10, color="purple")
        plt.scatter([path[-1][0]], [path[-1][1]], label="Goal", s=10, color="Red")
        plt.legend()
        plt.show()

