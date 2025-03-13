# Implementing Dijkstra's Algorithm for a Point Robot

## Summary
The following is my submission for a class project, Project 2 for ENPM661, Planning for Autonomous Robotics. For this project, I developed a pathfinding framework that employs Dijkstra's algorithm to find shortest-distance paths through an occupancy-grid representation of a 2D workspace. The results of the project are exemplified in the animation shown below. 

**Figure 1.** The results of the project: Employing Dijkstra's algorithm to find shortest-distance paths through 2D occupancy-grid representations of a workspace. 

![Animation GIF](/dijkstra_animation.gif)

While completing this project, I gained experience troubleshooting Dijkstra's algorithm, representing obstacles with semi-algebraic half-plane models, and writing modular computer code. I also gained a clearer understanding of the foundational path planning concepts of workspace, obstacle space, free space, action set, collision checking, and other related ideas. 

The reader is encouraged to review the source code for my solution, included in `dijkstras_ronen_aniti.py`. 

## How to Run 

1. **Ensure you have the proper dependencies.** This project relies on Numpy and Matplotlib. 
2. **Experiment with `dijkstras_ronen_aniti.py`**. Open the main file. Scroll to the bottom. Experiment with different start and goal combinations or a different safety margin. 
3. **Examine the results in a visual way**. Examine the resulting 2D plot of the shortest-distance path to ensure the planner is functioning as expected.  
5. **Examine runtime performance**. Executing the script will print the Dijkstra's pathfinding runtime to the console.    

## Rubric Points (For reference)

### Defining the Robot Actions
> "Define correct actions for BFS and Dijkstra's(optional) algorithm"

### Representing the Obstacle Space
> "Create obstacle space using half planes and semi-algebraic models"

### Checking the Start and Goal Points
> "Check the start and goal point if they fall in obstacle space"

### Exploring Nodes and Backtracking
> "Implement BFS. The code should have sufficient comments and documentation to explain the approach (Implement Dijkstra's algorithm- optional)"

### Visualizing the Results 
> "Show visualization video demonstrating node exploration and final path generated in form of a video or gif file"

### Following Submission Guidelines
> "README file (.md or .txt), Source Codes (.py) and Animation Videos" 

### Accounting for Runtime Limitations
> "The algorithms are able to find a solution(if possible) for any start and goal points in less than 5 minutes"