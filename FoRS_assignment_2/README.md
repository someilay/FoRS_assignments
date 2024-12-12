# Using `task_space.py` for Task-Space Control

The `task_space.py` script simulates task-space control for a robotic arm, allowing users to track predefined trajectories or manually control the end-effector's pose. This guide explains how to set up and run the script.

---

## Prerequisites

### Dependencies
Make sure you have the following libraries installed:

- Python 3.10+
- [NumPy](https://numpy.org/)
- [CVXPY](https://www.cvxpy.org/)
- [Pinocchio](https://stack-of-tasks.github.io/pinocchio/)
- [Matplotlib](https://matplotlib.org/)
- [MuJoCo](https://mujoco.org/)

You can install dependencies with pip:

```bash
conda env create -f env.yml
conda activate forc_hw
```

### Robot Assets
Ensure the required robot model files (e.g., `ur5e.xml`) are located in the `robots/universal_robots_ur5e` directory.

---

## Script Overview

### Main Features
- **Task-Space Trajectory Tracking**: Follows a predefined end-effector trajectory.
- **Manual Pose Control**: Allows user-defined pose inputs for the end-effector.
- **Visualization**: Generates plots and optional videos of the simulation.

### Command-Line Arguments
The script accepts the following arguments:

| Argument                 | Description                                                                 | Default      |
|--------------------------|-----------------------------------------------------------------------------|--------------|
| `--trajectory`      | Choose the trajectory type (`trajectory` or `manual`).                      | `False` |
| `--show-viewer`               | Display the simulation viewer (`True` or `False`).                         | `False`      |
| `--record-video`                | Record a video of the simulation (`True` or `False`).                      | `False`      |
| `--plots`          | Save plots                                          | `False`     |
| `--filename`          | Name of recorded video that would be stored in logs/videos folder, default is test.mp4                                          | `test.mp`     |
| `--time-limit`                 | Duration of the simulation in seconds.                                     | `10.0`       |
| `--omega`                | Regulation coefficients, default is 40         | `40`        |

---

## Running the Script

### Example Commands

#### Trajectory Tracking
Run the simulation with a predefined trajectory:

```bash
python task_space.py --trajectory --show-viewer --time-limit 15
```

- This example enables the viewer and runs the simulation for 15 seconds.
- The end-effector will follow a circular trajectory.

#### Manual Pose Control
Manually control the end-effector's pose:

```bash
python task_space.py --show-viewer --plots --time-limit 20
```

- Save the results to `logs`.
- Use the interactive viewer to specify poses.

#### Record a Simulation Video
To save a video of the simulation:

```bash
python task_space.py --show-viewer --record-video --time-limit 5 --filename sample.mp4
```

- Videos will be saved in the `logs` directory.

---

## Output

### Videos and Plots
The script generates:
- **Plots**: Visualizations of the end-effector's position, velocity, and control torques
- **Videos**: Videos of the simulation

### Videos
If `--record-video` is specified, the video is saved in the `logs`.

---

## Customization

### Modifying the Trajectory
To define a custom trajectory:
1. Edit the trajectory generation logic in the `task_space_controller_trajectory` function.
2. Replace the circular trajectory with your desired function.

### Changing Robot Parameters
Update the robot model configuration in the `robots/universal_robots_ur5e/ur5e.xml` file or adjust parameters in the script to suit your robot.

---

## Troubleshooting

### Common Issues

1. **Solver Errors**:
   - Ensure CVXPY is installed and compatible solvers (e.g., `OSQP`, `SCS`) are available.

2. **Viewer Not Displaying**:
   - Check if MuJoCo is correctly installed and the `mujoco_py` bindings are functional.

3. **Performance**:
   - For slow performance, try reducing the simulation time or disabling the viewer.

### Debugging Tips
Enable debugging logs in the script for detailed runtime information.

---
