# Robust Sliding Mode Control Example

This repository contains a Python script (`slide_mode.py`) that demonstrates robust sliding mode control and PD control for a robotic arm simulation. The simulation uses the Universal Robots UR5e model and Pinocchio for dynamics calculations. Plots of the results and optional video recordings of the simulation can also be generated.

---

## Requirements

### Python Dependencies

Before running the script, ensure you have the following Python packages installed:

- [NumPy](https://numpy.org/)
- [Pinocchio](https://stack-of-tasks.github.io/pinocchio/)
- [Matplotlib](https://matplotlib.org/)
- [MuJoCo](https://mujoco.org/)
- [mediapy](https://google.github.io/mediapy/mediapy.html)

You can install dependencies with pip:

```bash
conda env create -f env.yml
conda activate forc_final
```

### Other Requirements

- MJCF file for the UR5e robot model (`ur5e.xml`) located at `robots/universal_robots_ur5e/`

---

## Usage

The script provides a command-line interface to configure and run the simulation. Use the following command to run the script:

```bash
python slide_mode.py [options]
```

### Options

- `-pd, --pd`: Use the PD controller instead of sliding mode control (default: `False`).
- `-v, --show-viewer`: Show the simulation viewer (default: `False`).
- `-r, --record-video`: Record the simulation as a video (default: `False`).
- `-p, --plots`: Generate and save plots of the simulation results (default: `False`).
- `-f, --filename`: Specify the filename for the recorded video (default: `test.mp4`).
- `-t, --time-limit`: Set the simulation duration in seconds (default: `10`).
- `-s, --saturation_level`: Set the saturation level for sliding mode control (default: `None`).

---

## Examples

### Run with Sliding Mode Control

```bash
python slide_mode.py -v -p -t 15 -s 0.01
```
This runs the simulation for 15 seconds with sliding mode control, displays the viewer, generates plots, and applies a saturation level of `0.01`.

### Run with PD Control

```bash
python slide_mode.py -pd -p -t 10
```
This runs the simulation for 10 seconds with a PD controller and generates plots.

### Record Simulation Video

```bash
python slide_mode.py -r -f "my_simulation.mp4"
```
This records the simulation as a video named `my_simulation.mp4`.

---

## Output

### Plots

When the `-p` option is used, the following plots are saved in the `logs/plots_*` folder:

1. **Joint States**: Shows the joint positions over time compared to desired positions.
2. **Applied Torques**: Displays the torques applied to each joint.
3. **Sliding Variables**: (Only for sliding mode control) Plots the sliding variables and their norm.

### Videos

If the `-r` option is enabled, videos are saved in the `logs/videos` folder.

---

## Script Overview

### Key Functions

- **`sliding_surface_control`**: Implements sliding mode control with optional saturation.
- **`system_controller`**: Computes control torques using either sliding mode or PD control.
- **`save_plots_to_folder`**: Generates and saves plots of the simulation data.
- **`main`**: Sets up the simulator and runs the control loop.

---

## Customization

- **Dynamics Parameters**: Modify `OMEGA`, `K_p`, `K_d`, `L`, and `K` in the script to tune the control gains.
- **Robot Model**: Update the `xml_path` variable to use a different MJCF model.

---

## Troubleshooting

- Ensure all required dependencies are installed.
- Verify the MJCF file path and structure.
- If Pinocchio-related errors occur, confirm the UR5e model is compatible with the version of Pinocchio installed.
