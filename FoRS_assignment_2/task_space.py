"""Task space (operational space) control example.

This example demonstrates how to implement task space control for a robot arm,
allowing it to track desired end-effector positions and orientations. The example
uses a simple PD control law but can be extended to more sophisticated controllers.

Key Concepts Demonstrated:
    - Task space control implementation
    - End-effector pose tracking
    - Real-time target visualization
    - Coordinate frame transformations

Example:
    To run this example:
    
    $ python 03_task_space.py

Notes:
    - The target pose can be modified interactively using the MuJoCo viewer
    - The controller gains may need tuning for different trajectories
    - The example uses a simplified task space controller for demonstration
"""

import time
import numpy as np
import os
import pinocchio as pin
import cvxpy as cp
import argparse
import matplotlib.pyplot as plt

from simulator import Simulator
from pathlib import Path
from typing import Dict


class TaskSolver:
    def __init__(self, nv: int, absolute_bound: np.ndarray):
        self.nv = nv
        self.a = cp.Variable(nv)
        self.tau = cp.Variable(nv)
        self.mass_matrix = cp.Parameter((nv, nv), name="M", PSD=True)
        self.bias_force = cp.Parameter(nv, name="h")
        self.a_mat = cp.Parameter((6, nv), name="A")
        self.b_vec = cp.Parameter(6, name="b")
        self.absolute_bound = absolute_bound

        constraints = [
            self.mass_matrix @ self.a + self.bias_force == self.tau,
            -self.absolute_bound <= self.tau,
            self.tau <= self.absolute_bound,
        ]
        target = self.a_mat @ self.a - self.b_vec
        obj = cp.Minimize(cp.sum_squares(target))
        self.problem = cp.Problem(obj, constraints)

    def solve(
        self,
        mass_matrix: np.ndarray,
        bias_force: np.ndarray,
        a_mat: np.ndarray,
        b_vec: np.ndarray,
        **kwargs,
    ):
        self.mass_matrix.value = mass_matrix
        self.bias_force.value = bias_force
        self.a_mat.value = a_mat
        self.b_vec.value = b_vec
        self.problem.solve(**kwargs)


def task_space_controller_trajectory(
    q: np.ndarray, dq: np.ndarray, t: float, desired: Dict
) -> np.ndarray:
    """Example task space controller with predefined trajectory"""
    start = time.perf_counter()

    # Desired position
    omega = np.pi
    x_d = 0.5 + 0.1 * np.sin(omega * t)
    z_d = 0.4 + 0.1 * np.cos(omega * t)
    desired_position = np.array([x_d, 0, z_d])
    desired_orientation = np.array(
        [
            [0, 1, 0],
            [-x_d / np.sqrt(x_d**2 + z_d**2), 0, -z_d / np.sqrt(x_d**2 + z_d**2)],
            [-z_d / np.sqrt(x_d**2 + z_d**2), 0, x_d / np.sqrt(x_d**2 + z_d**2)],
        ]
    ).T
    desired_se3 = pin.SE3(desired_orientation, desired_position)

    # Desired velocity
    dx_d = 0.1 * omega * np.cos(omega * t)
    dz_d = -0.1 * omega * np.sin(omega * t)
    desired_velocity = np.array([dx_d, 0, dz_d])
    desired_ang_vel = np.array([dx_d * z_d - dz_d * x_d, 0, 0]) / (x_d**2 + z_d**2)
    desired_twist = np.array([*desired_velocity, *desired_ang_vel])

    # Desired acceleration
    ddx_d = -0.1 * omega * omega * np.sin(omega * t)
    ddz_d = -0.1 * omega * omega * np.cos(omega * t)
    desired_acceleration = np.array([ddx_d, 0, ddz_d])
    desired_ang_acc = np.array(
        [
            (x_d**2 + z_d**2) * (ddx_d * z_d - ddz_d * x_d)
            + 2 * (dx_d * x_d + dz_d * z_d) * (-dx_d * z_d + dz_d * x_d),
            0,
            0,
        ]
    ) / ((x_d**2 + z_d**2) ** 2)
    desired_acc = np.array([*desired_acceleration, *desired_ang_acc])

    # Get end-effector frame ID
    ee_frame_id = model.getFrameId("end_effector")
    # Compute forward kinematics
    pin.forwardKinematics(model, data, q, dq)
    pin.computeJointJacobiansTimeVariation(model, data, q, dq)
    # Calculate kinematics of frames
    pin.updateFramePlacement(model, data, ee_frame_id)
    # Get velocities and frame pose
    twist = pin.getFrameVelocity(model, data, ee_frame_id, pin.LOCAL_WORLD_ALIGNED)
    ee_pose = data.oMf[ee_frame_id]
    ee_position = ee_pose.translation
    ee_rotation = ee_pose.rotation
    # Jacobians
    jac = pin.getFrameJacobian(model, data, ee_frame_id, pin.LOCAL_WORLD_ALIGNED)
    d_jac = pin.getFrameJacobianTimeVariation(
        model, data, ee_frame_id, pin.LOCAL_WORLD_ALIGNED
    )
    # Translate angular velocity to desired frame
    jac[3:] = desired_se3.rotation.T @ jac[3:]
    d_jac[3:] = desired_se3.rotation.T @ d_jac[3:]
    twist_vector = twist.vector
    twist_vector[3:] = desired_se3.rotation.T @ twist_vector[3:]
    # Get mass matrix and bias forces
    mass_matrix = pin.crba(model, data, q)
    bias_force = pin.rnea(model, data, q, dq, np.zeros(model.nv))

    # Define optimization problem
    pose_diff = np.array(
        [
            *(desired_se3.translation - ee_position),
            *pin.log3(ee_rotation.T @ desired_se3.rotation),
        ]
    )
    vel_diff = desired_twist - twist_vector
    a_mat = jac
    b_vec = desired_acc - d_jac @ dq + K_d * vel_diff + K_p * pose_diff
    solver.solve(mass_matrix, bias_force, a_mat, b_vec, solver="SCS")

    if solver.problem.status not in ["optimal", "solved"]:
        raise RuntimeError(
            f"Cannot compute an optimal control! Solver status: {solver.problem.status}"
        )

    end = time.perf_counter()
    elapsed_time = (end - start) * 1e3
    print(f"Control calculation elapsed time: {elapsed_time:.3f} ms")

    if save_plots:
        sim_data["time"].append(t)
        sim_data["xpos"].append(ee_position.copy())
        sim_data["d_xpos"].append(desired_position.copy())
        sim_data["quat"].append(pin.SE3ToXYZQUAT(ee_pose)[3:].copy())
        sim_data["d_quat"].append(pin.SE3ToXYZQUAT(desired_se3)[3:].copy())
        sim_data["ctrl"].append(solver.tau.value.copy())

    return solver.tau.value


def task_space_controller_manual(
    q: np.ndarray, dq: np.ndarray, t: float, desired: Dict
) -> np.ndarray:
    """Example task space controller with manual control"""
    start = time.perf_counter()

    # Convert desired pose to SE3
    desired_position = desired["pos"]
    desired_quaternion = desired["quat"]  # [w, x, y, z] in MuJoCo format
    desired_quaternion_pin = np.array(
        [*desired_quaternion[1:], desired_quaternion[0]]
    )  # Convert to [x,y,z,w] for Pinocchio
    # Convert to pose and SE3
    desired_pose = np.concatenate([desired_position, desired_quaternion_pin])
    desired_se3 = pin.XYZQUATToSE3(desired_pose)

    # Get end-effector frame ID
    ee_frame_id = model.getFrameId("end_effector")
    # Compute forward kinematics
    pin.forwardKinematics(model, data, q, dq)
    pin.computeJointJacobiansTimeVariation(model, data, q, dq)
    # Calculate kinematics of frames
    pin.updateFramePlacement(model, data, ee_frame_id)
    # Get velocities and frame pose
    twist = pin.getFrameVelocity(model, data, ee_frame_id, pin.LOCAL_WORLD_ALIGNED)
    ee_pose = data.oMf[ee_frame_id]
    ee_position = ee_pose.translation
    ee_rotation = ee_pose.rotation
    # Jacobians
    jac = pin.getFrameJacobian(model, data, ee_frame_id, pin.LOCAL_WORLD_ALIGNED)
    d_jac = pin.getFrameJacobianTimeVariation(
        model, data, ee_frame_id, pin.LOCAL_WORLD_ALIGNED
    )
    # Translate angular velocity to desired frame
    jac[3:] = desired_se3.rotation.T @ jac[3:]
    d_jac[3:] = desired_se3.rotation.T @ d_jac[3:]
    twist_vector = twist.vector
    twist_vector[3:] = desired_se3.rotation.T @ twist_vector[3:]
    # Get mass matrix and bias forces
    mass_matrix = pin.crba(model, data, q)
    bias_force = pin.rnea(model, data, q, dq, np.zeros(model.nv))

    # Define optimization problem
    desired_acc = np.zeros(6)
    pose_diff = np.array(
        [
            *(desired_se3.translation - ee_position),
            *pin.log3(ee_rotation.T @ desired_se3.rotation),
        ]
    )
    vel_diff = np.zeros(6) - twist_vector
    a_mat = jac
    b_vec = desired_acc - d_jac @ dq + K_d * vel_diff + K_p * pose_diff
    solver.solve(mass_matrix, bias_force, a_mat, b_vec, solver="SCS")

    if solver.problem.status not in ["optimal", "solved"]:
        raise RuntimeError(
            f"Cannot compute an optimal control! Solver status: {solver.problem.status}"
        )

    end = time.perf_counter()
    elapsed_time = (end - start) * 1e3
    print(f"Control calculation elapsed time: {elapsed_time:.3f} ms")

    if save_plots:
        sim_data["time"].append(t)
        sim_data["xpos"].append(ee_position.copy())
        sim_data["d_xpos"].append(desired_position.copy())
        sim_data["quat"].append(pin.SE3ToXYZQUAT(ee_pose)[3:].copy())
        sim_data["d_quat"].append(desired_quaternion_pin.copy())
        sim_data["ctrl"].append(solver.tau.value.copy())

    return solver.tau.value


def save_plots_to_folder(trajectory: bool):
    # Create plots directory
    folder_path = (
        Path("logs/plots_manual") if not trajectory else Path("logs/plots_trajectory")
    )
    folder_path.mkdir(parents=True, exist_ok=True)

    t = np.array(sim_data["time"])
    xpos = np.array(sim_data["xpos"])
    d_xpos = np.array(sim_data["d_xpos"])
    quat = np.array(sim_data["quat"])
    d_quat = np.array(sim_data["d_quat"])
    ctrl = np.array(sim_data["ctrl"])

    plt.title(f"End effector position ($K_p={K_p:.1f}$ and $K_d={K_d:.1f}$)")
    plt.plot(t, xpos[:, 0], color="blue", label="$x$, meters")
    plt.plot(t, xpos[:, 1], color="green", label="$y$, meters")
    plt.plot(t, xpos[:, 2], color="orange", label="$z$, meters")
    plt.plot(t, d_xpos[:, 0], "--", color="blue", label="$x_d$, meters", alpha=0.6)
    plt.plot(t, d_xpos[:, 1], "--", color="green", label="$y_d$, meters", alpha=0.6)
    plt.plot(t, d_xpos[:, 2], "--", color="orange", label="$z_d$, meters", alpha=0.6)
    plt.grid()
    plt.legend(loc="lower right")
    plt.xlabel("$t$, seconds")
    plt.ylabel("meters")
    plt.savefig(folder_path / "positions.png")

    plt.cla()
    plt.title(f"End effector orientation ($K_p={K_p:.1f}$ and $K_d={K_d:.1f}$)")
    plt.plot(t, quat[:, 0], color="blue", label="$x$")
    plt.plot(t, quat[:, 1], color="green", label="$y$")
    plt.plot(t, quat[:, 2], color="orange", label="$z$")
    plt.plot(t, quat[:, 3], color="yellow", label="$w$")
    plt.plot(t, d_quat[:, 0], "--", color="blue", label="$x_d$", alpha=0.6)
    plt.plot(t, d_quat[:, 1], "--", color="green", label="$y_d$", alpha=0.6)
    plt.plot(t, d_quat[:, 2], "--", color="orange", label="$z_d$", alpha=0.6)
    plt.plot(t, d_quat[:, 3], "--", color="yellow", label="$w_d$", alpha=0.6)
    plt.grid()
    plt.legend(loc="best")
    plt.xlabel("$t$, seconds")
    plt.savefig(folder_path / "orientations.png")

    plt.cla()
    plt.title(f"Applied torques ($K_p={K_p:.1f}$ and $K_d={K_d:.1f}$)")
    plt.plot(t, ctrl[:, 0], color="blue", label="$u_1$")
    plt.plot(t, ctrl[:, 1], color="green", label="$u_2$")
    plt.plot(t, ctrl[:, 2], color="orange", label="$u_3$")
    plt.plot(t, ctrl[:, 3], color="yellow", label="$u_4$")
    plt.plot(t, ctrl[:, 4], color="gray", label="$u_5$")
    plt.plot(t, ctrl[:, 5], color="black", label="$u_6$")
    plt.grid()
    plt.legend(loc="best")
    plt.xlabel("$t$, seconds")
    plt.ylabel("$\log N \cdot m$")
    plt.yscale("symlog", base=10)
    plt.savefig(folder_path / "torques.png")


def main(
    trajectory: bool,
    show_viewer: bool,
    record_video: bool,
    filename: str,
    time_limit: float,
):
    # Create logging directories
    Path("logs/videos").mkdir(parents=True, exist_ok=True)

    print("\nRunning task space controller...")
    sim = Simulator(
        xml_path="robots/universal_robots_ur5e/scene.xml",
        enable_task_space=True,
        show_viewer=show_viewer,
        record_video=record_video,
        video_path=f"logs/videos/{filename}",
        fps=30,
        width=1920,
        height=1080,
    )

    if not trajectory:
        sim.set_controller(task_space_controller_manual)
    else:
        # Hide target mocap body
        target_geom_id = sim.model.body(sim.mocap_name).geomadr[0]
        sim.model.geom_rgba[target_geom_id] = [0, 0, 0, 0]  # Fully transparent
        # Hide all sites
        sim.model.site_rgba[1] = [0, 0, 0, 0]  # Fully transparent
        # Only disable mocap body interactions
        mocap_body_id = sim.model.body(sim.mocap_name).id
        sim.model.body_contype[mocap_body_id] = 0
        sim.model.body_conaffinity[mocap_body_id] = 0
        sim.set_controller(task_space_controller_trajectory)

    sim.run(time_limit=time_limit)

    if save_plots:
        save_plots_to_folder(trajectory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Task space (operational space) control example",
    )
    parser.add_argument(
        "-tr",
        "--trajectory",
        action="store_true",
        required=False,
        default=False,
        help="starts simulation with predefined trajectory",
    )
    parser.add_argument(
        "-v",
        "--show-viewer",
        action="store_true",
        required=False,
        default=False,
        help="show viewer",
    )
    parser.add_argument(
        "-r",
        "--record-video",
        action="store_true",
        required=False,
        default=False,
        help="record video",
    )
    parser.add_argument(
        "-p",
        "--plots",
        action="store_true",
        required=False,
        default=False,
        help="generate plots",
    )
    parser.add_argument(
        "-f",
        "--filename",
        type=str,
        required=False,
        default="test.mp4",
        help="name of recorded video that would be stored in logs/videos folder, default is test.mp4",
    )
    parser.add_argument(
        "-t",
        "--time-limit",
        type=float,
        required=False,
        default=10,
        help="duration of simulation, in seconds, default is 10 s",
    )
    parser.add_argument(
        "-o",
        "--omega",
        type=float,
        required=False,
        default=40,
        help="omega regulation coefficients, default is 40",
    )
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(current_dir, "robots/universal_robots_ur5e/ur5e.xml")
    model = pin.buildModelFromMJCF(xml_path)
    data = model.createData()

    OMEGA = args.omega
    K_p = OMEGA**2
    K_d = 2 * OMEGA
    absolute_bound = np.array([150, 150, 150, 28, 28, 28], dtype=float)
    solver = TaskSolver(model.nv, absolute_bound)

    save_plots = args.plots
    sim_data = {"xpos": [], "d_xpos": [], "quat": [], "d_quat": [], "ctrl": [], "time": []}

    main(
        args.trajectory,
        args.show_viewer,
        args.record_video,
        args.filename,
        args.time_limit,
    )
