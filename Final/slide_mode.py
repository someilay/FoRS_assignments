import time
import numpy as np
import os
import pinocchio as pin
import argparse
import matplotlib.pyplot as plt

from simulator import Simulator
from pathlib import Path
from typing import Dict


def sliding_surface_control(
    mass_matrix: np.ndarray,
    bias_force: np.ndarray,
    q: np.ndarray,
    dq: np.ndarray,
    q_d: np.ndarray,
    dq_d: np.ndarray,
    ddq_d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    delta = q_d - q
    d_delta = dq_d - dq

    s = d_delta + L * delta

    if saturation_level is not None:
        v_s = K * s / np.max([np.linalg.norm(s), saturation_level])
    else:
        v_s = K * s / np.linalg.norm(s)

    v = ddq_d + L * d_delta + v_s
    tau = mass_matrix.dot(v) + bias_force

    return tau, s


def system_controller(
    q: np.ndarray, dq: np.ndarray, t: float, desired: Dict
) -> np.ndarray:
    start = time.perf_counter()

    # Get mass matrix and bias forces
    mass_matrix = pin.crba(model, data, q)
    bias_force = pin.rnea(model, data, q, dq, np.zeros(model.nv))

    q_d = np.array(
        [-0.42647973, -1.34054273, 1.28128263, 0.0592601, -0.42647973, -3.14159266],
        dtype=float,
    )
    if not use_pd:
        tau, s = sliding_surface_control(
            mass_matrix=mass_matrix,
            bias_force=bias_force,
            q=q,
            dq=dq,
            q_d=q_d,
            dq_d=np.zeros(6, dtype=float),
            ddq_d=np.zeros(6, dtype=float),
        )
    else:
        tau = mass_matrix.dot(K_p * (q_d - q) - K_d * dq) + bias_force

    end = time.perf_counter()
    elapsed_time = (end - start) * 1e3
    print(f"Control calculation elapsed time: {elapsed_time:.3f} ms")

    if save_plots:
        sim_data["time"].append(t)
        sim_data["qs"].append(q.copy())
        sim_data["qs_d"].append(q_d.copy())
        sim_data["ctrl"].append(tau.copy())
        if not use_pd:
            sim_data["s"].append(s.copy())

    return tau


def save_plots_to_folder():
    # Create plots directory
    if saturation_level is None:
        folder_path = Path("logs/plots_slide") if not use_pd else Path("logs/plots_pd")
    else:
        folder_path = Path(f"logs/plots_slide_sat_{saturation_level:.3f}") if not use_pd else Path("logs/plots_pd")

    folder_path.mkdir(parents=True, exist_ok=True)

    t = np.array(sim_data["time"])
    qs = np.array(sim_data["qs"])
    qs_d = np.array(sim_data["qs_d"])
    ctrl = np.array(sim_data["ctrl"])
    slide = np.array(sim_data["s"])

    if not use_pd:
        if saturation_level is None:
            title_suf = f"($k={K:.1f}$ and $\\Lambda=$diag$({L}))$"
        else:
            title_suf = f"\n($k={K:.1f}$, $\\Lambda=$diag$({L})$, and $\\epsilon={saturation_level:.3f})$"
    else:
        title_suf = f"$K_p={K_p:.1f}$ and $K_d={K_d:.1f}$"

    plt.title(f"Joint state {title_suf}")
    plt.plot(t, qs[:, 0], color="blue", label="$q_1$, rads")
    plt.plot(t, qs[:, 1], color="green", label="$q_2$, rads")
    plt.plot(t, qs[:, 2], color="orange", label="$q_3$, rads")
    plt.plot(t, qs[:, 3], color="yellow", label="$q_4$, rads")
    plt.plot(t, qs[:, 4], color="gray", label="$q_5$, rads")
    plt.plot(t, qs[:, 5], color="black", label="$q_6$, rads")
    plt.plot(t, qs_d[:, 0], "--", color="blue", label="$q_{1,d}$, rads", alpha=0.6)
    plt.plot(t, qs_d[:, 1], "--", color="green", label="$q_{2,d}$, rads", alpha=0.6)
    plt.plot(t, qs_d[:, 2], "--", color="orange", label="$q_{3,d}$, rads", alpha=0.6)
    plt.plot(t, qs_d[:, 3], "--", color="yellow", label="$q_{4,d}$, rads", alpha=0.6)
    plt.plot(t, qs_d[:, 4], "--", color="gray", label="$q_{4,d}$, rads", alpha=0.6)
    plt.plot(t, qs_d[:, 5], "--", color="black", label="$q_{6,d}$, rads", alpha=0.6)
    plt.grid()
    plt.legend(loc="lower right")
    plt.xlabel("$t$, seconds")
    plt.ylabel("rads")
    plt.savefig(folder_path / "states.png")

    plt.cla()
    plt.title(f"Applied torques {title_suf}")
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

    if not use_pd:
        plt.cla()
        plt.title(f"$s$ {title_suf}")
        plt.plot(t, slide[:, 0], color="blue", label="$s_1$")
        plt.plot(t, slide[:, 1], color="green", label="$s_2$")
        plt.plot(t, slide[:, 2], color="orange", label="$s_3$")
        plt.plot(t, slide[:, 3], color="yellow", label="$s_4$")
        plt.plot(t, slide[:, 4], color="gray", label="$s_5$")
        plt.plot(t, slide[:, 5], color="black", label="$s_6$")
        plt.grid()
        plt.legend(loc="lower right")
        plt.xlabel("$t$, seconds")
        plt.savefig(folder_path / "slide_variable.png")

        plt.cla()
        plt.title(f"$\\|s\\|$ {title_suf}")
        plt.plot(t, np.linalg.norm(slide, axis=1), color="blue", label="$\\|s\\|$")
        plt.grid()
        plt.legend(loc="lower right")
        plt.xlabel("$t$, seconds")
        plt.savefig(folder_path / "slide_norm.png")


def main(
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

    # Only disable mocap body interactions
    mocap_body_id = sim.model.body(sim.mocap_name).id
    sim.model.body_contype[mocap_body_id] = 0
    sim.model.body_conaffinity[mocap_body_id] = 0
    # Set joint damping coefficients
    damping = np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.1])  # Nm/rad/s
    sim.set_joint_damping(damping)
    # Set joint friction coefficients
    friction = np.array([1.5, 0.5, 0.5, 0.1, 0.1, 0.1])  # Nm
    sim.set_joint_friction(friction)
    # Modify end-effector mass
    sim.modify_body_properties("end_effector", mass=4.0)

    sim.set_controller(system_controller)
    sim.run(time_limit=time_limit)

    if save_plots:
        save_plots_to_folder()


if __name__ == "__main__":
    np.printoptions(precision=1, suppress=True)
    parser = argparse.ArgumentParser(
        prog="Robust Sliding Mode Control example",
    )
    parser.add_argument(
        "-pd",
        "--pd",
        action="store_true",
        required=False,
        default=False,
        help="disable sliding mode control, use PD controller instead",
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
        "-s",
        "--saturation_level",
        type=float,
        required=False,
        default=None,
        help="saturation for sliding mode",
    )
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(current_dir, "robots/universal_robots_ur5e/ur5e.xml")
    model = pin.buildModelFromMJCF(xml_path)
    data = model.createData()

    OMEGA = 10
    K_p = OMEGA**2
    K_d = 2 * OMEGA
    L = np.array([50, 50, 50, 50, 25, 25]).astype(float)
    K = 2e3

    save_plots = args.plots
    saturation_level = args.saturation_level
    use_pd = args.pd
    sim_data = {
        "qs": [],
        "qs_d": [],
        "ctrl": [],
        "time": [],
        "s": [],
    }

    main(
        show_viewer=args.show_viewer,
        record_video=args.record_video,
        filename=args.filename,
        time_limit=args.time_limit,
    )
