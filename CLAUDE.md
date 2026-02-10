# CLAUDE.md

## Project Overview

**myScara** is a SCARA robot dynamics simulator for pick & place trajectory planning. It simulates a 2-DOF horizontal SCARA arm with variable payload, using MyActuator RMD-X8-25 motors.

Two Python files: `scara_pick_place.py` (trajectory simulator) and `scara_controller.py` (motor control).

## Tech Stack

- **Language**: Python 3.11+
- **Dependencies**: NumPy, SciPy, Matplotlib
- **No build system** — runs directly as a Python script

## How to Run

```bash
# Simulator only (trajectory planning + dynamics analysis)
python3 scara_pick_place.py

# Controller dry-run (simulator + motor control, no hardware)
python3 scara_controller.py --dry-run

# Controller with hardware (CAN bus)
python3 scara_controller.py --hardware --bus can --channel can0

# Controller with hardware (RS485)
python3 scara_controller.py --hardware --bus rs485 --serial-port /dev/ttyUSB0
```

Output: console logs + PNG visualizations saved to `/mnt/user-data/outputs/`.

## Project Structure

```
myScara/
├── scara_pick_place.py          # Trajectory simulator (kinematics, dynamics, visualization)
├── scara_controller.py          # Motor controller (X-V3 protocol, trajectory execution)
├── TODO.txt                     # Project tasks
├── CLAUDE.md                    # This file
└── (X-V3) Protocol and manual/  # Hardware documentation PDFs
```

## Key Concepts

- **Kinematics**: Forward/inverse kinematics, Jacobian, singularity detection
- **Dynamics**: Inertia matrix, Coriolis forces, inverse dynamics with variable payload
- **Trajectory**: Trapezoidal and S-curve velocity profiles, waypoint-based planning
- **Analysis**: Motor feasibility checks (peak/RMS torque), singularity margins
- **Motor Control**: MyActuator V3 protocol (CAN/RS485), position/speed/torque commands
- **Controller**: Real-time trajectory execution at configurable rate, safety monitoring

## Code Conventions

- Comments and docstrings are in **Italian** (variable names in English)
- All physics, trajectory planning, and visualization code is in a single file
- Uses Python dataclasses for configuration (`SCARAParams`, `Waypoint`, `TrajectoryConstraints`)
- Uses enums for arm configuration (`LEFT`/`RIGHT`) and velocity profile type

## No Tests / No Linting

There is no formal test suite or linter configured. Validation is built into the simulation (waypoint checks, kinematics verification, torque feasibility).

## Dependencies

No `requirements.txt` exists. Install manually:

```bash
pip install numpy scipy matplotlib

# Optional: for hardware motor control
pip install python-can   # CAN bus support
pip install pyserial     # RS485 support
```
