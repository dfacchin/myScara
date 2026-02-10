# CLAUDE.md

## Project Overview

**myScara** is a SCARA robot dynamics simulator for pick & place trajectory planning. It simulates a 2-DOF horizontal SCARA arm with variable payload, using MyActuator RMD-X8-25 motors.

Single-file Python project: `scara_pick_place.py` (~1100 lines).

## Tech Stack

- **Language**: Python 3.11+
- **Dependencies**: NumPy, SciPy, Matplotlib
- **No build system** — runs directly as a Python script

## How to Run

```bash
python3 scara_pick_place.py
```

Output: console logs (validation, timeline, analysis) + PNG visualization saved to `/mnt/user-data/outputs/scara_pick_place.png`.

## Project Structure

```
myScara/
├── scara_pick_place.py          # Main simulator (all code lives here)
├── TODO.txt                     # Project tasks
├── CLAUDE.md                    # This file
└── (X-V3) Protocol and manual/  # Hardware documentation PDFs
```

## Key Concepts

- **Kinematics**: Forward/inverse kinematics, Jacobian, singularity detection
- **Dynamics**: Inertia matrix, Coriolis forces, inverse dynamics with variable payload
- **Trajectory**: Trapezoidal and S-curve velocity profiles, waypoint-based planning
- **Analysis**: Motor feasibility checks (peak/RMS torque), singularity margins

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
```
