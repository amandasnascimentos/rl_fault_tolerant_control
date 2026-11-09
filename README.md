
# rl_fault_tolerant_control

Thruster allocation for the BlueROV2 Heavy in the HoloOcean simulator. This is the foundation for fault-tolerant control (FTC) using a weighted pseudo-inverse and, later, reinforcement learning.

At this stage the project contains only the allocator and its validation tests. There is no closed-loop controller yet.

## Requirements

- Linux (tested on Ubuntu 24)
- Python 3.10 or newer
- HoloOcean 2.4.0 with the `Ocean` world installed
- A graphics environment capable of running the Unreal Engine used by HoloOcean

## Installation

```bash
git clone <repository-url> rl_fault_tolerant_control
cd rl_fault_tolerant_control

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

On first run, HoloOcean may download the assets for the `Ocean` world. This takes a few minutes and requires an internet connection.

## Project structure

```
rl_fault_tolerant_control/
├── experiments/
│   ├── run_tests.py                # tests allocation with requested forces
│   └── andar.py                    # tests the 6 degrees of freedom (sign and coupling)
├── src/
│   ├── config/
│   │   └── bluerov2_config.yaml    # geometry, thruster order and signs
│   ├── control/
│   │   └── allocator.py            # GenericThrusterAllocator (B matrix and pseudo-inverse)
│   └── environment/
│       └── holoocean_config.py     # default scenario and HoloOcean wrapper
├── pyproject.toml
└── requirements.txt
```

## How to run

Run all commands from the `rl_fault_tolerant_control/` folder.

Allocation test with requested forces on each axis:

```bash
python3 experiments/run_tests.py
```

Six degrees-of-freedom test with sign and coupling checks:

```bash
python3 experiments/andar.py
```

The final summary of `andar.py` should show `6/6`.

## How allocation works

The allocator takes a generalized force vector `tau = [Fx, Fy, Fz, Mx, My, Mz]` and returns the commands for the 8 thrusters:

```
u = pinv(B) @ tau
```

If any command exceeds `u_max`, all commands are scaled down by the same factor, which preserves the direction of the requested effort.

The matrix `B` is built from the geometry in Wu (2018), Table 4.2 and Eq. 4.62, using:

- the HoloOcean thruster order (`ho_to_model`);
- the signs validated against the simulator (`sinais`).

## Units

- `u` is the force of each thruster in **newtons**. With `k_n: 1.0`, a command of 1 means 1 N.
- `tau` is in N for forces and N·m for torques.

The scale was measured in the simulator itself (acceleration with `u = 1` on the horizontal thrusters), not taken from the thesis. The thesis uses `K = 40`, which does not match HoloOcean's behavior.

## What is validated

- Sign and coupling of all 8 thrusters, compared against the matrix.
- Linear response of surge, sway and heave for small `tau`.
- Sign of roll, pitch and yaw, with coupling below 0.01°.

## Known limitations

- **No closed-loop control.** The allocator is open loop. With a constant `tau`, the vehicle accelerates without bound.
- **Yaw shows no apparent damping.** Under constant torque, yaw grows almost linearly. This is the simulator's behavior, not a matrix error.
- **Saturation limit not confirmed.** The `u_max` in the YAML is a reference value (40 N, from the thesis), not a limit measured in HoloOcean.
- **Mass and inertia were not measured.** The estimate of K used 11 kg, which is an assumption.
- **Magnitude tests are indicative.** The tests validate signs and coupling; they do not validate the exact magnitude of the forces.

## References

- Wu, Y. (2018). *Control of 6-DoF ROV*. Thesis. Geometry and allocation matrix for the BlueROV2 Heavy (Table 4.2 and Eq. 4.62).
- HoloOcean documentation: BlueROV2 (`control_scheme` 0: Thruster Forces).
