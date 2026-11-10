import numpy as np
import yaml
from pathlib import Path


class GenericThrusterAllocator:

    def __init__(self, config_path=None):
        if config_path is None:
            config_path = Path(__file__).resolve().parents[1] / "config" / "bluerov2_config.yaml"

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.k_n = float(self.config["k_n"])
        self.u_min = float(self.config["u_min"])
        self.u_max = float(self.config["u_max"])

        self.positions = np.array(self.config["thruster_positions"], float)
        self.directions = np.array(self.config["thruster_directions"], float)
        self.ho_to_model = np.array(self.config["ho_to_model"], int)
        self.sinais = np.array(self.config["sinais"], float)

        self.num_thrusters = len(self.positions)
        self.B = self._build_allocation_matrix()

    def _build_allocation_matrix(self):
        T = np.zeros((6, self.num_thrusters))
        for i in range(self.num_thrusters):
            T[0:3, i] = self.directions[i]
            T[3:6, i] = np.cross(self.positions[i], self.directions[i])
        return self.k_n * T[:, self.ho_to_model] * self.sinais