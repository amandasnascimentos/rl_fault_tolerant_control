import numpy as np
import yaml
from pathlib import Path

class GenericThrusterAllocator:
    def __init__(self, config_path=None):
        # Se nenhum caminho for passado, aponta automaticamente para o yaml na pasta config
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "bluerov2_config.yaml"

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.k_n = self.config.get("k_n", 40.0)
        self.u_max = self.config.get("u_max", 10.0)
        
        self.positions = np.array(self.config["thruster_positions"])
        self.directions = np.array(self.config["thruster_directions"])
        self.ho_to_model = np.array(self.config["ho_to_model"])
        self.sinais = np.array(self.config["sinais"], float)
        
        self.num_thrusters = len(self.positions)
        self.B = self._build_allocation_matrix()

    def _build_allocation_matrix(self):
        T = np.zeros((6, self.num_thrusters))
        for i in range(self.num_thrusters):
            d = self.directions[i]
            p = self.positions[i]
            T[0:3, i] = d
            T[3:6, i] = np.cross(p, d)
            
        return self.k_n * T[:, self.ho_to_model] * self.sinais

    def allocate(self, tau):
        u = np.linalg.pinv(self.B) @ tau
        m = np.max(np.abs(u))
        if m > self.u_max:
            u = u * (self.u_max / m)
        return u