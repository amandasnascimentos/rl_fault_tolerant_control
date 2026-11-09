import holoocean

def get_default_scenario():
    return {
        "name": "alocacao_pseudoinversa",
        "world": "SimpleUnderwater",
        "package_name": "Ocean",
        "main_agent": "auv0",
        "control_scheme": 0,
        "ticks_per_sec": 60,
        "agents": [{
            "agent_name": "auv0",
            "agent_type": "BlueROV2",
            "control_scheme": 0,
            "location": [0.0, 0.0, -10.0],
            "sensors": [{"sensor_type": "PoseSensor", "sensor_name": "Pose"}],
        }],
    }

class HoloOceanBlueROVWrapper:
    def __init__(self, scenario_cfg=None):
        self.scenario_cfg = scenario_cfg or get_default_scenario()
        self.env = None

    def __enter__(self):
        self.env = holoocean.make(scenario_cfg=self.scenario_cfg)
        return self.env

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.env is not None:
            self.env.__exit__(exc_type, exc_val, exc_tb)