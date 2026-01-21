# utils.py
import os
from dataclasses import dataclass

@dataclass
class Paths:
    root: str
    runs: str
    yaml_out: str

    @staticmethod
    def make(root: str) -> "Paths":
        runs = os.path.join(root, "runs")
        yaml_out = os.path.join(root, "yolov8n_mbv3_pan_p2.yaml")
        os.makedirs(runs, exist_ok=True)
        return Paths(root, runs, yaml_out)
