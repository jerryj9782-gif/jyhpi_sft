"""Optional NAVSIM v1 adapter. Requires an independently installed NAVSIM/nuPlan."""
import numpy as np
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SensorConfig, Trajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from .policy import NavPiPolicy


class NavPiAgent(AbstractAgent):
    def __init__(self, checkpoint, device="cuda", steps=10, history_frames=4, seed=42):
        super().__init__(requires_scene=False)
        self.checkpoint = checkpoint
        self.device_name = device
        self.steps = steps
        self.history_frames = history_frames
        self.seed = seed
        self.policy = None

    def name(self):
        return "NavPiThreeCamera"

    def initialize(self):
        self.policy = NavPiPolicy(self.checkpoint, self.device_name, self.steps)

    def get_sensor_config(self):
        config = SensorConfig.build_no_sensors()
        config.cam_f0 = config.cam_l0 = config.cam_r0 = [self.history_frames - 1]
        return config

    def compute_trajectory(self, agent_input):
        if self.policy is None:
            self.initialize()
        ego, cameras = agent_input.ego_statuses[-1], agent_input.cameras[-1]
        state = [float(np.linalg.norm(ego.ego_velocity)), *ego.ego_velocity, *ego.ego_acceleration]
        images = {"cam_front_center": cameras.cam_f0.image,
                  "cam_front_left": cameras.cam_l0.image,
                  "cam_front_right": cameras.cam_r0.image}
        prompt = "Drive safely. Recorded navigation command: " + " ".join(map(str, np.asarray(ego.driving_command, int)))
        trajectory = self.policy.predict(images, state, prompt, seed=self.seed)
        return Trajectory(trajectory, TrajectorySampling(time_horizon=4, interval_length=.5))
