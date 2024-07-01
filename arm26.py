import mujoco
import time
from mujoco import viewer
import os
import gym
import numpy as np
from dep_controller import DEP
import skvideo.io

FRAMESKIP = 2
FRAMERATE = 30

# offline without sphere drawing so far
OFFLINE_RENDER = False
curr_dir = os.path.dirname(os.path.realpath(__file__))


def save_video(frames):
    save_path = os.path.join(curr_dir, "rendered_video.mp4")
    print(f"Saving video to: {save_path}")
    skvideo.io.vwrite(
        save_path,
        np.asarray(frames),
        outputdict={"-pix_fmt": "yuv420p"},
        inputdict={"-r": str(FRAMERATE)},
    )


class Arm26:
    def __init__(self):
        curr_dir = os.path.dirname(__file__)
        xml_path = os.path.join(curr_dir, "./assets/arm26.xml")
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.init_qpos = self.mj_data.qpos.copy()
        nactions = self.mj_model.nu
        self.action_space = gym.spaces.Box(-np.ones((nactions,)), np.ones((nactions,)))
        self.observation_space = gym.spaces.Box(-np.ones((5,)), np.ones((5,)))
        self.init = False
        self.init_visuals = False

    def generate_target(self):
        self.target = np.random.uniform(0.2, 0.5, size=(2,))

    def draw_sphere(self):
        if hasattr(self, "viewer"):
            if self.viewer.user_scn.ngeom >= self.viewer.user_scn.maxgeom:
                return
            if not self.init_visuals:
                self.viewer.user_scn.ngeom += 1  # increment ngeom
                # initialise a new sphere, add it to the scene using mjv_makeConnector
                mujoco.mjv_initGeom(
                    self.viewer.user_scn.geoms[i],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[0.02, 0, 0],
                    pos=np.array([self.target[0], self.target[1], 0]),
                    mat=np.eye(3).flatten(),
                    rgba=np.array([1.0, 0.1, 0.1, 1.0]),
                )
                self.init_visuals = True
            else:
                self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom - 1].pos[:] = (
                    np.array([self.target[0], self.target[1], 0.0])
                )

            self.viewer.sync()

    def reset(self):
        self.draw_sphere()
        self.generate_target()
        self.mj_data.qpos[:] = self.init_qpos + np.random.normal(0, 0.2)
        self.mj_data.ctrl[:] = 0.0
        self.mj_data.act[:] = np.random.uniform(size=self.mj_model.nu)
        self.init = True
        return self.compute_obs()

    def compute_obs(self):
        np.concatenate(
            [
                self.mj_data.qpos.copy(),
                self.mj_data.qvel.copy(),
                self.muscle_length(),
                self.muscle_velocity(),
                self.muscle_activity(),
                self.muscle_force(),
                self.target.copy(),
            ],
            dtype=np.float32,
        ).copy()

    def step(self, action):
        if not self.init:
            raise Exception("Reset has to be called once before step")
        self.mj_data.ctrl[:] = action
        for i in range(FRAMESKIP):
            mujoco.mj_step(self.mj_model, self.mj_data)
        done = 0
        reward = 0
        return self.compute_obs(), reward, done, {}

    def render(self):
        if not OFFLINE_RENDER:
            if not hasattr(self, "viewer"):
                self.viewer = viewer.launch_passive(
                    self.mj_model, self.mj_data, show_left_ui=False, show_right_ui=False
                )
                self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True
                self.viewer.sync()
            with self.viewer.lock():
                self.draw_sphere()
            self.viewer.sync()
            time.sleep(0.01)

        else:
            if not hasattr(self, "renderer"):
                self.renderer = mujoco.Renderer(
                    self.mj_model,
                    height=420,
                    width=640,
                )
                self.frames = []
            self.renderer.update_scene(self.mj_data, camera="camera_view")
            frame = self.renderer.render()
            self.frames.append(frame)

    def muscle_length(self):
        return self.mj_data.actuator_length.copy()

    def muscle_velocity(self):
        return self.mj_data.actuator_velocity.copy()

    def muscle_force(self):
        return self.mj_data.actuator_force.copy()

    def muscle_activity(self):
        return self.mj_data.act.copy()


if __name__ == "__main__":
    env = Arm26()

    dep = DEP()
    dep.initialize(env.observation_space, env.action_space)

    for ep in range(2):
        env.reset()
        for i in range(100):
            action = dep.step(env.muscle_length())
            env.step(action)
            env.render()

    if OFFLINE_RENDER:
        save_video(env.frames)
