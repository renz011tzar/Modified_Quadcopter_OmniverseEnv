import math
import numpy as np
import torch
from omni.isaac.core.objects import DynamicSphere
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.torch.rotations import *
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.crazyflie import Crazyflie
from omniisaacgymenvs.robots.articulations.views.crazyflie_view import CrazyflieView
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6  # small constant to avoid divisions by 0 and log(0)

class CrazyflieTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        self.update_config(sim_config)

        self._num_observations = 18
        self._num_actions = 4

        self._crazyflie_position = torch.tensor([0, 0, 1.0])
        self._ball_position = torch.tensor([0, 0, 1.0])

        RLTask.__init__(self, name=name, env=env)

        # Initialize the hash NN
        self.fcuri = self.HashNN(input_dim=2 * (1))  # Adjust input_dim as per the total dimension of o_curi
        self.fcuri.to(self._device)
        self.fcuri.eval()  # Set the network to evaluation mode
        for param in self.fcuri.parameters():
            param.requires_grad = False  # Freeze the network parameters

        return

    class HashNN(nn.Module):
        def __init__(self, input_dim):
            super(CrazyflieTask.HashNN, self).__init__()
            self.fc1 = nn.Linear(input_dim, 32)
            self.fc2 = nn.Linear(32, 64)
            self.fc3 = nn.Linear(64, 64)
            self.fc4 = nn.Linear(64, 32)
            self.fc5 = nn.Linear(32, 5)
        
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = F.relu(self.fc4(x))
            x = self.fc5(x)
            return x
        
    def bin2dec(self, binary_array):
        decimal_values = torch.zeros(binary_array.shape[0], device=binary_array.device, dtype=torch.int64)
        for i in range(binary_array.shape[1]):
            decimal_values += binary_array[:, i].int() * (2 ** (binary_array.shape[1] - 1 - i))
        return decimal_values
    
    def normalize_observations(self, observations, max_range):
        normalized = (observations / max_range) * torch.pi
        normalized_sin = torch.sin(normalized).unsqueeze(-1)
        normalized_cos = torch.cos(normalized).unsqueeze(-1)
        normalized_sin_cos = torch.cat([normalized_sin, normalized_cos], dim=-1)
        return normalized_sin_cos
    
    def update_bucket_occurrences(self, bucket_ids):
        bucket_ids = bucket_ids.unsqueeze(1)
        updates = torch.ones_like(bucket_ids, device=self._device, dtype=torch.int64)
        self.episode_sums["bucket_occurrences"].scatter_add_(1, bucket_ids, updates)
    
    def update_config(self, sim_config):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config
        self._num_states=3

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._max_episode_length = self._task_cfg["env"]["maxEpisodeLength"]

        self.dt = self._task_cfg["sim"]["dt"]

        # parameters for the crazyflie
        self.arm_length = 0.05

        # parameters for the controller
        self.motor_damp_time_up = 0.15
        self.motor_damp_time_down = 0.15

        # I use the multiplier 4, since 4*T ~ time for a step response to finish, where
        # T is a time constant of the first-order filter
        self.motor_tau_up = 4 * self.dt / (self.motor_damp_time_up + EPS)
        self.motor_tau_down = 4 * self.dt / (self.motor_damp_time_down + EPS)

        # thrust max
        self.mass = 0.028
        self.thrust_to_weight = 7.5

        self.motor_assymetry = np.array([1.0, 1.0, 1.0, 1.0])
        # re-normalizing to sum-up to 4
        self.motor_assymetry = self.motor_assymetry * 4.0 / np.sum(self.motor_assymetry)

        self.grav_z = -1.0 * self._task_cfg["sim"]["gravity"][2]

    def set_up_scene(self, scene) -> None:
        self.get_crazyflie()
        self.get_target()
        RLTask.set_up_scene(self, scene)
        self._copters = CrazyflieView(prim_paths_expr="/World/envs/.*/Crazyflie", name="crazyflie_view")
        self._balls = RigidPrimView(prim_paths_expr="/World/envs/.*/ball", name="ball_view")
        scene.add(self._copters)
        scene.add(self._balls)
        for i in range(4):
            scene.add(self._copters.physics_rotors[i])
        return

    def initialize_views(self, scene):
        super().initialize_views(scene)
        if scene.object_exists("crazyflie_view"):
            scene.remove_object("crazyflie_view", registry_only=True)
        if scene.object_exists("ball_view"):
            scene.remove_object("ball_view", registry_only=True)
        for i in range(1, 5):
            scene.remove_object(f"m{i}_prop_view", registry_only=True)
        self._copters = CrazyflieView(prim_paths_expr="/World/envs/.*/Crazyflie", name="crazyflie_view")
        self._balls = RigidPrimView(prim_paths_expr="/World/envs/.*/ball", name="ball_view")
        scene.add(self._copters)
        scene.add(self._balls)
        for i in range(4):
            scene.add(self._copters.physics_rotors[i])

    def get_crazyflie_positions(self):
        positions = torch.zeros((self._num_envs, 3), device=self._device)
        
        # Assign [0, 0, 1] to the first 1024 elements
        positions[:1024] = torch.tensor([0, 0, 1.0], device=self._device)
        
        # Assign [0, 0, 3] to the remaining elements
        positions[1024:] = torch.tensor([0, 0, 3.0], device=self._device)
        
        return positions
    
    def get_crazyflie(self):
        copter = Crazyflie(
            prim_path=self.default_zero_env_path + "/Crazyflie", name="crazyflie", translation=self._crazyflie_position
        )
        
        self._sim_config.apply_articulation_settings(
            "crazyflie", get_prim_at_path(copter.prim_path), self._sim_config.parse_actor_config("crazyflie")
        )

    def get_target(self):
        radius = 0.01
        color = torch.tensor([1, 0, 0])
        ball = DynamicSphere(
            prim_path=self.default_zero_env_path + "/ball",
            translation=self._ball_position,
            name="target_0",
            radius=radius,
            color=color,
        )
        self._sim_config.apply_articulation_settings(
            "ball", get_prim_at_path(ball.prim_path), self._sim_config.parse_actor_config("ball")
        )
        ball.set_collision_enabled(False)

    def get_observations(self) -> dict:
        self.root_pos, self.root_rot = self._copters.get_world_poses(clone=False)
        self.root_velocities = self._copters.get_velocities(clone=False)

        root_positions = self.root_pos - self._env_pos
        root_quats = self.root_rot

        rot_x = quat_axis(root_quats, 0)
        rot_y = quat_axis(root_quats, 1)
        rot_z = quat_axis(root_quats, 2)

        root_linvels = self.root_velocities[:, :3]
        root_angvels = self.root_velocities[:, 3:]

        self.obs_buf[..., 0:3] = self.target_positions - root_positions

        self.obs_buf[..., 3:6] = rot_x
        self.obs_buf[..., 6:9] = rot_y
        self.obs_buf[..., 9:12] = rot_z

        self.obs_buf[..., 12:15] = root_linvels
        self.obs_buf[..., 15:18] = root_angvels

        observations = {self._copters.name: {"obs_buf": self.obs_buf}}
        return observations

    def pre_physics_step(self, actions) -> None:
        if not self.world.is_playing():
            return

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        set_target_ids = (self.progress_buf % 500 == 0).nonzero(as_tuple=False).squeeze(-1)
        if len(set_target_ids) > 0:
            self.set_targets(set_target_ids)

        actions = actions.clone().to(self._device)
        self.actions = actions

        # clamp to [-1.0, 1.0]
        thrust_cmds = torch.clamp(actions, min=-1.0, max=1.0)
        # scale to [0.0, 1.0]
        thrust_cmds = (thrust_cmds + 1.0) / 2.0
        # filtering the thruster and adding noise
        motor_tau = self.motor_tau_up * torch.ones((self._num_envs, 4), dtype=torch.float32, device=self._device)
        motor_tau[thrust_cmds < self.thrust_cmds_damp] = self.motor_tau_down
        motor_tau[motor_tau > 1.0] = 1.0

        # Since NN commands thrusts we need to convert to rot vel and back
        thrust_rot = thrust_cmds**0.5
        self.thrust_rot_damp = motor_tau * (thrust_rot - self.thrust_rot_damp) + self.thrust_rot_damp
        self.thrust_cmds_damp = self.thrust_rot_damp**2

        ## Adding noise
        thrust_noise = 0.01 * torch.randn(4, dtype=torch.float32, device=self._device)
        thrust_noise = thrust_cmds * thrust_noise
        self.thrust_cmds_damp = torch.clamp(self.thrust_cmds_damp + thrust_noise, min=0.0, max=1.0)

        # Determine max thrust based on conditions
        thrusts = self.thrust_max * self.thrust_cmds_damp

        # thrusts given rotation
        root_quats = self.root_rot
        rot_x = quat_axis(root_quats, 0)
        rot_y = quat_axis(root_quats, 1)
        rot_z = quat_axis(root_quats, 2)
        rot_matrix = torch.cat((rot_x, rot_y, rot_z), 1).reshape(-1, 3, 3)

        force_x = torch.zeros(self._num_envs, 4, dtype=torch.float32, device=self._device)
        force_y = torch.zeros(self._num_envs, 4, dtype=torch.float32, device=self._device)
        force_xy = torch.cat((force_x, force_y), 1).reshape(-1, 4, 2)
        thrusts = thrusts.reshape(-1, 4, 1)
        thrusts = torch.cat((force_xy, thrusts), 2)

        thrusts_0 = thrusts[:, 0]
        thrusts_0 = thrusts_0[:, :, None]

        thrusts_1 = thrusts[:, 1]
        thrusts_1 = thrusts_1[:, :, None]

        thrusts_2 = thrusts[:, 2]
        thrusts_2 = thrusts_2[:, :, None]

        thrusts_3 = thrusts[:, 3]
        thrusts_3 = thrusts_3[:, :, None]

        mod_thrusts_0 = torch.matmul(rot_matrix, thrusts_0)
        mod_thrusts_1 = torch.matmul(rot_matrix, thrusts_1)
        mod_thrusts_2 = torch.matmul(rot_matrix, thrusts_2)
        mod_thrusts_3 = torch.matmul(rot_matrix, thrusts_3)

        self.thrusts[:, 0] = torch.squeeze(mod_thrusts_0)
        self.thrusts[:, 1] = torch.squeeze(mod_thrusts_1)
        self.thrusts[:, 2] = torch.squeeze(mod_thrusts_2)
        self.thrusts[:, 3] = torch.squeeze(mod_thrusts_3)

        # clear actions for reset envs
        self.thrusts[reset_env_ids] = 0

        # spin spinning rotors
        prop_rot = self.thrust_cmds_damp * self.prop_max_rot
        self.dof_vel[:, 0] = prop_rot[:, 0]
        self.dof_vel[:, 1] = -1.0 * prop_rot[:, 1]
        self.dof_vel[:, 2] = prop_rot[:, 2]
        self.dof_vel[:, 3] = -1.0 * prop_rot[:, 3]

        self._copters.set_joint_velocities(self.dof_vel)

        # apply actions
        for i in range(4):
            self._copters.physics_rotors[i].apply_forces(self.thrusts[:, i], indices=self.all_indices)

    def post_reset(self):
        thrust_max = self.grav_z * self.mass * self.thrust_to_weight * self.motor_assymetry / 4.0
        self.thrusts = torch.zeros((self._num_envs, 4, 3), dtype=torch.float32, device=self._device)
        self.thrust_cmds_damp = torch.zeros((self._num_envs, 4), dtype=torch.float32, device=self._device)
        self.thrust_rot_damp = torch.zeros((self._num_envs, 4), dtype=torch.float32, device=self._device)
        self.thrust_max = torch.tensor(thrust_max, device=self._device, dtype=torch.float32)

        self.motor_linearity = 1.0
        self.prop_max_rot = 433.3

        self.target_positions = torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float32)
        self.target_positions[:, 2] = 3

        self.target_quats = torch.zeros((self._num_envs, 4), device=self._device, dtype=torch.float32)
        self.target_quats[:, 2] = 1

        self.actions = torch.zeros((self._num_envs, self.num_actions), device=self._device, dtype=torch.float32)

        self.all_indices = torch.arange(self._num_envs, dtype=torch.int32, device=self._device)

        # Extra info
        self.extras = {}

        torch_zeros = lambda: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.episode_sums = {
            "curiosity_reward": torch_zeros(),
            "successful_flip_count": torch_zeros(),
            "states_visited": torch_zeros(),
            "task_reward": torch_zeros(),
            "bucket_occurrences": torch.zeros((self._num_envs, 32), dtype=torch.int64, device=self._device),
            "states_count": torch.zeros((self._num_envs, 3), dtype=torch.int64, device=self._device),
            "hovering_reward":torch_zeros(),
            "flipping_reward":torch_zeros(),
            "approaching_target_reward":torch_zeros(),
        }

        self.root_pos, self.root_rot = self._copters.get_world_poses()
        self.root_velocities = self._copters.get_velocities()
        self.dof_pos = self._copters.get_joint_positions()
        self.dof_vel = self._copters.get_joint_velocities()

        self.initial_ball_pos, self.initial_ball_rot = self._balls.get_world_poses(clone=False)
        self.initial_root_pos, self.initial_root_rot = self.root_pos.clone(), self.root_rot.clone()

        # control parameters
        self.thrusts = torch.zeros((self._num_envs, 4, 3), dtype=torch.float32, device=self._device)
        self.thrust_cmds_damp = torch.zeros((self._num_envs, 4), dtype=torch.float32, device=self._device)
        self.thrust_rot_damp = torch.zeros((self._num_envs, 4), dtype=torch.float32, device=self._device)

        self.set_targets(self.all_indices)

    def set_targets(self, env_ids):
        num_sets = len(env_ids)
        envs_long = env_ids.long()
        # shift the target up so it visually aligns better
        ball_pos = self.target_positions[envs_long] + self._env_pos[envs_long]
        ball_pos[:, 2] += 0.0
        self._balls.set_world_poses(ball_pos[:, 0:3], self.initial_ball_rot[envs_long].clone(), indices=env_ids)

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)

        self.dof_pos[env_ids, :] = torch_rand_float(-0.0, 0.0, (num_resets, self._copters.num_dof), device=self._device)
        self.dof_vel[env_ids, :] = 0

        root_pos = self.initial_root_pos.clone()
        root_pos[env_ids, 0] += torch_rand_float(-0.0, 0.0, (num_resets, 1), device=self._device).view(-1)
        root_pos[env_ids, 1] += torch_rand_float(-0.0, 0.0, (num_resets, 1), device=self._device).view(-1)
        root_pos[env_ids, 2] += torch_rand_float(-0.0, 0.0, (num_resets, 1), device=self._device).view(-1)
        root_velocities = self.root_velocities.clone()
        root_velocities[env_ids] = 0

        # apply resets
        self._copters.set_joint_positions(self.dof_pos[env_ids], indices=env_ids)
        self._copters.set_joint_velocities(self.dof_vel[env_ids], indices=env_ids)

        self._copters.set_world_poses(root_pos[env_ids], self.initial_root_rot[env_ids].clone(), indices=env_ids)
        self._copters.set_velocities(root_velocities[env_ids], indices=env_ids)

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

        self.thrust_cmds_damp[env_ids] = 0
        self.thrust_rot_damp[env_ids] = 0

        # Fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            # Convert the tensor to float before calculating the mean
            self.extras["episode"][key] = torch.mean(self.episode_sums[key][env_ids].float()) / self._max_episode_length
            self.episode_sums[key][env_ids] = 0.0

    def _check_flip_completion(self):
        root_quats = self.root_rot
        # Update the position reached indicator
        successful_flip_threshold = 0.05  # Stricter threshold for flip completion
        flip_completion_0 = (torch.abs(root_quats[:, 2]-1) < successful_flip_threshold).float()

        # Count successful flips
        flip_state=(self.states_buf[:, 1] == 1)
        flip_completion=torch.where(flip_state, flip_completion_0, torch.zeros_like(flip_completion_0))
        self.episode_sums["successful_flip_count"] += flip_completion
        num_flips = self.episode_sums["successful_flip_count"]

        # Check if the number of flips reached 2
        double_flip = (num_flips >= 2)

        # Reset successful flip count if hover phase is reached
        self.episode_sums["successful_flip_count"] = torch.where(
            double_flip,
            torch.zeros((self._num_envs), device=self._device, dtype=torch.float32),
            self.episode_sums["successful_flip_count"]
        )
        return double_flip
    
    def _calculate_approaching_target_reward(self, target_positions, root_positions, root_angvels, root_linvels, root_quats, time_in_state_1):
        position_temp = 1.0  # Lower temperature to widen the effective range
        spin_temp = 1.0

        # pos reward
        target_dist = torch.norm(target_positions - root_positions, dim=-1)
        position_error = target_dist
        position_reward = torch.exp(-position_temp * position_error ** 2)
        
        spin = torch.abs(root_angvels).sum(-1)
        spin_reward = torch.exp(-spin_temp * spin)

        ups = quat_axis(root_quats, 2)
        self.orient_z = ups[..., 2]
        up_reward = torch.clamp(ups[..., 2], min=0.0, max=1.0)

        # combined reward
        reward = position_reward + up_reward + spin_reward

        approaching_target_reward=reward

        return approaching_target_reward
    

    def _calculate_flipping_reward(self, target_positions, root_positions, root_angvels, root_linvels, root_quats, time_in_state_2):
        flip_temp = 0.05     # Lower temperature to reduce sensitivity to exact angular velocity
        quat_temp = 0.1

        position_temp = 1.0  # Lower temperature to widen the effective range

        # pos reward
        target_dist = torch.norm(target_positions - root_positions, dim=-1)
        position_error = target_dist
        position_reward = torch.exp(-position_temp * position_error ** 2)

        stability_penalty = torch.norm(root_linvels, dim=-1) 
        stability_reward = torch.exp(-position_temp * stability_penalty ** 2)

        '''
        # Phase 2: Flip reward
        target_angvel_y = 5  # Reduced target angular velocity threshold to make it achievable
        angvel_error = torch.abs(root_angvels[..., 1] - target_angvel_y)
        flip_reward = torch.exp(-flip_temp * angvel_error ** 2)

        target_quat = self.target_quats
        quat_variation = torch.norm(root_quats - target_quat, dim=-1)
        quaternion_reward = torch.exp(-quat_temp * quat_variation ** 2)
        '''

        flipping_reward=position_reward+stability_reward

        return flipping_reward
    
    def _calculate_hovering_reward(self, target_positions, root_positions, root_angvels, root_linvels, root_quats, time_in_state_3):
        position_temp = 1.0  # Lower temperature to widen the effective range
        spin_temp = 0.1

        # pos reward
        target_dist = torch.norm(target_positions - root_positions, dim=-1)
        position_error = target_dist
        position_reward = torch.exp(-position_temp * position_error ** 2)

        stability_penalty = torch.norm(root_linvels, dim=-1) 
        stability_reward = torch.exp(-position_temp * stability_penalty ** 2)
        
        spin = torch.abs(root_angvels).sum(-1)
        spin_reward = 10*torch.exp(-spin_temp * spin)

        ups = quat_axis(root_quats, 2)
        up_reward = torch.clamp(ups[..., 2], min=0.0, max=1.0)

        # combined reward
        reward = position_reward + up_reward + spin_reward

        hovering_reward=reward

        return hovering_reward+stability_reward

    def get_ID_quaternion_w(self, input):
        quaternions_normalized = self.normalize_observations(input, math.pi)

        # Compute the hash
        with torch.no_grad():
            hash_values = self.fcuri(quaternions_normalized)

        # Convert hash values to boolean
        hash_bool = hash_values > 0

        # Convert boolean to decimal (BIN2DEC)
        bucket_ids = self.bin2dec(hash_bool)

        return bucket_ids

    def get_curiosity_reward(self, bucket_ids):
        # Get the number of visits for each bucket ID
        visits = self.episode_sums["bucket_occurrences"].gather(1, bucket_ids.unsqueeze(1)).squeeze(1)
        # Calculate the curiosity reward
        r_curi = 1.0 / visits.float()**4 
        return r_curi
    
    def get_count(self,state_boolean_1, state_boolean_2, state_boolean_3):
        self.episode_sums["states_count"][:, 0] += state_boolean_1.int()
        self.episode_sums["states_count"][:, 1] += state_boolean_2.int()
        self.episode_sums["states_count"][:, 2] += state_boolean_3.int()

        time_in_state_1 = self.episode_sums["states_count"][:, 0].float()
        time_in_state_2 = self.episode_sums["states_count"][:, 1].float()
        time_in_state_3 = self.episode_sums["states_count"][:, 2].float()

        return time_in_state_1,time_in_state_2,time_in_state_3
    
    def task_reward(self, state_boolean_2, state_boolean_3):
        task_reward = torch.zeros(self._num_envs, device=self._device, dtype=torch.float)
        task_reward_1 = torch.where(state_boolean_2, 10*torch.ones(self._num_envs, device=self._device, dtype=torch.float), task_reward)
        task_reward_2 = torch.where(state_boolean_3, 200*torch.ones(self._num_envs, device=self._device, dtype=torch.float), task_reward)

        return task_reward_1, task_reward_2
    
    def get_states(self):
        root_positions = self.root_pos - self._env_pos
        target_positions = torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float32)
        target_positions[:, 2] = 3
        target_dist = torch.norm(target_positions - root_positions, dim=-1)
        state_1=torch.tensor([1, 0, 0], device=self._device, dtype=torch.float)
        state_2=torch.tensor([0, 1, 0], device=self._device, dtype=torch.float)
        state_3=torch.tensor([0, 0, 1], device=self._device, dtype=torch.float)
        
        initial= (self.states_buf[:, 0] == 0)&(self.states_buf[:, 1] == 0)&(self.states_buf[:, 2] == 0)
        self.states_buf[initial] = state_1
        
        approaching_target_mask = (target_dist < 0.5)&(self.states_buf[:, 0] == 1)
        self.states_buf[approaching_target_mask] = state_2

        flip_completed = self._check_flip_completion()
        flipping_mask = (self.states_buf[:, 0] == 0) & (self.states_buf[:, 1] == 1) & flip_completed
        self.states_buf[flipping_mask] = state_3

        return self.states_buf
    
    def calculate_metrics(self):
        root_positions = self.root_pos - self._env_pos
        root_angvels = self.root_velocities[:, 3:]
        root_linvels = self.root_velocities[:, :3]
        root_quats = self.root_rot

        target_positions = torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float32)
        #target_positions[:, 1] = 1
        target_positions[:, 2] = 3
        self.target_positions=target_positions

        target_dist = torch.norm(target_positions - root_positions, dim=-1)
        self.target_dist = target_dist
        self.root_positions = root_positions

        # State Transition Logic
        state_boolean_1 = (self.states_buf[:, 0]== 1)
        state_boolean_2 = (self.states_buf[:, 1] == 1)
        state_boolean_3 = (self.states_buf[:, 2] == 1)

        time_in_state_1,time_in_state_2,time_in_state_3=self.get_count(state_boolean_1, state_boolean_2, state_boolean_3)
        task_reward_1, task_reward_2= self.task_reward(state_boolean_2, state_boolean_3)

        # Calculate rewards based on the current state
        approaching_target_reward = self._calculate_approaching_target_reward(target_positions, root_positions, root_angvels, root_linvels, root_quats,time_in_state_1)
        flipping_reward = self._calculate_flipping_reward(target_positions, root_positions, root_angvels, root_linvels, root_quats, time_in_state_2)
        hovering_reward = self._calculate_hovering_reward(target_positions, root_positions, root_angvels, root_linvels, root_quats, time_in_state_3)

        zeros=torch.zeros(self._num_envs, device=self._device, dtype=torch.float32)
        approaching_target_reward = torch.where(state_boolean_1, approaching_target_reward, zeros)
        flipping_reward = torch.where(state_boolean_2, flipping_reward, zeros)
        hovering_reward = torch.where(state_boolean_3, hovering_reward, zeros)

        self.episode_sums["approaching_target_reward"]+=approaching_target_reward
        self.episode_sums["flipping_reward"]+=flipping_reward
        self.episode_sums["hovering_reward"]+=hovering_reward

        states= self.states_buf.clone()
        states_visited= states.sum(-1)

        # Update rewards based on state buffer
        self.rew_buf[:]  = approaching_target_reward+task_reward_1*flipping_reward+task_reward_2*hovering_reward 

        success_reward=zeros
        end_on_state_3=(self.progress_buf==self._max_episode_length - 2)&(state_boolean_3)
        success_reward=torch.where(end_on_state_3, 100000*torch.ones(self._num_envs, device=self._device, dtype=torch.float), success_reward)
        self.rew_buf[:] += success_reward

        curiosity_vector=root_quats[:,2]
        # Convert boolean to decimal (BIN2DEC)
        self.bucket_ids = self.get_ID_quaternion_w(curiosity_vector)
        # Update bucket occurrences
        self.update_bucket_occurrences(self.bucket_ids)
        curiosity_reward = self.get_curiosity_reward(self.bucket_ids)
        self.rew_buf[:]+= 10000*curiosity_reward

        self.episode_sums["curiosity_reward"]+=curiosity_reward

        self.episode_sums["states_visited"]+=states_visited

        print(torch.sum(self.states_buf, dim=0))

    def is_done(self) -> None:
        # Resets due to misbehavior
        ones = torch.ones_like(self.reset_buf)
        die = torch.zeros_like(self.reset_buf)
        die = torch.where(self.target_dist > 5.0, ones, die)

        # z >= 0.5 & z <= 7.0
        die = torch.where(self.root_positions[..., 2] < 0.5, ones, die)
        die = torch.where(self.root_positions[..., 2] > 7.0, ones, die)

        '''
        check=(self.states_buf[:,0]==1)&(self.progress_buf==500)
        check_state_2=(self.states_buf[:,1]==1)&(self.progress_buf==1000)
        die=torch.where(check, ones, die)
        die=torch.where(check_state_2, ones, die)
        '''

        # Resets due to episode length
        episode_end = torch.where(self.progress_buf >= self._max_episode_length - 1, ones, die)

        # Apply resets
        self.reset_buf[:] = episode_end

        reset_indices = (self.reset_buf == 1)
        
        if (self._num_envs<5000):
            self.states_buf[reset_indices] = torch.tensor([1, 0, 0], device=self._device, dtype=torch.float)
        else:
            # Define the index ranges
            last_2000_start = self._num_envs - 3072
            last_1000_start = self._num_envs - 2048
            
            # Get the last 1000 indices
            last_1000_indices = torch.arange(last_1000_start, self._num_envs, device=self._device)
            
            # Update the states_buf for the last 1000 rows
            self.states_buf[last_1000_indices] = torch.where(
                reset_indices[last_1000_indices].unsqueeze(1),
                torch.tensor([0, 0, 1], device=self._device, dtype=torch.float),
                self.states_buf[last_1000_indices]
            )
            
            # Update the states_buf for the rows from -2000 to -1001
            indices_1000_2000 = torch.arange(last_2000_start, last_1000_start, device=self._device)
            self.states_buf[indices_1000_2000] = torch.where(
                reset_indices[indices_1000_2000].unsqueeze(1),
                torch.tensor([0, 1, 0], device=self._device, dtype=torch.float),
                self.states_buf[indices_1000_2000]
            )

            # Update the states_buf for the remaining rows
            remaining_indices = torch.arange(0, last_2000_start, device=self._device)
            self.states_buf[remaining_indices] = torch.where(
                reset_indices[remaining_indices].unsqueeze(1),
                torch.tensor([1, 0, 0], device=self._device, dtype=torch.float),
                self.states_buf[remaining_indices]
            )
    
