from copy import deepcopy
from logging import getLogger

import torch
import numpy as np
import gym

from pfrl.initializers import init_xavier_uniform
from pfrl.nn import EmpiricalNormalization
from pfrl.env import VectorEnv, Env

class RND(torch.nn.Module):
    """Implements Random Network Distillation.

    Args:
        model (nn.Module): Body of model, must predict a vector of some size, given a state
        obs_shape (int or tuple of int): Shape of input values except batch axis.
        reward_shape (int or tuple of int): Shape of reward values except batch axis.
        init_steps (int): How long to initialize the normalizers
        gpu (int): GPU device id if not None nor negative
        optimizer (torch.optim.Optimizer): Optimizer used to train the model, default is Adam
    """
    def __init__(self,
            model,
            obs_shape,
            init_steps,
            gpu=None,
            optimizer=None,
            logger=getLogger(__name__),
    ):
        super(RND, self).__init__()
        self.init_steps = init_steps
        self.obs_shape = obs_shape

        self.target = model
        self.predictor = deepcopy(self.target)

        self.target.apply(init_xavier_uniform)
        self.predictor.apply(init_xavier_uniform)

        self.target.eval()

        self.logger = logger

        self.obs_normalizer = EmpiricalNormalization(obs_shape, clip_threshold=5.0)
        self.reward_normalizer = EmpiricalNormalization(1, clip_threshold=np.inf)

        if gpu is not None and gpu >= 0:
            assert torch.cuda.is_available()
            self.device = torch.device("cuda:{}".format(gpu))
        else:
            self.device = torch.device("cpu")

        self.target.to(self.device)
        self.predictor.to(self.device)
        self.obs_normalizer.to(self.device)
        self.reward_normalizer.to(self.device)

        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.predictor.parameters())
        else:
            self.optimizer = optimizer

    def initialize_normalizer(self, env):
        env.reset()
        if isinstance(env, VectorEnv):
            for _ in range(self.init_steps):
                next_states, _, _, _ = env.step([env.action_space.sample() for _ in range(env.num_envs)])
                next_states = torch.cat(map(torch.from_numpy, next_states)).to(self.device)
                self.obs_normalizer(next_states)
        elif isinstance(env, (gym.Env, Env)):
            for _ in range(self.init_steps):
                next_state, _, _, _ = env.step(env.action_space.sample())
                next_state = torch.from_numpy(next_state).to(self.device)
                self.obs_normalizer(next_state)
        else:
            raise ValueError("{} env type not recognized".format(type(env)))

    def forward(self, states, update_params=False, log=True):
        states = self.obs_normalizer(states)

        predicted_vector = self.predictor(states)
        target_vector = self.target(states)

        intrinsic_reward = torch.nn.functional.mse_loss(predicted_vector, target_vector, reduction='mean')
        intrinsic_reward = intrinsic_reward.unsqueeze(0) # dim: 1,
        intrinsic_reward = self.reward_normalizer(intrinsic_reward)

        loss = intrinsic_reward.mean(dim=0)

        if log:
            self.logger.debug('int_rew: %f, rnd_loss: %f, mean: %f, std: %f',
                loss.item(),
                intrinsic_reward.mean().item(),
                self.reward_normalizer.mean.item(),
                self.reward_normalizer.std.item(),
            )

        if update_params:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return intrinsic_reward.detach()
