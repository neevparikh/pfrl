from copy import deepcopy
from logging import getLogger

import torch
from torch import nn
from torch.nn import init
import numpy as np
import gym

from pfrl.initializers import init_xavier_uniform
from pfrl.nn import EmpiricalNormalization, Lambda
from pfrl.env import VectorEnv, Env


class RNDModel(torch.nn.Module):
    """
    Torch RND architecture taken from https://github.com/jcwleo/random-network-distillation-pytorch/blob/master/model.py
    """
    def __init__(self, input_size, frame_stack=4, output_size=512):
        super(RNDModel, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        feature_output = 7 * 7 * 64
        self.predictor = nn.Sequential(
            nn.Conv2d(in_channels=frame_stack, out_channels=32, kernel_size=8, stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            Lambda(lambda x: x.view(x.size(0), -1)),
            nn.Linear(feature_output, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512))

        self.target = nn.Sequential(
            nn.Conv2d(in_channels=frame_stack, out_channels=32, kernel_size=8, stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            Lambda(lambda x: x.view(x.size(0), -1)),
            nn.Linear(feature_output, 512))

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

    def forward(self):
        raise NotImplementedError(
            "Should not call forward on this module, see module.predictor, module.target")


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
    def __init__(
            self,
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

        self.target = model.target
        self.predictor = model.predictor  # deepcopy(self.target)

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

        intrinsic_reward = torch.nn.functional.mse_loss(predicted_vector,
                                                        target_vector,
                                                        reduction='mean')
        intrinsic_reward = intrinsic_reward.unsqueeze(0)  # dim: 1,
        intrinsic_reward = self.reward_normalizer(intrinsic_reward)

        loss = intrinsic_reward.mean(dim=0)

        if log:
            self.logger.debug(
                'int_rew: %f, rnd_loss: %f, mean: %f, std: %f',
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
