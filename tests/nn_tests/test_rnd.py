import pytest
import torch
import numpy as np

from pfrl.envs.abc import ABC
from pfrl.nn import RND

make_random_episodes = ABC.make_random_episodes

@pytest.mark.parametrize("init_steps", [10])
@pytest.mark.parametrize("update_params", [True, False])
@pytest.mark.parametrize("log", [True, False])
class TestRND:
    @pytest.fixture(autouse=True)
    def setUp(self):
        self.env = ABC(deterministic=True)
        self.obs_shape = self.env.observation_space.shape

        self.model = torch.nn.Sequential(
                torch.nn.Linear(self.obs_shape[0], 16), torch.nn.ReLU(),
                torch.nn.Linear(16, 16), torch.nn.ReLU(),
                torch.nn.Linear(16, 3),
            )
    
    def _test_rnd(self, rnd_module, update_params, log):
        rnd_module.initialize_normalizer(self.env)
        ep = make_random_episodes(1, self.obs_shape[0], self.env.action_space.n)[0]
        states = torch.cat([torch.from_numpy(t['state']).unsqueeze(0).to(dtype=torch.float) for t in ep])
        states = states.to(device=rnd_module.device)
        rnd_module(states, update_params=update_params, log=log)

    def test_rnd(self, init_steps, update_params, log):
        rnd_module = RND(self.model, self.obs_shape, init_steps, gpu=None)
        self._test_rnd(rnd_module, update_params, log)

    @pytest.mark.gpu
    def test_rnd_gpu(self, init_steps, update_params, log):
        rnd_module = RND(self.model, self.obs_shape, init_steps, gpu=0)
        self._test_rnd(rnd_module, update_params, log)
