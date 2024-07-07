import unittest
import torch
import torch.nn as nn

from q_learning.environment import EnvTest
from q_learning.network import DQN, Linear, LinearSchedule, LinearExploration
from utils import read_config


class TestExploration(unittest.TestCase):
    def test_variety(self):
        env = EnvTest((5, 5, 1))
        exp = LinearExploration(env, 1, 0, 10)
        found_diff = False
        for i in range(10):
            rnd_act = exp.get_action(0)
            if rnd_act != 0 and rnd_act is not None:
                found_diff = True
        assert found_diff

    def test_eps_in_range(self):
        env = EnvTest((5, 5, 1))
        exp = LinearExploration(env, 1, 0, 10)
        exp.update(5)
        assert exp.epsilon == 0.5

    def test_eps_out_of_range(self):
        env = EnvTest((5, 5, 1))
        exp = LinearExploration(env, 1, 0.5, 10)
        exp.update(20)
        assert exp.epsilon == 0.5


class TestLinearDQN(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear_config = read_config('test_linear.yml')

    def test_config(self):
        env = EnvTest((5, 5, 1))
        model = Linear(env, self.linear_config)
        state_shape = list(env.observation_space.shape)
        img_height, img_width, n_channels = state_shape
        num_actions = env.action_space.n
        self.assertTrue(isinstance(model.q_network, nn.Linear))
        self.assertEqual(
            model.q_network.weight.size(),
            torch.Size(
                [
                    num_actions,
                    img_height
                    * img_width
                    * n_channels
                    * self.linear_config["hyper_params"]["state_history"],
                ]
            ),
        )
        self.assertTrue(isinstance(model.target_network, nn.Linear))
        self.assertEqual(
            model.target_network.weight.size(),
            torch.Size(
                [
                    num_actions,
                    img_height
                    * img_width
                    * n_channels
                    * self.linear_config["hyper_params"]["state_history"],
                ]
            ),
        )

    def test_loss(self):
        env = EnvTest((5, 5, 1))
        model = Linear(env, self.linear_config)
        state_shape = list(env.observation_space.shape)
        img_height, img_width, n_channels = state_shape
        num_actions = env.action_space.n

        nn.init.constant_(model.q_network.weight, 0.7)
        nn.init.constant_(model.q_network.bias, 0.7)
        nn.init.constant_(model.target_network.weight, 0.2)
        nn.init.constant_(model.target_network.bias, 0.2)
        state = torch.full(
            (
                2,
                img_height,
                img_width,
                n_channels * self.linear_config["hyper_params"]["state_history"],
            ),
            0.5,
        )

        with torch.no_grad():
            q_values = model.get_q_values(state, "q_network")
            target_q_values = model.get_q_values(state, "target_network")
        actions = torch.tensor([1, 3], dtype=torch.int)
        rewards = torch.tensor([5, 5], dtype=torch.float)
        terminated_mask = torch.tensor([0, 0], dtype=torch.bool)
        truncated_mask = torch.tensor([0, 0], dtype=torch.bool)

        q_values[0,[0,2,3]] -= 1
        q_values[1,[0,1,2]] -= 1
        loss = model.calc_loss(q_values, target_q_values, actions, rewards, terminated_mask, truncated_mask)
        self.assertEquals(round(loss.item(), 1), 424.4)

    def test_optimizer(self):
        env = EnvTest((5, 5, 1))
        model = Linear(env, self.linear_config)
        state_shape = list(env.observation_space.shape)
        img_height, img_width, n_channels = state_shape
        num_actions = env.action_space.n

        model.add_optimizer()
        self.assertTrue(isinstance(model.optimizer, torch.optim.Adam))
        self.assertEqual(len(model.optimizer.param_groups), 1)
        self.assertTrue(
            model.optimizer.param_groups[0]["params"][0] is model.q_network.weight
        )
        self.assertTrue(
            model.optimizer.param_groups[0]["params"][1] is model.q_network.bias
        )

    def test_run(self):
        env = EnvTest((5, 5, 1))

        # exploration strategy
        exp_schedule = LinearExploration(
            env,
            self.linear_config["hyper_params"]["eps_begin"],
            self.linear_config["hyper_params"]["eps_end"],
            self.linear_config["hyper_params"]["eps_nsteps"],
        )

        # learning rate schedule
        lr_schedule = LinearSchedule(
            self.linear_config["hyper_params"]["lr_begin"],
            self.linear_config["hyper_params"]["lr_end"],
            self.linear_config["hyper_params"]["lr_nsteps"],
        )

        # train model
        model = Linear(env, self.linear_config)
        model.run(exp_schedule, lr_schedule)


class TestDeepMindDQN(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dqn_deepmind_config = read_config('test_dqn.yml')

    def test_input_output_shapes(self):
        env = EnvTest((80, 80, 1))
        model = DQN(env, self.dqn_deepmind_config)

        state_shape = list(env.observation_space.shape)
        img_height, img_width, n_channels = state_shape
        num_actions = env.action_space.n
        sample_input = torch.randn(
            1,
            img_height,
            img_width,
            n_channels * self.dqn_deepmind_config["hyper_params"]["state_history"],
            device=model.device
        )
        output = model.get_q_values(sample_input, "q_network")

        self.assertTrue(model.q_network, nn.Sequential)
        self.assertTrue(any([isinstance(x, nn.Linear) for x in model.q_network]))
        self.assertTrue(any([isinstance(x, nn.ReLU) for x in model.q_network]))
        self.assertTrue(any([isinstance(x, nn.Flatten) for x in model.q_network]))
        self.assertEqual(output.shape[-1], num_actions)