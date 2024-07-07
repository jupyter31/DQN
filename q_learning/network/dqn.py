import torch
import torch.nn as nn
from .dqn_abstract import AbstractDQN


class DQN(AbstractDQN):
    """
    Implementation of DeepMind's Nature paper:
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    """

    def initialize_models(self):
        """Create two separate networks (Q network and Target network). The in_channels
        to Conv2d networks will n_channels * self.config["hyper_params"]["state_history"]

        Args:
            q_network (torch model): variable to store our q network implementation
            target_network (torch model): variable to store our target network implementation
        """
        state_shape = list(self.env.observation_space.shape)
        img_height, img_width, n_channels = state_shape
        num_actions = self.env.action_space.n

        in_channels = n_channels * self.config["hyper_params"]["state_history"]

        # Q-Network architecture
        self.q_network = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

        # Target network architecture (initialized from scratch)
        self.target_network = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def get_q_values(self, state, network):
        """ Returns Q values for all actions """
        out = None
        reshaped_input = state.permute(0, 3, 1, 2)
        if network == "q_network":
            out = self.q_network(reshaped_input)
        elif network == "target_network":
            out = self.target_network(reshaped_input)
        return out

    def update_target(self) -> None:
        self.target_network.load_state_dict(self.q_network.state_dict())

    def calc_loss(self, q_values: torch.Tensor, target_q_values: torch.Tensor, actions: torch.Tensor,
                  rewards: torch.Tensor, truncated_mask: torch.Tensor, terminated_mask: torch.Tensor) -> torch.Tensor:
        gamma = self.config["hyper_params"]["gamma"]

        terminated_mask = terminated_mask.float()
        truncated_mask = truncated_mask.float()
        actions = actions.long()

        q_samp = rewards + gamma * torch.max(target_q_values, dim=1).values
        q_samp = q_samp * (1 - terminated_mask) * (1 - truncated_mask)

        # Gather the Q-values for the selected actions
        q_selected = torch.gather(q_values, 1, actions.unsqueeze(1)).squeeze(1)

        # Calculate the MSE loss
        loss = torch.mean((q_samp - q_selected) ** 2)
        return loss

    def add_optimizer(self):
        self.optimizer = torch.optim.Adam(self.q_network.parameters())