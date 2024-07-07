import torch
import torch.nn as nn

from .dqn_abstract import AbstractDQN


class Linear(AbstractDQN):
    """
    We represent Q function as linear approximation Q_\theta(s,a) = \thetaT*\delta(s,a)
       where [\delta(s,a)]_{s‘,a‘} = 1 iff s‘ = s, a‘ = a.
    Implementation of a single fully connected layer with Pytorch to be utilized
    in the DQN algorithm.
    """

    def initialize_models(self):
        """
        Creates the 2 separate networks (Q network and Target network). The input
        to these networks will be an image of shape img_height * img_width with
        channels = n_channels * self.config["hyper_params"]["state_history"].
        """
        state_shape = list(self.env.observation_space.shape)
        img_height, img_width, n_channels = state_shape
        num_actions = self.env.action_space.n
        # linear layer with num_actions as the output size
        self.q_network = nn.Linear(
            img_height * img_width * n_channels * self.config["hyper_params"]["state_history"],
            num_actions)
        self.target_network = nn.Linear(
            img_height * img_width * n_channels * self.config["hyper_params"]["state_history"],
            num_actions)

    def get_q_values(self, state, network="q_network"):
        """
        Returns Q values for all actions.

        Args:
            state (torch tensor): shape =
                (batch_size,
                 img height,
                 img width,
                 nchannels x config["hyper_params"]["state_history"]
                )
            network (str): The name of the network, either "q_network" or "target_network"

        Returns:
            out (torch tensor): shape = (batch_size, num_actions)
        """
        if network == "q_network":
            out = self.q_network(torch.flatten(state, start_dim=1))
        elif network == "target_network":
            out = self.target_network(torch.flatten(state, start_dim=1))
        else:
            raise ValueError("Invalid network.")

        return out

    def update_target(self):
        """
        The update_target function will be called periodically to copy self.q_network
        weights to self.target_network.
        """
        self.target_network.load_state_dict(self.q_network.state_dict())

    def calc_loss(
            self,
            q_values: torch.Tensor,
            target_q_values: torch.Tensor,
            actions: torch.Tensor,
            rewards: torch.Tensor,
            terminated_mask: torch.Tensor,
            truncated_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the MSE loss of a given step. The loss for an example is defined:
            Q_samp(s) = r if terminated or truncated
                        = r + gamma * max_a' Q_target(s', a') otherwise
            loss = (Q_samp(s) - Q(s, a))^2

        Args:
            q_values: (torch tensor) shape = (batch_size, num_actions)
                The Q-values that your current network estimates (i.e. Q(s, a') for all a')

            target_q_values: (torch tensor) shape = (batch_size, num_actions)
                The Target Q-values that your target network estimates (i.e. (i.e. Q_target(s', a') for all a')

            actions: (torch tensor) shape = (batch_size,)
                The actions that you actually took at each step (i.e. a)

            rewards: (torch tensor) shape = (batch_size,)
                The rewards that you actually got at each step (i.e. r)

            terminated_mask: (torch tensor) shape = (batch_size,)
                A boolean mask of examples where we reached the terminal state

            truncated_mask: (torch tensor) shape = (batch_size,)
                A boolean mask of examples where the episode was truncated
        """
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
        """
        This function sets the optimizer for our linear network (optimize only q_network).
        """
        self.optimizer = torch.optim.Adam(self.q_network.parameters())
