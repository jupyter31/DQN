import os
import random
import sys

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


class LinearSchedule(object):
    """
    Sets test_linear schedule for exploration parameter epsilon.

    Args:
        eps_begin (float): initial exploration
        eps_end (float): end exploration
        nsteps (int): number of steps between the two values of eps
    """

    def __init__(self, eps_begin, eps_end, nsteps):

        assert (
            eps_begin >= eps_end
        ), "Epsilon begin ({}) needs to be greater than equal to end ({})".format(
            eps_begin, eps_end
        )

        self.epsilon = eps_begin
        self.eps_begin = eps_begin
        self.eps_end = eps_end
        self.nsteps = nsteps

    def update(self, t):
        """
        Updates epsilon.

        Args:
            t (int): frame number
        """
        if t <= self.nsteps:
            delta = (self.eps_end - self.eps_begin) / self.nsteps
            self.epsilon = self.eps_begin + delta * t
        else:
            self.epsilon = self.eps_end


class LinearExploration(LinearSchedule):
    """
    Implements e-greedy exploration with test_linear decay.

    Args:
        env (object): gym environment
        eps_begin (float): initial exploration rate
        eps_end (float): final exploration rate
        nsteps (int): number of steps taken to linearly decay eps_begin to eps_end
    """

    def __init__(self, env, eps_begin, eps_end, nsteps):
        self.env = env
        super(LinearExploration, self).__init__(eps_begin, eps_end, nsteps)

    def get_action(self, best_action):
        """
        Returns a random action with prob epsilon, otherwise returns the best_action.

        Args:
            best_action (int): best action according some policy

        Returns:
                (int) action
        """
        if random.random() <= self.epsilon:
            return self.env.action_space.sample()
        else:
            return best_action
