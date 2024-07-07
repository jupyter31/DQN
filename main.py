import sys
import argparse
import gym
from q_learning.utils import read_config
from q_learning.preprocess import greyscale
from q_learning.environment import PreproWrapper, MaxPoolSkipEnv, EnvTest
from q_learning.network import Linear, DQN, LinearExploration, LinearSchedule

"""
This script lets us run deep Q network or linear approximation according to a custom config file.
(Configuration specified in the configs, config/ folder).
Results, log and recording of the agent are stored in the results folder.

We can monitor the progress of the agent with Tensorboard:
To launch tensorboard (default port is 6006):
    >tensorboard --logdir=results/ --host 0.0.0.0
"""


def run():
    parser = argparse.ArgumentParser(
        description="A program to run DQN training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config_filename",
        help="The name of the config file in the config/ directory to be used for model training.",
        default="test_linear.yml",
    )

    args = parser.parse_args()

    config = read_config(args.config_filename)

    if config["env"]["env_name"] == "test_environment":

        if config["model"] == "dqn":
            env = EnvTest((80, 80, 3))

            # exploration strategy
            exp_schedule = LinearExploration(
                env,
                config["hyper_params"]["eps_begin"],
                config["hyper_params"]["eps_end"],
                config["hyper_params"]["eps_nsteps"],
            )

            # learning rate schedule
            lr_schedule = LinearSchedule(
                config["hyper_params"]["lr_begin"],
                config["hyper_params"]["lr_end"],
                config["hyper_params"]["lr_nsteps"],
            )

            # train model
            model = DQN(env, config)
            model.run(exp_schedule, lr_schedule)

        elif config["model"] == "linear":
            env = EnvTest((5, 5, 1))

            # exploration strategy
            exp_schedule = LinearExploration(
                env,
                config["hyper_params"]["eps_begin"],
                config["hyper_params"]["eps_end"],
                config["hyper_params"]["eps_nsteps"],
            )

            # learning rate schedule
            lr_schedule = LinearSchedule(
                config["hyper_params"]["lr_begin"],
                config["hyper_params"]["lr_end"],
                config["hyper_params"]["lr_nsteps"],
            )

            # train model
            model = Linear(env, config)
            model.run(exp_schedule, lr_schedule)

        else:
            sys.exit(
                "Incorrectly specified model, config['model'] should either be 'dqn' or 'linear'."
            )
    elif config["env"]["env_name"] == "ALE/Pong-v5":
            # create env
            env = gym.make(
                config["env"]["env_name"],
                frameskip=(2, 5),
                full_action_space=False,
                render_mode=config["env"]["render_mode"],
            )
            env = MaxPoolSkipEnv(env, skip=config["hyper_params"]["skip_frame"])
            env = PreproWrapper(
                env,
                prepro=greyscale,
                shape=(80, 80, 1),
                overwrite_render=config["env"]["overwrite_render"],
            )

            # exploration strategy
            exp_schedule = LinearExploration(
                env,
                config["hyper_params"]["eps_begin"],
                config["hyper_params"]["eps_end"],
                config["hyper_params"]["eps_nsteps"],
            )

            # learning rate schedule
            lr_schedule = LinearSchedule(
                config["hyper_params"]["lr_begin"],
                config["hyper_params"]["lr_end"],
                config["hyper_params"]["lr_nsteps"],
            )

            if config["model"] == "dqn":
                model = DQN(env, config)
                model.run(exp_schedule, lr_schedule)

            elif config["model"] == "linear":
                model = Linear(env, config)
                model.run(exp_schedule, lr_schedule)

            else:
                sys.exit("Incorrectly specified model, config['model'] should either be dqn or linear.")
    else:
        sys.exit(
            "Incorrectly specified environment, config['model'] should either be 'Pong-v5' or 'test_environment'."
        )


if __name__ == "__main__":
    run()
