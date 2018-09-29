import click
import gym
from osim.env import ProstheticsEnv
import numpy as np
from sac import policy
from sac import replaybuffer
from tensorboardX import SummaryWriter


@click.command()
@click.option('--env-name', default="ProstheticsEnv")