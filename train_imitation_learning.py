import os
import environment_imitation
from environment_imitation import Grid_World
from environment_imitation import generatelabel_env,generate_env
import numpy as np
from stable_baselines3.common.monitor import Monitor
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import MlpPolicy

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.data.types import TrajectoryWithRew
import argparse

def create_trajectories(env,episodes,datapath):
  """
  args:
     env object of gridworld environment
     episodes number of files in the dataset
     datapath path of the dataset
  returns:
     trajectories for expert demonstration for supervised learning
  """
  actions = ['move', 'turnLeft','turnRight', 'finish','pickMarker', 'putMarker']
  trajectories = []
  for i in range (episodes):
    env.reset()
    actionlist = generatelabel_env(datapath,training_mode="data",episode = i)
    int_to_actionlist = [actions.index(action) for action in actionlist]
    config = generate_env(datapath,"data",i)
    state_space= env.get_observation(config)
    obs = [state_space]
    rews =[]
    
    # generate observation for each action
    for action in actionlist :
       state_space, reward, _,_  = env.step(actions.index(action))
       obs.append(state_space)
       rews.append(reward)
    trajectories.append(TrajectoryWithRew(obs = np.array(obs).astype(int),
                                          acts= np.array(int_to_actionlist).astype(int),terminal = True if actionlist[-1] == "finish" else False,rews=np.array(rews).astype(float),infos =None))
  return trajectories



rng = np.random.default_rng(0)


def main(args):
  rng = np.random.default_rng(0)
  n_episodes = len([name for name in os.listdir(args.datapath+"data/train/seq")])
  gamma=0.99
  env = Grid_World(n_episodes)
  env = Monitor(env,filename="ppo_log/")

  #transitions = rollout.flatten_trajectories(rollouts)
  n_episodes = len([name for name in os.listdir(args.datapath+"data/train/seq")])
  print(n_episodes)
  trajectories = create_trajectories(env,episodes=n_episodes,datapath = args.datapath)
  # create a rollout object of the trajectories
  transitions = rollout.flatten_trajectories_with_rew(trajectories)
  bc_trainer = bc.BC(
	    observation_space=env.observation_space,
	    action_space=env.action_space,
	    demonstrations=transitions,
	    rng=rng
	)
  bc_trainer.train(n_epochs=args.n_epochs)
  bc_trainer.save_policy(policy_path ="bc_trainer")
        #test(env,100000,102399,bc_trainer)




def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", help='use this option to provide a file/corpus for topic modeling.'
                                           'By default, samples from second line onwards are considered '
                                           '(assuming line 1 gives header info). To change this behaviour, '
                                           'use --include_first_line.')
    parser.add_argument("--n_epochs", type=int)

   
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = arg_parser()
    main(args)

