
from stable_baselines3 import PPO
from environment_imitation import Grid_World
import argparse
import logging
from environment_imitation import generatelabel_val,generate_val,generate_test
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
import os
from os import listdir
from os.path import isfile, join
import json

def generate_seq(env,model):
    """
    evaluates the model
    """
    predicted_actions = []
    
    correct = 0
    correct_and_min = 0
    total = 0
    onlyfiles = [f for f in listdir(env.root_path) if isfile(join(env.root_path, f))]
    os.makedirs(env.root_path.replace('task','seq'),exist_ok=True)
    for file_name in onlyfiles:
            sequence ={}
            obs = env.reset()
            done = False
            steps = 0
            config =  generate_test(env.root_path,file_name)
            obs = env.get_observation(config)
            while((done is False) and (steps<=100)):        
                steps += 1
                action, _ = model.predict(obs, deterministic=True)
                obs, reward,done, info = env.step(action)
                predicted_actions.append(env.actions[action])
            total += 1
            sequence["sequence"] = predicted_actions
            
            with open(join(env.root_path, file_name).replace('task','seq'), 'w') as f:
              json.dump(sequence, f)

                    
            predicted_actions.clear()
               


def main():
      args = arg_parser()
      env = Grid_World(1,args.datapath,'data')
      if args.model_type == 'ppo':
        model =  PPO.load(path = args.model_path, env=env)
      else:
        model = bc.reconstruct_policy(args.model_path)
      generate_seq(env,model)



def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", help='use this option to provide a file/corpus for topic modeling.'
                                           'By default, samples from second line onwards are considered '
                                           '(assuming line 1 gives header info). To change this behaviour, '
                                           'use --include_first_line.')
    parser.add_argument("--model_type",help='ppo/bc')
    parser.add_argument("--model_path")
    args = parser.parse_args()
    print(args)
    return args

if __name__ == '__main__':
    main()



