import numpy as np
import gym
from gym import spaces
import os
import json


def generate_env(root_datapath, training_mode,episode):
        """
        returns training task configuration given an episode
        """
        global flag
        flag = 0
        root_fd = root_datapath +training_mode + '/train/task'

        file_name = str(episode) + '_task.json'
        file_path = os.path.join(root_fd, file_name)
        temp_list = []
        with open(file_path, 'r') as f:
            distros_dict = json.load(f)
        myvars = {}
        for distro in distros_dict:
            name = distro[0]
            temp_list.append(distros_dict[distro])
        return  temp_list

def generatelabel_env(root_datapath, training_mode,episode):
        """
	      returns action sequence given an episode """
        root_fd = root_datapath +training_mode +'/train/seq'
        file_name = str(episode) + '_seq.json'
        file_path = os.path.join(root_fd, file_name)
        temp_list = []
        with open(file_path, 'r') as f:
            distros_dict = json.load(f)
        myvars = {}
        for distro in distros_dict:
            name = distro[0]
            temp_list = distros_dict[distro]

        return temp_list   
        
def generate_val(root_datapath,filename):
        """
        returns validation task configuration given an episode
        """
        root_fd = root_datapath
        file_path = os.path.join(root_fd, filename)
        temp_list = []
        with open(file_path, 'r') as f:
            distros_dict = json.load(f)
        myvars = {}
        for distro in distros_dict:
            name = distro[0]
            temp_list.append(distros_dict[distro])

       
        return temp_list   


def generatelabel_val(root_datapath,filename):
        file_path = os.path.join(root_datapath, filename)
        file_path=file_path.replace('task','seq')
        temp_list = []
        with open(file_path, 'r') as f:
            distros_dict = json.load(f)
        myvars = {}
        for distro in distros_dict:
            name = distro[0]
            actions = distros_dict[distro]
            temp_list.append(distros_dict[distro])

        return actions        

        
def generate_test(root_datapath ,file_name):
        """
        returns test task configuration given an episode
        """
        file_path = os.path.join(root_datapath, file_name)
        temp_list = []
        with open(file_path, 'r') as f:
            distros_dict = json.load(f)
        myvars = {}
        for distro in distros_dict:
            name = distro[0]
            temp_list.append(distros_dict[distro])

       
        return temp_list   
      



    

class Grid_World(gym.Env):
  """
  Custom Environment that follows gym interface. 
  """
  metadata = {'render.modes': ['console']}
  # Define constants for clearer code
  pregrid_location = 0
  postgrid_location = 1
  pregrid_marker = 2
  postgrid_marker = 3
  wall = 4
  orientation = [(-1,0),(0,-1),(1,0),(0,1)]
  directions = ['north', 'west', 'south', 'east']
  actions = ['move', 'turnLeft','turnRight', 'finish','pickMarker', 'putMarker']
  flag = 0
  def __init__(self,episodes =1,root_path=None,training_mode=None):
    super(Grid_World, self).__init__()
    self.training_mode=training_mode
    self.episodes = episodes
    self.rng = np.random.default_rng()
    # Define action and observation space
    self.training_mode =training_mode
    self.root_path=root_path
    self.observation_space = spaces.MultiBinary([88])
    #self.observation_space = np.zeros((5,4, config[0],  config[1]))
    #orientation = np.zeros((2, 4, 1))
    n_actions = 6
    self.action_space = spaces.Discrete(n_actions)
    self.en_state_space = np.zeros((88))


  def get_observation(self,config):
    state_space =  np.zeros((5, config[0],  config[1]))
    direction = np.zeros((2 ,4,1))
    state_space[self.pregrid_location,config[2], config [3]] = 1
    direction[self.pregrid_location,self.directions.index(config[4]) ] =1
    #Post Grid Location
    state_space[self.postgrid_location,config[5], config [6]] = 1
    direction[self.postgrid_location,self.directions.index(config[7]) ] =1
    #print("get",direction)
    #Walls
    for loc in config[8]:
            state_space[self.wall,loc[0], loc [1]] = 1

    for loc in config[9]:
            state_space[ self.pregrid_marker,loc[0], loc [1]] = 1

    for loc in config[10]:
            state_space[ self.postgrid_marker, loc[0], loc [1]] = 1

   
    flatten_states = np.ndarray.flatten(state_space)
    flatten_direction = np.ndarray.flatten(direction)
    flatten_state = np.concatenate((flatten_states, flatten_direction), axis=0) 
    self.en_state_space =flatten_state.copy() 
    return flatten_state


  def reset(self):
    """
    Important: the observation must be a numpy array
    :return: (np.array) 
    """
    # Initialize the agent at the right of the gri
    state_space =  np.zeros((5,4, 4,  4))      
    # Initialize the agent at the right of the grid
    #self.agent_pos = self.grid_size - 1
    # here we convert to float32 to make it more general (in case we want to use continuous actions)
    return state_space


   
    flatten_states = np.ndarray.flatten(state_space)
    flatten_direction = np.ndarray.flatten(direction)
    flatten_state = np.concatenate((flatten_states, flatten_direction), axis=0) 
    self.en_state_space =flatten_state.copy() 
    return flatten_state

    # here we convert to float32 to make it more general (in case we want to use continuous actions
  def step(self, action):
            space = self.en_state_space.copy()
            state_space = space[:80].reshape((5,4,4))
            flat_direction = space[80:].reshape((2,4,1))

            action = action
            global flag
            reward = 0
            crash = False
            done = False
            state = np.where(state_space[self.pregrid_location, :, :] == 1)
            direction = np.where(flat_direction[self.pregrid_location, :,0] == 1)
            #print(direction)
            #index_z, index_z2 = np.where(state_space[self.postgrid_location,:, :] == 1)
            direction = direction[0].item()
            x=state[0].item()
            y=state[1].item()
            #print(x,y)
            #print('action', self.actions[action])
            #print(direction,x,y)

            if self.actions[action] == 'move':
                state_space[self.pregrid_location,x,y]=0
                new_direction,new_x,new_y = direction,x+ self.orientation[direction][0] ,y + self.orientation[direction][1]
                #print("new")
                #print(new_x,new_y)
                if new_x > 3 or new_y > 3:
                    crash = True
                elif new_x < 0 or new_y < 0:
                    crash = True
                elif state_space[self.wall,new_x,new_y] == 1:
                    crash = True
                else:
                    state_space[self.pregrid_location,new_x, new_y] = 1

                #if crash:
                  #print("wrong move")    
            elif  self.actions[action] == 'turnLeft':
                flat_direction[self.pregrid_location,direction]=0
                new_direction = self.orientation[direction][1],-self.orientation[direction][0]
                new_direction = new_direction[1],- new_direction[0]
                new_direction = new_direction[1],- new_direction[0]
                flat_direction[self.pregrid_location,self.orientation.index(new_direction)]=1
                #state_space[self.pregrid_location,self.orientation.index(new_direction),x,y]=1
      
                #state_space[self.pregrid_location,direction,:, :] = 1
            elif  self.actions[action] == 'turnRight':
                flat_direction[self.pregrid_location,direction]=0
                new_direction = self.orientation[direction][1],-self.orientation[direction][0]
                #state_space[self.pregrid_location,self.orientation.index(new_direction),x,y]=1
                flat_direction[self.pregrid_location,self.orientation.index(new_direction)] = 1

            elif self.actions[action]  == 'pickMarker':
                if state_space[self.pregrid_marker,x,y] == 1 and state_space[self.postgrid_marker,x,y] == 0:
                #if(state1[pregrid_mark, index_y, index_x]  == 1):
                    state_space[self.pregrid_marker,x,y] = 0
                    reward = 0.5
                else:
                    crash = True

            elif self.actions[action]  == 'putMarker':
                if state_space[self.pregrid_marker,x,y] == 0 and state_space[self.postgrid_marker,x,y] == 1:
                    #if(state1[pregrid_mark, index_y, index_x]  == 1):
                    state_space[self.pregrid_marker,x,y] = 1
                    reward = 0.5
                else:
                    crash = True

            else: #self.actions[action]  == 'finish':
              if((np.array_equiv(state_space[self.pregrid_location], state_space[self.postgrid_location])) and
                        (np.array_equiv(state_space[self.pregrid_marker], state_space[self.postgrid_marker])) and 
                 (np.array_equiv(flat_direction[self.pregrid_location], flat_direction[self.postgrid_location]))):
                    reward = 1
                    done = True
                    crash = True  
              else:
                crash = True   
                #print("wrong finish")          
            info={"done":done}
            #print(crash)
            #rdef render(self, mode="human"):eturn state_space , reward, crash, info
            flatten_states = np.ndarray.flatten(state_space)
            flatten_direction = np.ndarray.flatten(flat_direction)
            #print(flatten_direction)
            #print(action)
            flatten_state = np.concatenate((flatten_states, flatten_direction), axis=0) 
            self.en_state_space =flatten_state.copy() 
            #print(np.where(self.en_state_space[self.pregrid_location,:, :, :] == 1))
            #print(crash)
            return flatten_state, reward, crash, info

