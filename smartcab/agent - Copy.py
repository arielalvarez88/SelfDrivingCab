import random
import operator
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        
        self.runs_per_param = 20
        self.params_tested = -1
        self.alpha_idx = 0
        self.report_file= open("report.txt","w")
        self.learning_rate_idx = -1
        self.alpha_vals = [0.8, 0.5,0.1,0.01,0.001, 0]
        self.learning_rate_vals = [1,0.8,0.5,0.3,0.1,0]
        self.runs = 0
        self.steps = 0        
        self.state = None
        self.runs_with_current_param = 0
        self.steps_with_current_param = 0
        self.initialize_params()
        
        self.runs_before_in_time = 0

        """ @property state: The current state. The possible states are : 
        green_light_cant_left, green_light_can_left, red_light_cant_right, red_light_can_right.

        @type state: str """
        
        self.state_factory = StateFactory()
        """ @property state_factory: It will map from inputs to state. @type state_factory: StateFactory """

        self.utilities = self.init_utilities()
        """ @property utilities: A map of state to utility. @type utilities: dict"""
        # TODO: Initialize any additional variables here


    def init_utilities(self):
        utilities = {}
        for state in States.values():
            utilities[state] = {
                "forward" : 0,
                "left" : 0,
                "right" : 0,
                "put" : 0
            }
        return utilities


    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.state = None
        self.runs_with_current_param += 1  
        
        if(self.net_rewards > 0):
            self.net_reward_positive_runs += 1
                                 

        if(self.runs % self.runs_per_param == 0 and self.runs != 0):
            self.print_report()            
            self.initialize_params()

        self.runs += 1
            
    def initialize_params(self):

        self.params_tested += 1
        self.learning_rate_idx += 1

        if(self.learning_rate_idx >= len(self.learning_rate_vals)):
            self.learning_rate_idx = 0
            self.alpha_idx += 1
                    

        if(self.alpha_idx >= len(self.alpha_vals)):
            print("Its over, thankyou")
            self.report_file.close()
            exit()

        
        self.alpha = self.alpha_vals[self.alpha_idx]
        self.runs_before_in_time = 0
        self.learning_rate = self.learning_rate_vals[self.learning_rate_idx]

        
        self.net_reward_positive_runs = 0
        self.steps_before_in_time = 0
        self.steps_with_current_params = 0
        self.runs_with_current_param = 0
        self.in_time = 0
        self.net_rewards = 0
        self.invalid_steps = 0
        self.max_steps = 100000000
        self.init_utilities()
        

    def print_report(self):

        self.report_file.write("**************************************************" + "\n")
        self.report_file.write("{} Steps completed for params Alpha={}, learning_rate={} : ".format(self.steps_with_current_params ,self.alpha, self.learning_rate) + "\n")        
        self.report_file.write("Net Reward Postive Runs/ Runs with params : {}".format(float(self.net_reward_positive_runs) / self.runs_with_current_param) + "\n" )
        self.report_file.write("Invalid Steps with params / Stpes with params: {}".format(float(self.invalid_steps) / self.steps_with_current_params) + "\n")
        self.report_file.write("In time / All runs: {}".format(float(self.in_time) / self.runs_with_current_param) + "\n")
        self.report_file.write("Steps before in time: {}".format(self.steps_before_in_time) + "\n")
        self.report_file.write("Runs before in time: {}".format(self.runs_before_in_time) + "\n")
        self.report_file.write("Percentage of runs not in time: {}".format(float(self.runs_before_in_time)/self.runs_with_current_param) + "\n")

        #self.report_file.write("---Object state: " + "\n")
        #for key, value in self.__dict__.iteritems():
        #    if(not key.startswith("_")):
        #        self.report_file.write("{} : {}".format(key, value) + "\n")

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulato
        inputs = self.env.sense(self)
      
        deadline = self.env.get_deadline(self)        
        self.steps+= 1
        self.steps_with_current_params +=1
        # TODO: Update state
        self.state = self.state_factory.map_state(self.next_waypoint, inputs)
        
        # TODO: Select action according to your policy
        actionToUtility = self.utilities[self.state]

        bestActionVal = max(actionToUtility.iteritems() ,key=operator.itemgetter(1)) 

        bestActionVal = bestActionVal[1] if len(bestActionVal) > 0 else 0

        allMaxsActions = [i for i, j in actionToUtility.items() if j == bestActionVal]        

        
        action = random.choice(allMaxsActions)        

    

        """littleRandomnes = random.randint(0, int(t/2)) == 1

        if (littleRandomnes):            
            action = random.choice((None, 'forward', 'left', 'right'))"""
        
        # Execute action and get reward
        reward = self.env.act(self, action if action != "put" else None)
        was_invalid_move = reward < 0 or reward == 9
        reached_destination_in_time = reward > 5
        self.net_rewards += reward

        if(was_invalid_move):
            self.invalid_steps += 1

        if(reached_destination_in_time):
            self.in_time += 1

        if(self.in_time == 1 and self.runs_before_in_time <= 0):            
            self.runs_before_in_time = self.runs_with_current_param

        if(self.in_time == 1):
            self.steps_before_in_time = self.steps_with_current_params


        # TODO: Learn policy based on state, action, reward
        qUpdate = reward + self.alpha * sum(actionToUtility.values())
        actionToUtility[action] = (1-self.learning_rate) * actionToUtility[action] + self.learning_rate * qUpdate
        print "LearningAgent.update(): deadline = {}, way_point = {}, state = {}, inputs = {}, action = {}, reward = {}".format(deadline, self.next_waypoint, self.state, inputs, action, reward)  # [debug]
        print "utilities[{}] = {}".format(action, actionToUtility[action])  # [debug]

        



class States(object):
    green_light_cant_left_forward = "green_light_cant_left_forward"  
    green_light_cant_left_left ="green_light_cant_left_left"
    green_light_cant_left_right ="green_light_cant_left_right"

    green_light_can_left_forward ="green_light_can_left_forward"
    green_light_can_left_left ="green_light_can_left_left"
    green_light_can_left_right ="green_light_can_left_right"

    red_light_cant_right_forward ="red_light_cant_right_forward"
    red_light_cant_right_left ="red_light_cant_right_left"
    red_light_cant_right_right ="red_light_cant_right_right"


    red_light_can_right_forward ="red_light_can_right_forward"
    red_light_can_right_left ="red_light_can_right_left"
    red_light_can_right_right ="red_light_can_right_right"

    @staticmethod
    def values():
        for key, val in States.__dict__.items():
            if(key.startswith("_")):
                continue
            yield val

class StateFactory(object):
    """ This class will map the inputs to one of the states."""    

    def map_state(self, next_waypoint, inputs):
        possible_states = []

        if(inputs['light'] == "green"):

            factory = GreenLightFactory() 
        else:
            factory = RedLightFactory()  

        return factory.map_state(inputs) + "_" + next_waypoint

class GreenLightFactory(object):
    def map_state(self,inputs):            

            if(inputs['oncoming'] != "forward"):
                return "green_light_can_left"
            else:
                return "green_light_cant_left"


class RedLightFactory(object):
    def map_state(self,inputs):            
            if(inputs['left'] != "forward" and inputs['oncoming'] != "left"):
                return "red_light_can_right"                
            else:
                return "red_light_cant_right"

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=False)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.5, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=10000)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()



