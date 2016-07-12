import random
import operator
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env, gamma_vals=None,learning_rate_vals=None):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        
        self.runs_per_param = 20
        self.params_tested = -1
        self.gamma_idx = 0
        self.report_file= open("report.txt","w")
        self.learning_rate_idx = -1
        if(gamma_vals == None):
            self.gamma_vals = [0.8, 0.5,0.1,0.01,0.001, 0]
        if(learning_rate_vals == None):
            self.learning_rate_vals = [1,0.8,0.5,0.3,0.1,0]


        self.runs = 0        
        self.steps = 0        
        
        self.state = None
        """ @property state: The current state. 

        @type state: str """
        
        self.runs_with_current_param = 0
        self.steps_with_current_param = 0
        self.test_next_param()
        
        self.runs_before_in_time = 0                            
        # TODO: Initialize any additional variables here

    

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.state = None
        self.runs_with_current_param += 1  
        
        if(self.net_rewards > 0):
            self.net_reward_positive_runs += 1
                                 

        if(self.runs % self.runs_per_param == 0 and self.runs != 0):
            self.print_report()            
            self.test_next_param()

        self.runs += 1
            
    def test_next_param(self):

        self.params_tested += 1
        self.learning_rate_idx += 1

        if(self.learning_rate_idx >= len(self.learning_rate_vals)):
            self.learning_rate_idx = 0
            self.gamma_idx += 1
                    

        if(self.gamma_idx >= len(self.gamma_vals)):
            print("Its over, thank you")
            self.report_file.close()
            exit()

        
        self.gamma = self.gamma_vals[self.gamma_idx]
        self.runs_before_in_time = 0
        self.learning_rate = self.learning_rate_vals[self.learning_rate_idx]

        
        self.net_reward_positive_runs = 0
        self.steps_before_in_time = 0
        self.steps_with_current_params = 0
        self.runs_with_current_param = 0
        self.in_time = 0
        self.net_rewards = 0
        self.invalid_steps = 0        
        self.invalid_steps_after_policy_learnt = 0        

        self.utilities = {}
        """ @property utilities: A dictuonary of states to utilities. @type utilities: dict"""
        

    def get_best_action(self, state): 

        if(state not in self.utilities):
            self.utilities[state] = {
                "forward" : 0,
                "left" : 0,
                "right" : 0,
                "put" : 0
            }


        action_to_utility =  self.utilities[state]

        best_action_val = max(action_to_utility.iteritems(), key=operator.itemgetter(1)) 

        best_action_val = best_action_val[1] if len(best_action_val) > 0 else 0

        all_maxs_actions = [i for i, j in action_to_utility.items() if j == best_action_val]        

        
        return random.choice(all_maxs_actions)        

    def print_report(self):

        self.report_file.write("**************************************************" + "\n")
        self.report_file.write("{} Steps completed for params gamma={}, learning_rate={} : ".format(self.steps_with_current_params ,self.gamma, self.learning_rate) + "\n")        
        self.report_file.write("Net Reward Postive Runs/ Runs with params : {}".format(float(self.net_reward_positive_runs) / self.runs_with_current_param) + "\n" )
        self.report_file.write("Invalid Steps with params / Steps with params: {}".format(float(self.invalid_steps) / self.steps_with_current_params) + "\n")
        self.report_file.write("Invalid Steps after policy learnt / Steps with params: {}".format(float(self.invalid_steps_after_policy_learnt) / (self.steps_with_current_params - self.steps_before_in_time) ) + "\n")
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
        self.state = (self.next_waypoint, str(inputs))

        
        # TODO: Select action according to your policy
        action = self.get_best_action(self.state)

        
        # Execute action and get reward
        reward = self.env.act(self, action if action != "put" else None)
        was_invalid_move = reward < 0 or reward == 9
        reached_destination_in_time = reward > 5
        self.net_rewards += reward

        if(was_invalid_move):
            self.invalid_steps += 1


        if(self.runs_before_in_time > 0 and was_invalid_move):
            self.invalid_steps_after_policy_learnt += 1        

        if(reached_destination_in_time):
            self.in_time += 1

        if(self.in_time == 1 and self.runs_before_in_time <= 0):            
            self.runs_before_in_time = self.runs_with_current_param

        if(self.in_time == 1):
            self.steps_before_in_time = self.steps_with_current_params

        future_input = self.env.sense(self)
        future_waypoint = self.planner.next_waypoint()
        future_state =  (self.next_waypoint, str(inputs))
        future_best_action = self.get_best_action(future_state)


        # TODO: Learn policy based on state, action, reward        
        q_update = reward + self.gamma * self.utilities[future_state][future_best_action]
        self.utilities[self.state][action] = (1-self.learning_rate) * self.utilities[self.state][action] + self.learning_rate * q_update
        print "LearningAgent.update(): deadline = {}, way_point = {}, state = {}, inputs = {}, action = {}, reward = {}".format(deadline, self.next_waypoint, self.state, inputs, action, reward)  # [debug]
        print "utilities[{}] = {}".format(action, self.utilities[self.state][action])  # [debug]

        
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



