from typing import Dict, Tuple
from mpi4py import MPI
import numpy as np
from dataclasses import dataclass
from repast4py import core, random, space, schedule, logging, parameters
from repast4py import context as ctx
import repast4py
from repast4py.space import DiscretePoint as dpt
import yaml

# TODO: create logs
@dataclass
class Log:
    pass


class AEP(core.Agent):

    TYPE = 0
    OFFSETS = np.array([-1, 1])

    def __init__(self, local_id: int, rank: int, pt: dpt):
        super().__init__(id=local_id, type=AEP.TYPE, rank=rank)
        self.pt = pt #current location
        self.state = params["aep_state.active"]

    def save(self) -> Tuple: 
        return (self.uid, self.state)
    
    def hyperactive(self):
        self.state =  params["aep_state.hyperactive"]

    def step(self):
        pass


class Protein(core.Agent):

    TYPE = 1
    OFFSETS = np.array([-1, 1])

    def __init__(self, local_id: int, rank: int, pt: dpt, protein_name):
        super().__init__(id=local_id, type=Protein.TYPE, rank=rank)
        self.pt = pt #current location
        self.name = protein_name

    def save(self) -> Tuple: 
        return (self.uid, self.name)

    def step(self):
        pass


class CleavedProtein(core.Agent):

    TYPE = 2
    OFFSETS = np.array([-1, 1])

    def __init__(self, local_id: int, rank: int, pt: dpt, protein_name):
        super().__init__(id=local_id, type=Protein.TYPE, rank=rank)
        self.pt = pt #current location
        self.name = protein_name

    def save(self) -> Tuple: 
        return (self.uid, self.name)

    def step(self):
        pass


agent_cache = {}


#restore 2 agents: enzyme and protein
def restore_agent(agent_data: Tuple):
    uid = agent_data[0]
    #pt_tuple = agent_data[?]
    #pt = dpt(*pt_tuple)

    #uid: 0 id, 1 type, 2 rank
    if uid[1] == AEP.TYPE:

        if uid in agent_cache:
            agent = agent_cache[uid]
        else: 
            agent = AEP(uid[0], uid[2])
            agent_cache[uid] = agent
        
        agent.state = agent_data[1]
        return agent
    
    else:

        if uid in agent_cache:
            agent = agent_cache[uid]
        else:
            agent = Protein(uid[0], uid[2], agent_data[1])
            agent_cache[uid] = agent

        return agent
    
    




#execution function (when starting the program you have to pass the yaml file: setup.yaml)
if __name__ == "__main__":
    parser = parameters.create_args_parser()
    args = parser.parse_args()
    params = parameters.init_params(args.parameters_file, args.parameters)
    
    #for debug
    #stream = open("dcc_project_in_repast4py/setup.yaml", "r")
    #params = yaml.load(stream)
    #params = params

    run(params)