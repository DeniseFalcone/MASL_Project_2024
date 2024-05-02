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

@jitclass(spec)
class GridNghFinder:

    def __init__(self, xmin, ymin, xmax, ymax):
        self.mo = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1], dtype=np.int32)
        self.no = np.array([1, 1, 1, 0, 0, 0, -1, -1, -1], dtype=np.int32)
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

    def find(self, x, y):
        xs = self.mo + x
        ys = self.no + y

        xd = (xs >= self.xmin) & (xs <= self.xmax)
        xs = xs[xd]
        ys = ys[xd]

        yd = (ys >= self.ymin) & (ys <= self.ymax)
        xs = xs[yd]
        ys = ys[yd]

        return np.stack((xs, ys, np.zeros(len(ys), dtype=np.int32)), axis=-1)

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
        #Take the agent position and get all the nghs 
        grid = model.grid
        pt = grid.get_location(self)
        nghs = model.ngh_finder.find(pt.x, pt.y)

        empty_nghs = [ngh for ngh in nghs if not grid.is_cell_occupied(ngh)]
        not_empty_nghs = [ngh for ngh in nghs if grid.is_cell_occupied(ngh)]

        if empty_nghs:            
            #Randomly select a ngh cell
            random_ngh = random.choice(nghs)

            #Calculate the direction toward the selected nghs
            direction = (random_ngh - pt.coordinates[0:3]) * 0.25

            #Move the agent randomly
            cpt = model.space.get_location(self)
            model.move(self, cpt.x + direction[0], cpt.y + direction[1])

            x = self.percept(not_empty_nghs)
            if x is not None:
                self.cleave(x)

    #enzima controlla se il vicino è una proteina
    def percept(self, not_empty_nghs):
        for ngh in not_empty_nghs:
            if type(ngh) == Protein:
                return ngh
        return None    
    
                
    #TODO           
    #azione dell'enzima che taglia una proteina
    def cleave(self,ngh):  
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