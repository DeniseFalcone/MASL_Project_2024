from typing import Dict, Tuple
from mpi4py import MPI
import numpy as np
from dataclasses import dataclass
from repast4py import core, random, space, schedule, logging, parameters
from repast4py import context as ctx
import repast4py
import repast4py.random
from repast4py.space import DiscretePoint as dpt
from repast4py.space import ContinuousPoint as cpt
import yaml
import numba
from numba import int32, int64
from numba.experimental import jitclass
import math

@dataclass
class Log:
    number_of_active_microglia: int = 0
    number_of_resting_microglia: int = 0
    number_of_healthy_neuron: int = 0
    number_of_damaged_neuron: int = 0
    number_of_dead_neuron: int = 0
    number_alpha_syn_oligomer: int = 0
    number_of_cleaved_alpha_syn: int = 0
    number_tau_oligomer: int = 0
    number_of_cleaved_tau: int = 0
    number_of_cytokine: int = 0

@numba.jit((int64[:], int64[:]), nopython=True)
def is_equal(a1, a2):
    return a1[0] == a2[0] and a1[1] == a2[1]

spec = [
    ('mo', int32[:]),
    ('no', int32[:]),
    ('xmin', int32),
    ('ymin', int32),
    ('ymax', int32),
    ('xmax', int32)
]

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

        # Remove the original (x, y) coordinate
        mask = (xs != x) | (ys != y)
        xs = xs[mask]
        ys = ys[mask]

        return np.stack((xs, ys, np.zeros(len(ys), dtype=np.int32)), axis=-1)
    
#class to create agent Microglia
class Microglia(core.Agent):

    TYPE = 0
    OFFSETS = np.array([-1, 1])

    # params["agent_state"]["resting"] state of an agent that is resting
    def __init__(self, local_id: int, rank: int, initial_state, pt):
        super().__init__(id=local_id, type=Microglia.TYPE, rank=rank)
        self.state = initial_state
        self.pt = pt
    
    def save(self) -> Tuple:
        return (self.uid, self.state, self.pt.coordinates)

    def step(self):
        nghs_coords = model.ngh_finder.find(self.pt.x, self.pt.y)
        ngh = self.check_oligomer_nghs(nghs_coords)
        if ngh is not None: 
            if self.state == params["microglia_state"]["resting"]:
                self.state = params["microglia_state"]["active"]
            else: 
                ngh.toRemove = True

    def check_oligomer_nghs(self, nghs_coords):
        for ngh_coords in nghs_coords:
            ngh_dpt = dpt(ngh_coords[0], ngh_coords[1])
            ngh = model.grid.get_agent(ngh_dpt)
            if (type(ngh) == Oligomer):
                    return ngh
        return None


#class to create Neurons
class Neuron(core.Agent): 

    TYPE = 1

    def __init__(self, local_id: int, rank: int, initial_state, pt):
        super().__init__(id=local_id, type=Neuron.TYPE, rank=rank)
        self.state = initial_state
        self.pt = pt

    def save(self) -> Tuple:
        return (self.uid, self.state, self.pt.coordinates)
    
    def step(self): 
        pass


class CleavedProtein(core.Agent): 

    TYPE = 2

    def __init__(self, local_id: int, rank: int, name, pt):
        super().__init__(id=local_id, type=CleavedProtein.TYPE, rank=rank)
        self.name = name
        self.pt = pt

    def save(self) -> Tuple:
        return (self.uid, self.name, self.pt.coordinates)
    
    def step(self): 
        #TODO
        pass


class Oligomer(core.Agent): 

    TYPE = 3

    def __init__(self, local_id: int, rank: int, name, pt):
        super().__init__(id=local_id, type=Oligomer.TYPE, rank=rank)
        self.name = name
        self.pt = pt
        self.toRemove = False

    def save(self) -> Tuple:
        return (self.uid, self.name, self.pt.coordinates, self.toRemove)
    
    def step(self): 
        #TODO
        pass


class Cytokine(core.Agent): 

    TYPE = 4

    def __init__(self, local_id: int, rank: int, pt):
        super().__init__(id=local_id, type=Cytokine.TYPE, rank=rank)
        self.pt = pt

    def save(self) -> Tuple:
        return (self.uid, self.pt.coordinates)
    
    def step(self): 
        #TODO
        #percepisce la microglia a riposo e la attiva
        pass


agent_cache= {}

def restore_agent(agent_data: Tuple):
    uid = agent_data[0]

    if uid[1] == Microglia.TYPE:
        agent_state = agent_data[1]
        pt = agent_data[2]
        if uid in agent_cache:
            agent = agent_cache[uid]
        else:
            agent = Microglia(uid[0], uid[2], agent_state, pt)
            agent_cache[uid] = agent
        agent.state = agent_state
        agent.pt = pt
    elif uid[1] == Neuron.TYPE:
        agent_state = agent_data[1]
        pt = agent_data[2]
        if uid in agent_cache:
            agent = agent_cache[uid]
        else:
            agent = Neuron(uid[0], uid[2], agent_state, pt)
            agent_cache[uid] = agent
        agent.state = agent_state
        agent.pt = pt
    elif uid[1] == CleavedProtein.TYPE:
        agent_name = agent_data[1]
        pt = agent_data[2]
        if uid in agent_cache:
            agent = agent_cache[uid]
        else:
            agent = CleavedProtein(uid[0], uid[2], agent_name, pt)
            agent_cache[uid] = agent
        agent.name = agent_name
        agent.pt = pt
    elif uid[1] == Oligomer.TYPE:
        agent_name = agent_data[1]
        pt = agent_data[2]
        toRemove = agent_data[3]
        if uid in agent_cache:
            agent = agent_cache[uid]
        else:
            agent = Oligomer(uid[0], uid[2], agent_name, pt)
            agent_cache[uid] = agent
        agent.name = agent_name
        agent.pt = pt
        agent.toRemove = toRemove
    elif uid[1] == Cytokine.TYPE:
        pt = agent_data[1]
        if uid in agent_cache:
            agent = agent_cache[uid]
        else:
            agent = Cytokine(uid[0], uid[2], pt)
            agent_cache[uid] = agent
        agent.pt = pt
    
    return agent

class Model:
    def __init__(self, comm: MPI.Intracomm, params: Dict):
        self.comm = comm
        self.context = ctx.SharedContext(comm)
        self.rank = comm.Get_rank()

        box = space.BoundingBox(0,params['world.width']-1, 0, params['world.height']-1,0,0)
        self.grid = space.SharedGrid(name='grid', bounds=box, borders=space.BorderType.Sticky, occupancy=space.OccupancyType.Single,
                                     buffer_size=1, comm=comm)
        self.context.add_projection(self.grid)    
        self.ngh_finder = GridNghFinder(0,0,box.xextent,box.yextent)

        self.runner = schedule.init_schedule_runner(comm)
        self.runner.schedule_repeating_event(1, 1, self.step)    
        self.runner.schedule_stop(params['stop.at'])
        self.runner.schedule_end_event(self.at_end)

        repast4py.random.seed = params['seed']
        self.rng = repast4py.random.default_rng

        self.counts = Log()
        loggers = logging.create_loggers(self.counts, op=MPI.SUM, rank=self.rank)
        self.data_set = logging.ReducingDataSet(loggers, self.comm, params['brain_log_file'], buffer_size=1)

        #parallel distribution of the agents
        world_size = self.comm.Get_size()

        total_h_neuron_count = params['neuron_healthy.count']
        pp_h_neuron_count = int(total_h_neuron_count/world_size)
        if self.rank < total_h_neuron_count % world_size:
            pp_h_neuron_count += 1

        total_dam_neuron_count = params['neuron_damaged.count']
        pp_dam_neuron_count = int(total_dam_neuron_count/world_size)
        if self.rank < total_dam_neuron_count % world_size:
            pp_dam_neuron_count += 1

        total_dead_neuron_count = params['neuron_dead.count']
        pp_dead_neuron_count = int(total_dead_neuron_count/world_size)
        if self.rank < total_dead_neuron_count % world_size:
            pp_dead_neuron_count += 1
        
        total_rest_microglia_count = params['resting_microglia.count']
        pp_rest_microglia_count = int(total_rest_microglia_count/world_size)
        if self.rank < total_rest_microglia_count % world_size:
            pp_rest_microglia_count += 1

        total_active_microglia_count = params['active_microglia.count']
        pp_active_microglia_count = int(total_active_microglia_count/world_size)
        if self.rank < total_active_microglia_count % world_size:
            pp_active_microglia_count += 1
        
        total_alpha_cleaved_count = params['alpha_syn_cleaved.count']
        pp_alpha_cleaved_count = int(total_alpha_cleaved_count/world_size)
        if self.rank < total_alpha_cleaved_count % world_size:
            pp_alpha_cleaved_count += 1

        total_tau_cleaved_count = params['tau_cleaved.count']
        pp_tau_cleaved_count = int(total_tau_cleaved_count/world_size)
        if self.rank < total_tau_cleaved_count % world_size:
            pp_tau_cleaved_count += 1

        total_alpha_oligomer_count = params['alpha_syn_oligomer.count']
        pp_alpha_oligomer_count = int(total_alpha_oligomer_count/world_size)
        if self.rank < total_alpha_oligomer_count % world_size:
            pp_alpha_oligomer_count += 1

        total_tau_oligomer_count = params['tau_oligomer.count']
        pp_tau_oligomer_count = int(total_tau_oligomer_count/world_size)
        if self.rank < total_tau_oligomer_count % world_size:
            pp_tau_oligomer_count += 1
        
        total_cytokine_count = params['cytokine.count']
        pp_cytokine_count = int(total_cytokine_count/world_size)
        if self.rank < total_cytokine_count % world_size:
            pp_cytokine_count += 1

        self.added_agents_id = pp_h_neuron_count + pp_tau_oligomer_count + pp_dam_neuron_count + pp_dead_neuron_count + pp_rest_microglia_count + pp_active_microglia_count + pp_alpha_cleaved_count + pp_tau_cleaved_count + pp_alpha_oligomer_count + pp_cytokine_count

        rng = repast4py.random.default_rng
        for j in range(pp_h_neuron_count):
            pt = self.grid.get_random_local_pt(rng)
            while(self.grid.get_agent(pt) is not None):
                pt = self.grid.get_random_local_pt(self.rng)
            h_neuron = Neuron(j,self.rank,params["neuron_state"]["healthy"], pt)
            self.context.add(h_neuron)
            self.move(h_neuron, pt)
        for j in range(pp_dam_neuron_count):
            pt = self.grid.get_random_local_pt(rng)
            while(self.grid.get_agent(pt) is not None):
                pt = self.grid.get_random_local_pt(self.rng)
            dam_neuron = Neuron(pp_h_neuron_count + j,self.rank,params["neuron_state"]["damaged"], pt)
            self.context.add(dam_neuron)
            self.move(dam_neuron, pt)
        for j in range(pp_dead_neuron_count):
            pt = self.grid.get_random_local_pt(rng)
            while(self.grid.get_agent(pt) is not None):
                pt = self.grid.get_random_local_pt(self.rng)
            dead_neuron = Neuron(pp_h_neuron_count + pp_dam_neuron_count + j,self.rank,params["neuron_state"]["dead"], pt)
            self.context.add(dead_neuron)
            self.move(dead_neuron, pt)
        for j in range(pp_rest_microglia_count):
            pt = self.grid.get_random_local_pt(rng)
            while(self.grid.get_agent(pt) is not None):
                pt = self.grid.get_random_local_pt(self.rng)
            rest_microglia = Microglia(pp_h_neuron_count + pp_dam_neuron_count + pp_dead_neuron_count + j,self.rank,params["microglia_state"]["resting"], pt)
            self.context.add(rest_microglia)
            self.move(rest_microglia, pt)
        for j in range(pp_active_microglia_count):
            pt = self.grid.get_random_local_pt(rng)
            while(self.grid.get_agent(pt) is not None):
                pt = self.grid.get_random_local_pt(self.rng)
            active_microglia = Microglia(pp_h_neuron_count + pp_dam_neuron_count + pp_dead_neuron_count + pp_rest_microglia_count + j,self.rank,params["microglia_state"]["active"], pt)
            self.context.add(active_microglia)
            self.move(active_microglia, pt)
        for j in range(pp_alpha_cleaved_count):
            pt = self.grid.get_random_local_pt(rng)
            while(self.grid.get_agent(pt) is not None):
                pt = self.grid.get_random_local_pt(self.rng)
            alpha_cleaved = CleavedProtein(pp_h_neuron_count + pp_dam_neuron_count + pp_dead_neuron_count + pp_rest_microglia_count + pp_active_microglia_count + j,self.rank,params["protein_name"]["alpha_syn"], pt)
            self.context.add(alpha_cleaved)
            self.move(alpha_cleaved, pt)
        for j in range(pp_tau_cleaved_count):
            pt = self.grid.get_random_local_pt(rng)
            while(self.grid.get_agent(pt) is not None):
                pt = self.grid.get_random_local_pt(self.rng)
            tau_cleaved = CleavedProtein(pp_h_neuron_count + pp_dam_neuron_count + pp_dead_neuron_count + pp_rest_microglia_count + pp_active_microglia_count + pp_alpha_cleaved_count + j,self.rank,params["protein_name"]["tau"], pt)
            self.context.add(tau_cleaved)
            self.move(tau_cleaved, pt)
        for j in range(pp_alpha_oligomer_count):
            pt = self.grid.get_random_local_pt(rng)
            while(self.grid.get_agent(pt) is not None):
                pt = self.grid.get_random_local_pt(self.rng)
            alpha_oligomer = Oligomer(pp_h_neuron_count + pp_dam_neuron_count + pp_dead_neuron_count + pp_rest_microglia_count + pp_active_microglia_count + pp_alpha_cleaved_count + pp_tau_cleaved_count + j,self.rank,params["protein_name"]["alpha_syn"], pt)
            self.context.add(alpha_oligomer)
            self.move(alpha_oligomer, pt)
        for j in range(pp_tau_oligomer_count):
            pt = self.grid.get_random_local_pt(rng)
            while(self.grid.get_agent(pt) is not None):
                pt = self.grid.get_random_local_pt(self.rng)
            tau_oligomer = Oligomer(pp_h_neuron_count + pp_dam_neuron_count + pp_dead_neuron_count + pp_rest_microglia_count + pp_active_microglia_count + pp_alpha_cleaved_count + pp_tau_cleaved_count + pp_alpha_oligomer_count + j,self.rank,params["protein_name"]["tau"], pt)
            self.context.add(tau_oligomer)
            self.move(tau_oligomer, pt)
        for j in range(pp_cytokine_count):
            pt = self.grid.get_random_local_pt(rng)
            while(self.grid.get_agent(pt) is not None):
                pt = self.grid.get_random_local_pt(self.rng)
            cytokine = Cytokine(pp_h_neuron_count + pp_tau_oligomer_count + pp_dam_neuron_count + pp_dead_neuron_count + pp_rest_microglia_count + pp_active_microglia_count + pp_alpha_cleaved_count + pp_tau_cleaved_count + pp_alpha_oligomer_count + j,self.rank, pt)
            self.context.add(cytokine)
            self.move(cytokine, pt)

    def step(self):
        #considerare rimuovere neuroni
        #considerare citochine anti infiammatorie

        self.context.synchronize(restore_agent)
        self.log_counts()

        for agent in self.context.agents():
            agent.step()

        oligomer_to_remove = []
        active_microglia = 0
        damaged_neuron = 0

        for agent in self.context.agents():
            if (type(agent) == Oligomer and agent.toRemove == True):
                oligomer_to_remove.append(agent)
            elif (type(agent) == Microglia and agent.state == params["microglia_state"]["active"]):
                active_microglia += 1
            elif (type(agent) == Neuron and agent.state == params["neuron_state"]["damaged"]):
                damaged_neuron += 1
        
        for i in range(active_microglia):
            self.add_cytokine()
            
        for i in range(damaged_neuron):
            self.add_cleaved_protein()

        for oligomer in oligomer_to_remove:
            self.remove_agent(oligomer)

    def remove_agent(self, agent):
        self.context.remove(agent)

    def add_cleaved_protein(self):
        self.added_agents_id += 1
        possible_types = [params["protein_name"]["alpha_syn"], params["protein_name"]["tau"]]
        random_index = np.random.randint(0, len(possible_types))
        cleaved_protein_name = possible_types[random_index]
        pt = self.grid.get_random_local_pt(self.rng)
        while(self.grid.get_agent(pt) is not None):
            pt = self.grid.get_random_local_pt(self.rng)  
        cleaved_protein = CleavedProtein(self.added_agents_id, self.rank, cleaved_protein_name, pt)   
        self.context.add(cleaved_protein)   
        self.move(cleaved_protein, cleaved_protein.pt)

    def add_cytokine(self):
        self.added_agents_id += 1
        pt = self.grid.get_random_local_pt(self.rng)
        while(self.grid.get_agent(pt) is not None):
            pt = self.grid.get_random_local_pt(self.rng)  
        cytokine= Cytokine(self.added_agents_id, self.rank, pt)   
        self.context.add(cytokine)   
        self.move(cytokine, cytokine.pt)

    def move(self, agent, pt):
        self.grid.move(agent, pt)
        agent.pt = self.grid.get_location(agent)

    def log_counts(self):
        tick = self.runner.schedule.tick

        #se rimuoviamo neuroni morti, salvare il quantitativo in una variabile in model: self.deadNeuron che fa da contatore

        microglia_resting = 0
        microglia_active = 0
        neuron_healthy = 0
        neuron_damaged = 0
        neuron_dead = 0
        cytokine = 0
        alpha_cleaved = 0
        tau_cleaved = 0
        alpha_oligomer = 0
        tau_oligomer = 0

        for agent in self.context.agents(): 
            if(type(agent) == Oligomer): 
                if (agent.name == params["protein_name"]["alpha_syn"]): 
                    alpha_oligomer += 1 
                else: 
                    tau_oligomer += 1 
            if(type(agent) == CleavedProtein): 
                if (agent.name == params["protein_name"]["alpha_syn"]): 
                    alpha_cleaved += 1 
                else: 
                    tau_cleaved += 1 
            elif(type(agent) == Neuron): 
                if (agent.state == params["neuron_state"]["healthy"]): 
                    neuron_healthy += 1 
                elif (agent.state == params["neuron_state"]["damaged"]):
                    neuron_damaged += 1
                else: 
                    neuron_dead += 1
            elif(type(agent) == Microglia): 
                if(agent.state == params["microglia_state"]["active"]): 
                    microglia_active += 1 
                else:  
                    microglia_resting += 1
            elif(type(agent) == Cytokine):
                cytokine += 1
        
        self.counts.number_of_healthy_neuron = neuron_healthy
        self.counts.number_of_damaged_neuron = neuron_damaged
        self.counts.number_of_dead_neuron = neuron_dead
        self.counts.number_of_cytokine = cytokine
        self.counts.number_of_cleaved_alpha_syn = alpha_cleaved
        self.counts.number_of_cleaved_tau = tau_cleaved
        self.counts.number_alpha_syn_oligomer = alpha_oligomer
        self.counts.number_tau_oligomer = tau_oligomer
        self.counts.number_of_resting_microglia = microglia_resting
        self.counts.number_of_active_microglia = microglia_active
        self.data_set.log(tick)

    def at_end(self):
        self.data_set.close()

    def start(self):
        self.runner.execute()


#run function
def run(params: Dict):
    global model
    model = Model(MPI.COMM_WORLD, params)
    model.start()

#execution function (when starting the program you have to pass the yaml file: setup.yaml)
if __name__ == "__main__":
    parser = parameters.create_args_parser()
    args = parser.parse_args()
    params = parameters.init_params(args.parameters_file, args.parameters)
    
    #for debug
    #stream = open("/home/sakurahanami120/Projects/Multiagent/MASL_project/setup.yaml", "r")
    #params = yaml.load(stream)
    #params = params

    run(params)

