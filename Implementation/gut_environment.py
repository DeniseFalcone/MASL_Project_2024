from typing import Dict, Tuple
from mpi4py import MPI
import numpy as np
from dataclasses import dataclass
from repast4py import core, random, space, schedule, logging, parameters
from repast4py import context as ctx
import repast4py
from repast4py.space import DiscretePoint as dpt
from repast4py.space import ContinuousPoint as cpt
import yaml
import numba
from numba import int32, int64
from numba.experimental import jitclass
import math


@dataclass
class Log:
    aep_active: int = 0
    aep_hyperactive: int = 0
    alpha_protein: int = 0
    tau_protein: int = 0
    alpha_cleaved: int = 0
    tau_cleaved: int = 0
    alpha_oligomer: int = 0
    tau_oligomer: int = 0
    barrier_permeability : int = 0
    microbiota_good_bacteria_class : int = 0
    microbiota_pathogenic_bacteria_class : int = 0
    microbiota_diversity_threshold : int = 0



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

        return np.stack((xs, ys, np.zeros(len(ys), dtype=np.int32)), axis=-1)
    

class AEP(core.Agent):

    TYPE = 0
    OFFSETS = np.array([-1, 1])

    def __init__(self, local_id: int, rank: int, pt: dpt):
        super().__init__(id=local_id, type=AEP.TYPE, rank=rank)
        self.pt = pt #current location
        self.state = params["aep_state"]["active"]

    def save(self) -> Tuple: 
        return (self.uid, self.state, (self.pt.x, self.pt.y))
    
    def hyperactive(self):
        self.state =  params["aep_state"]["hyperactive"]

    def step(self):
        grid = model.grid
        pt = grid.get_location(self)
        nghs = model.ngh_finder.find(self.pt.x, self.pt.y)
        
        #Randomly select a ngh cell
        random_index = np.random.randint(0, len(nghs))
        random_ngh_coordinates = nghs[random_index]
        random_ngh_dpt = dpt(random_ngh_coordinates[0], random_ngh_coordinates[1])
        random_ngh = model.grid.get_agent(random_ngh_dpt)

        if random_ngh is None:
            #Move the agent randomly
            model.move(self, random_ngh_coordinates[0], random_ngh_coordinates[1])
        else: 
            x = self.percept(nghs)
            if x is not None:
                self.hyperactive()
                self.cleave(x)

    #enzyme checks if the neighbour is a Protein
    def percept(self, nghs_coordinates):
        for ngh_coordinate in nghs_coordinates:
            ngh_dpt = dpt(ngh_coordinate[0], ngh_coordinate[1])
            ngh = model.grid.get_agent(ngh_dpt)
            if type(ngh) == Protein:
                return ngh
        return None    
              
    #enzyme cleaves the protein changing the state of the protein
    def cleave(self, ngh):        
        ngh.change_state()  


class Protein(core.Agent):

    TYPE = 1
    OFFSETS = np.array([-1, 1])

    def __init__(self, local_id: int, rank: int, pt: dpt, protein_name):
        super().__init__(id=local_id, type=Protein.TYPE, rank=rank)
        self.pt = pt 
        self.name = protein_name
        self.cleaved = False

    def save(self) -> Tuple: 
        return (self.uid, self.name, (self.pt.x, self.pt.y))

    def step(self):
        grid = model.grid
        pt = grid.get_location(self)
        nghs = model.ngh_finder.find(self.pt.x, self.pt.y)
        
        #Randomly select a ngh cell
        random_index = np.random.randint(0, len(nghs))
        random_ngh_coordinates = nghs[random_index]
        random_ngh_dpt = dpt(random_ngh_coordinates[0], random_ngh_coordinates[1])
        random_ngh = model.grid.get_agent(random_ngh_dpt)

        if random_ngh is None:
            #Move the agent randomly
            model.move(self, random_ngh_coordinates[0], random_ngh_coordinates[1])


    def change_state(self):
        if self.cleaved == False:
            self.cleaved = True


class CleavedProtein(core.Agent):
    TYPE = 2
    OFFSETS = np.array([-1, 1])

    def __init__(self, local_id: int, rank: int, pt: dpt, protein_name):
        super().__init__(id=local_id, type=Protein.TYPE, rank=rank)
        self.pt = pt #current location
        self.name = protein_name
        self.aggregates = False
        self.nghs_cleaved = []

    def save(self) -> Tuple: 
        return (self.uid, self.name, (self.pt.x, self.pt.y))

    def step(self):
        grid = model.grid
        pt = grid.get_location(self)
        nghs = model.ngh_finder.find(self.pt.x, self.pt.y)


        if (self.has_cleaved_ngh(nghs) == 1):
            #Randomly select a ngh cell
            random_index = np.random.randint(0, len(nghs))
            random_ngh_coordinates = nghs[random_index]
            random_ngh_dpt = dpt(random_ngh_coordinates[0], random_ngh_coordinates[1])
            random_ngh = model.grid.get_agent(random_ngh_dpt)

            if random_ngh is None:
                #Move the agent randomly
                model.move(self, random_ngh_coordinates[0], random_ngh_coordinates[1])

        elif (self.has_cleaved_ngh(nghs) == 4):
            self.aggregates = True
            for ngh in nghs:
                ngh_dpt = dpt(ngh[0], ngh[1])
                ngh_agent = model.grid.get_agent(ngh_dpt)
                if (type(ngh_agent) == CleavedProtein and self.name == ngh_agent.name):
                    self.nghs_cleaved.append(ngh_agent)

    
    #checks the state of the cleaved protein and if it's wrong it changes it back 
    def check_status(self):
        if self.aggregates == True:
            grid = model.grid
            pt = grid.get_location(self)
            nghs = model.ngh_finder.find(self.pt.x, self.pt.y)

            if (self.has_cleaved_ngh(nghs) < 4):   
                for ngh in nghs:
                    ngh_dpt = dpt(ngh[0], ngh[1])
                    ngh_agent = model.grid.get_agent(ngh_dpt)
                    if (type(ngh_agent) == CleavedProtein and self.name == ngh_agent.name):
                        ngh_agent.aggregates = False 
                return False
            else:
                return True


    def has_cleaved_ngh(self, nghs):
        nghs_cleaved = 0
        for ngh in nghs:
            ngh_dpt = dpt(ngh[0], ngh[1])
            ngh_agent = model.grid.get_agent(ngh_dpt)
            if (type(ngh_agent) == CleavedProtein and self.name == ngh_agent.name):
                nghs_cleaved += 1
        return nghs_cleaved    


class Oligomer(core.Agent):

    TYPE = 3
    OFFSETS = np.array([-1, 1])

    def __init__(self, local_id: int, rank: int, pt: dpt, oligomer_name):
        super().__init__(id=local_id, type=Protein.TYPE, rank=rank)
        self.pt = pt #current location
        self.name = oligomer_name

    def save(self) -> Tuple: 
        return (self.uid, self.name, (self.pt.x, self.pt.y))

    def step(self):
        pass


agent_cache = {}


#restore agents: enzyme, protein, cleavedProtein and Oligomer
def restore_agent(agent_data: Tuple):
    uid = agent_data[0]
    pt_tuple = agent_data[2]
    pt = dpt(*pt_tuple)

    #uid: 0 id, 1 type, 2 rank
    if uid[1] == AEP.TYPE:

        if uid in agent_cache:
            agent = agent_cache[uid]
        else: 
            agent = AEP(uid[0], uid[2], pt)
            agent_cache[uid] = agent
        
        agent.state = agent_data[1]
        return agent
    
    elif uid[1] == Protein.TYPE:

        if uid in agent_cache:
            agent = agent_cache[uid]
        else:
            name = agent_data[1]
            agent = Protein(uid[0], uid[2], pt, name)
            agent_cache[uid] = agent

        return agent
    
    elif uid[1] == CleavedProtein.TYPE:

        if uid in agent_cache:
            agent = agent_cache[uid]
        else:
            name = agent_data[1]
            agent = CleavedProtein(uid[0], uid[2], pt, name)
            agent_cache[uid] = agent

        return agent
    
    elif uid[1] == Oligomer.TYPE:

        if uid in agent_cache:
            agent = agent_cache[uid]
        else:
            name = agent_data[1]
            agent = Oligomer(uid[0], uid[2], pt, name)
            agent_cache[uid] = agent

        return agent


class Model:

    def __init__(self, comm: MPI.Intracomm, params: Dict):
        
        # TODO scheduler
        self.runner = schedule.init_schedule_runner(comm)
        self.runner.schedule_repeating_event(1, 1, self.step)
        #self.runner.schedule_repeating_event(1, 2, self.step2)
        self.runner.schedule_repeating_event(1, 1, self.log_agents)
        #self.runner.schedule_stop(params['stop.at'])
        self.runner.schedule_stop(50)
        schedule.runner().schedule_end_event(self.at_end)

        self.comm = comm
        self.context = ctx.SharedContext(comm)
        self.rank = comm.Get_rank()

        box = space.BoundingBox(0,params['world.width'], 0, params['world.height'],0,0)
        self.grid = space.SharedGrid(name='grid', bounds=box, borders=space.BorderType.Sticky, occupancy=space.OccupancyType.Single,
                                     buffer_size=2, comm=comm)
        self.context.add_projection(self.grid)
        self.space = space.SharedCSpace('space', bounds=box, borders=space.BorderType.Sticky,
                                        occupancy=space.OccupancyType.Single,
                                        buffer_size=2, comm=comm,
                                        tree_threshold=100)    
        self.context.add_projection(self.space)
        
        self.ngh_finder = GridNghFinder(0,0,box.xextent,box.yextent)

        self.rng = repast4py.random.default_rng

        #parallel distribution of the agents
        world_size = self.comm.Get_size()

        #gets the total number of different type of agents in that process.
        total_aep_count = params['aep_enzyme']
        pp_aep_count = int(total_aep_count/world_size)
        if self.rank < total_aep_count % world_size:
            pp_aep_count += 1

        total_tau_p_count = params['tau_proteins']
        pp_tau_p_count = int(total_tau_p_count/world_size)
        if self.rank < total_tau_p_count % world_size:
            pp_tau_p_count += 1
        
        total_alpha_p_count = params['alpha_syn_proteins']
        pp_alpha_p_count = int(total_alpha_p_count/world_size)
        if self.rank < total_alpha_p_count % world_size:
            pp_alpha_p_count += 1
        
        total_alpha_ol_count = params['alpha_syn_oligomers']
        pp_alpha_ol_count = int(total_alpha_ol_count/world_size)
        if self.rank < total_alpha_ol_count % world_size:
            pp_alpha_ol_count += 1

        total_tau_ol_count = params['tau_oligomers']
        pp_tau_ol_count = int(total_tau_ol_count/world_size)
        if self.rank < total_tau_ol_count % world_size:
            pp_tau_ol_count += 1

        self.added_agents_id = pp_aep_count + pp_alpha_p_count + pp_tau_p_count

        #add aep enzyme to the space
        for j in range(pp_aep_count):
            pt = self.grid.get_random_local_pt(self.rng)
            while(self.grid.get_agent(pt) is not None):
                pt = self.grid.get_random_local_pt(self.rng)
            aep_enzyme = AEP(j,self.rank,pt)
            self.context.add(aep_enzyme)
            self.grid.move(aep_enzyme, pt)

        #add tau proteins to the space
        for x in range(pp_tau_p_count):
            pt = self.grid.get_random_local_pt(self.rng)
            while(self.grid.get_agent(pt) is not None):
                pt = self.grid.get_random_local_pt(self.rng)
            tau_p = Protein(x + params['aep_enzyme'], self.rank, pt, params["protein_name"]["tau"])
            self.context.add(tau_p)
            self.grid.move(tau_p, pt)

        #add alpha syn proteins to the space
        for x in range(pp_alpha_p_count):
            pt = self.grid.get_random_local_pt(self.rng)
            while(self.grid.get_agent(pt) is not None):
                pt = self.grid.get_random_local_pt(self.rng)
            alpha_syn_p = Protein(x + params['aep_enzyme'] + params['tau_proteins'], self.rank, pt, params["protein_name"]["alpha_syn"])
            self.context.add(alpha_syn_p)
            self.grid.move(alpha_syn_p, pt)
        
        #add alpha syn oligomers to the space
        for i in range(pp_alpha_ol_count):
            pt = self.grid.get_random_local_pt(self.rng)
            while(self.grid.get_agent(pt) is not None):
                pt = self.grid.get_random_local_pt(self.rng)
            alpha_syn_oligomer = Oligomer(i + params['aep_enzyme'] + params['tau_proteins'] + params['alpha_syn_proteins'], self.rank, pt, params["protein_name"]["alpha_syn"])
            self.context.add(alpha_syn_oligomer)
            self.grid.move(alpha_syn_oligomer, pt)
            self.added_agents_id += 1 

        #add tau oligomers to the space
        for i in range(pp_tau_ol_count):
            pt = self.grid.get_random_local_pt(self.rng)
            while(self.grid.get_agent(pt) is not None):
                pt = self.grid.get_random_local_pt(self.rng)
            tau_oligomer = Oligomer(i + params['aep_enzyme'] + params['tau_proteins'] + params['alpha_syn_proteins'] + params['alpha_syn_oligomers'], self.rank, pt, params["protein_name"]["tau"])
            self.context.add(tau_oligomer)
            self.grid.move(tau_oligomer, pt)
            self.added_agents_id += 1 

        # Da aggiungere: 'barrier_permeability', 'microbiota_good_bacteria_class', 'microbiota_pathogenic_bacteria_class', 'microbiota_diversity_threshold'
        self.agent_logger = logging.TabularLogger(comm, params['agent_log_file'], ['tick', 'rank', 'aep_active', 'aep_hyperactive', 'alpha_protein', 'tau_protein', 'alpha_cleaved', 'tau_cleaved', 'alpha_oligomer', 'tau_oligomer'])
        self.agent_log = Log()
        #self.log_agents()

    #the model remove the agent from the context
    def remove_agent(self, agent):
        self.context.remove(agent)

    #the model add a cleaved protein agent in the context
    def add_cleaved_protein(self, protein_name): 
        self.added_agents_id += 1 
        pt = self.grid.get_random_local_pt(self.rng) 
        while(self.grid.get_agent(pt) is not None):
            pt = self.grid.get_random_local_pt(self.rng)
        #print("posizione scelta", pt,", id prot: ", self.added_agents_id)    
        cleaved_protein = CleavedProtein(self.added_agents_id, self.rank, pt, protein_name)
        self.context.add(cleaved_protein)    
        self.move(cleaved_protein, pt.x, pt.y)  

    #the model moves the agent passed at the position passed
    def move(self, agent, x, y): 
        self.space.move(agent, cpt(x, y)) 
        self.grid.move(agent, dpt(x, y)) 
        agent.pt =  dpt(x, y)
    
    #the model add an oligomer protein agent in the context
    def add_oligomer_protein(self, protein_name): 
        self.added_agents_id += 1 
        pt = self.grid.get_random_local_pt(self.rng) 
        while(self.grid.get_agent(pt) is not None):
            pt = self.grid.get_random_local_pt(self.rng)
        #print("posizione scelta", pt,", id prot: ", self.added_agents_id)    
        oligomer_protein = Oligomer(self.added_agents_id, self.rank, pt, protein_name)
        self.context.add(oligomer_protein)    
        self.move(oligomer_protein, pt.x, pt.y)  
    
    def step(self):
        print("Sono nello step")
        #makes the agents in the context execute their step.
        for agent in self.context.agents():
            if(type(agent) == AEP):
                agent.step()
            elif(type(agent) == Protein):
                agent.step()
            elif(type(agent) == CleavedProtein):
                agent.step()        

        
        protein_to_remove = []
        all_true_cleaved_aggregates = []
        
        #gets all the agents to remove
        for agent in self.context.agents():
            if(type(agent) == Protein and agent.cleaved == True):
                protein_to_remove.append(agent)
            elif(type(agent) == CleavedProtein and agent.aggregates == True):
                all_true_cleaved_aggregates.append(agent)     

       
        #print("proteine da rimuovere", protein_to_remove)

        #removes the protein that are not needed anymore
        for agent in protein_to_remove:
            protein_name = agent.name
            self.remove_agent(agent)        
            #print("agent: ", agent.uid)         
            self.add_cleaved_protein(protein_name)
            self.add_cleaved_protein(protein_name)
     

        #removes the cleaved protein that are not needed anymore
        for agent in all_true_cleaved_aggregates:
            if (self.context.agent(agent.uid) != None):
                if (agent.check_status()):
                    protein_name = agent.name  
                    for i in range(0,4):
                        x = agent.nghs_cleaved[i]
                        self.remove_agent(x)  
                    self.add_oligomer_protein(protein_name)
                else:
                    agent.nghs_cleaved = []

        '''
        #PER STAMPARE
        print("STEP")
        for agent in self.context.agents():
            print("Agenti dopo di ogni cosa: ", agent," , ", agent.TYPE, ", posizione: ", agent.pt)    
        
        cleaved_protein_count_alpha = 0
        cleaved_protein_count_tau = 0
        protein_count = 0
        oligomer_count_alpha = 0
        oligomer_count_tau = 0
        
        for agent in self.context.agents():
            if(type(agent) == Oligomer):
                if (agent.name == 1):
                    oligomer_count_alpha += 1
                else:
                    oligomer_count_tau += 1
            if(type(agent) == CleavedProtein):
                if (agent.name == 1):
                    cleaved_protein_count_alpha += 1
                else:
                    cleaved_protein_count_tau += 1
            elif(type(agent) == Protein):
                protein_count = protein_count + 1

        print("Proteine intere rimanenti", protein_count)          
        print("cleaved proteine rimanenti tau", cleaved_protein_count_tau)  
        print("cleaved proteine rimanenti alpha", cleaved_protein_count_alpha)  
        print("oligomeri creati tau", oligomer_count_tau)  
        print("oligomeri creati alpha", oligomer_count_alpha)              
        '''

        self.context.synchronize(restore_agent)
        tick = self.runner.schedule.tick

    def log_agents(self):
        tick = self.runner.schedule.tick

        aep_active = 0
        aep_hyperactive = 0
        alpha_protein = 0
        tau_protein = 0
        alpha_cleaved = 0
        tau_cleaved = 0
        alpha_oligomer = 0
        tau_oligomer = 0
        #barrier_permeability = 0
        #microbiota_good_bacteria_class = 0
        #microbiota_pathogenic_bacteria_class = 0
        #microbiota_diversity_threshold = 0


        for agent in self.context.agents():
            if(type(agent) == Oligomer):
                if (agent.name == 1):
                    alpha_oligomer += 1
                else:
                    tau_oligomer += 1
            if(type(agent) == CleavedProtein):
                if (agent.name == 1):
                    alpha_cleaved += 1
                else:
                    tau_cleaved += 1
            elif(type(agent) == Protein):
                if (agent.name == 1):
                    alpha_protein += 1
                else:
                    tau_protein += 1
            elif(type(agent) == AEP):
                if(agent.state == 0):
                    aep_active += 1
                else: 
                    aep_hyperactive += 1
        
        self.agent_log.aep_active = aep_active
        self.agent_log.aep_hyperactive = aep_hyperactive
        self.agent_log.alpha_protein = alpha_protein
        self.agent_log.tau_protein = tau_protein
        self.agent_log.alpha_cleaved = alpha_cleaved
        self.agent_log.tau_cleaved = tau_cleaved
        self.agent_log.alpha_oligomer = alpha_oligomer
        self.agent_log.tau_oligomer = tau_oligomer

        #da aggiungere i quattro mancanti
        self.agent_logger.log_row(tick, self.context.comm.Get_rank(), self.agent_log.aep_active, self.agent_log.aep_hyperactive, self.agent_log.alpha_protein, self.agent_log.tau_protein, self.agent_log.alpha_cleaved, self.agent_log.tau_cleaved, self.agent_log.alpha_oligomer, self.agent_log.tau_oligomer)
        self.agent_logger.write()
    
    def at_end(self):
        self.agent_logger.close()

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