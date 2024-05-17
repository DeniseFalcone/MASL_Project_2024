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
    aep_active: int = 0
    aep_hyperactive: int = 0
    alpha_protein: int = 0
    tau_protein: int = 0
    alpha_cleaved: int = 0
    tau_cleaved: int = 0
    alpha_oligomer: int = 0
    tau_oligomer: int = 0
    barrier_impermeability : int = 0
    microbiota_good_bacteria_class : int = 0
    microbiota_pathogenic_bacteria_class : int = 0

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

class AEP(core.Agent):

    TYPE = 0
    OFFSETS = np.array([-1, 1])

    def __init__(self, local_id: int, rank: int, pt: dpt):
        super().__init__(id=local_id, type=AEP.TYPE, rank=rank)
        self.pt = pt #current location
        self.state = params["aep_state"]["active"]

    def save(self) -> Tuple:
        return (self.uid, self.state, (self.pt.x, self.pt.y))
    
    def is_hyperactive(self):
        if self.state == params["aep_state"]["active"]:
            return False
        else:
            return True

    def step(self):
        #gets the max 8 neighbours of the agent          
        nghs_coords = model.ngh_finder.find(self.pt.x, self.pt.y)
        dpt_array = self.percepts(nghs_coords)
        if type(dpt_array) == dpt:
            if(self.is_hyperactive() == True):
                self.cleave(dpt_array)
        elif len(dpt_array) == 0:
            pass
        else: 
            random_index = np.random.randint(0, len(dpt_array))
            chosen_dpt = dpt_array[random_index]
            model.move(self, chosen_dpt)


    # if the aep agent has a protein returns the coords of that protein, 
    #  otherwise it returns available grid cells where it can move
    def percepts(self, nghs_coords):
        protein_dpt = []
        for ngh_coords in nghs_coords:
            ngh_dpt = dpt(ngh_coords[0], ngh_coords[1])
            ngh = model.grid.get_agent(ngh_dpt)
            if ngh is None:
                protein_dpt.append(ngh_dpt)
            elif type(ngh) == Protein:
                return ngh_dpt   
        return protein_dpt   
    
    #changes the state of the protein to cleave
    def cleave(self, protein_coords):
        protein = model.grid.get_agent(protein_coords)
        protein.change_state()

class Protein(core.Agent):

    TYPE = 1
    OFFSETS = np.array([-1, 1])

    def __init__(self, local_id: int, rank: int, pt: dpt, protein_name):
        super().__init__(id=local_id, type=Protein.TYPE, rank=rank)
        self.pt = pt 
        self.name = protein_name
        self.toCleave = False

    def save(self) -> Tuple: 
        #(self.pt.x, self.pt.y)
        return (self.uid, self.name, (self.pt.x, self.pt.y), self.toCleave)
    
    def step(self):
        nghs_coords = model.ngh_finder.find(self.pt.x, self.pt.y)
        empty_dpt = []
        for ngh_coords in nghs_coords:
            ngh_dpt = dpt(ngh_coords[0], ngh_coords[1])
            ngh = model.grid.get_agent(ngh_dpt)
            if ngh is None:
                empty_dpt.append(ngh_dpt)

        if len(empty_dpt) == 0:
            pass
        else: 
            random_index = np.random.randint(0, len(empty_dpt))
            chosen_dpt = empty_dpt[random_index]
            model.move(self, chosen_dpt)

    def change_state(self):
        if self.toCleave == False:
            self.toCleave = True

class CleavedProtein(core.Agent):
    TYPE = 2
    OFFSETS = np.array([-1, 1])

    def __init__(self, local_id: int, rank: int, pt: dpt, cleaved_protein_name):
        super().__init__(id=local_id, type=CleavedProtein.TYPE, rank=rank)
        self.pt = pt #current location
        self.name = cleaved_protein_name
        self.toAggregate = False
        self.alreadyAggregate = False
        #self.groupToAggregate = []
        
    def save(self) -> Tuple: 
        return (self.uid, self.name, (self.pt.x, self.pt.y), self.toAggregate, self.pt_from_groupToAggregate(), self.alreadyAggregate)
    
    def get_cleaved_nghs(self):
        nghs_coords = model.ngh_finder.find(self.pt.x, self.pt.y)
        cleavedProteins = []
        for ngh_coords in nghs_coords:
            ngh_dpt = dpt(ngh_coords[0], ngh_coords[1])
            agent = model.grid.get_agent(ngh_dpt)
            if (type(agent) == CleavedProtein and self.name == agent.name):
                cleavedProteins.append(agent)
        return cleavedProteins



    def step(self):
        if self.alreadyAggregate == True or self.toAggregate == True:
            pass
        else:
            nghs_coords = model.ngh_finder.find(self.pt.x, self.pt.y)
            #print("Nghs coord in step agent ", self.uid, ": ", nghs_coords)
            cleaved_nghs_number = self.check_cleaved_nghs(nghs_coords)
            if cleaved_nghs_number == 0:
                empty_coords_array = self.get_empty_nghs(nghs_coords)
                if len(empty_coords_array) > 0:
                    random_index = np.random.randint(0, len(empty_coords_array))
                    chosen_dpt = empty_coords_array[random_index]
                    #TODO check if it works
                    model.move(self, chosen_dpt)
            elif cleaved_nghs_number >= 3:               
                self.change_state()
            else:
                self.change_group_aggregate_status()
            

    def change_group_aggregate_status(self):
        nghs_coords = model.ngh_finder.find(self.pt.x, self.pt.y)
        for ngh_coords in nghs_coords:
            ngh_dpt = dpt(ngh_coords[0], ngh_coords[1])
            agent = model.grid.get_agent(ngh_dpt)
            if agent is not None:
                agent.alreadyAggregate = False

    def is_valid(self):
        cont = 0
        for agent in self.get_cleaved_nghs():
            if (agent.alreadyAggregate == True):
                cont += 1

        if cont >= 3:
            return True
        else: 
            return False
    
    def change_state(self):
        if self.toAggregate == False:
            self.toAggregate = True

    def check_cleaved_nghs(self, nghs_coords):
        cont = 0
        for ngh_coords in nghs_coords:
            ngh_dpt = dpt(ngh_coords[0], ngh_coords[1])
            ngh = model.grid.get_agent(ngh_dpt)
            if (type(ngh) == CleavedProtein and self.name == ngh.name):
                #print("Vicino: ", ngh.uid)
                #if ngh not in self.groupToAggregate:
                if ngh.toAggregate == False and ngh.alreadyAggregate == False:
                    #self.groupToAggregate.append(ngh)
                    ngh.alreadyAggregate = True
                    cont += 1
        return cont

    def get_empty_nghs(self, nghs_coords):
        empty_nghs = []
        for ngh_coords in nghs_coords:
            ngh_dpt = dpt(ngh_coords[0], ngh_coords[1])
            ngh = model.grid.get_agent(ngh_dpt)
            if ngh is None:
                empty_nghs.append(ngh_dpt)
        return empty_nghs

class Oligomer(core.Agent):

    TYPE = 3
    OFFSETS = np.array([-1, 1])

    def __init__(self, local_id: int, rank: int, pt: dpt, oligomer_name):
        super().__init__(id=local_id, type=Oligomer.TYPE, rank=rank)
        self.pt = pt #current location
        self.name = oligomer_name

    def save(self) -> Tuple: 
        return (self.uid, self.name, (self.pt.x, self.pt.y))
    
    def step(self):
        pass

class ExternalInput(core.Agent):
    TYPE = 4

    def __init__(self, local_id: int, rank: int, pt: dpt):
        super().__init__(id=local_id, type=ExternalInput.TYPE, rank=rank)
        self.pt = pt
        possible_types = [params["external_input"]["diet"],params["external_input"]["antibiotics"],params["external_input"]["stress"]]
        random_index = np.random.randint(0, len(possible_types))
        input_name = possible_types[random_index]
        self.input_name = input_name

    #if the external input is "diet" or "stress" then the microbiota bacteria decrease in good bacteria classes and increase in pathogenic ones.
    #otherwise it only decreases the good bacteria classes.
    #random percentage to change the params of the microbiota
    def step(self):
        if model.barrier_impermeability >= model.barrier_permeability_threshold_stop:
            if self.input_name == params["external_input"]["diet"]:
                to_remove = int((model.microbiota_good_bacteria_class * np.random.uniform(0, 3))/100)
                #while model.microbiota_good_bacteria_class - to_remove <= 0:
                #    to_remove = to_remove/2
                model.microbiota_good_bacteria_class = model.microbiota_good_bacteria_class - to_remove
                to_add = int((params["microbiota_pathogenic_bacteria_class"] * np.random.uniform(0, 3))/100)
                model.microbiota_pathogenic_bacteria_class = model.microbiota_pathogenic_bacteria_class + to_add
            elif self.input_name == params["external_input"]["antibiotics"]:
                to_remove = int((model.microbiota_good_bacteria_class * np.random.uniform(0, 5))/100)
                model.microbiota_good_bacteria_class = model.microbiota_good_bacteria_class - to_remove
                to_add = int((params["microbiota_pathogenic_bacteria_class"] * np.random.uniform(0, 2))/100)
                model.microbiota_pathogenic_bacteria_class = model.microbiota_pathogenic_bacteria_class + to_add
            else:
                value = int((model.microbiota_good_bacteria_class * np.random.uniform(0, 3))/100)
                model.microbiota_good_bacteria_class = model.microbiota_good_bacteria_class - value
                value = int((params["microbiota_pathogenic_bacteria_class"] * np.random.uniform(0, 3))/100)
                model.microbiota_pathogenic_bacteria_class = model.microbiota_pathogenic_bacteria_class + value


class Treatment(core.Agent):
    TYPE = 5

    def __init__(self, local_id: int, rank: int, pt: dpt):
        super().__init__(id=local_id, type=Treatment.TYPE, rank=rank)
        self.pt = pt
        possible_types = [params["treatment_input"]["diet"],params["treatment_input"]["probiotics"]]
        random_index = np.random.randint(0, len(possible_types))
        input_name = possible_types[random_index]
        self.input_name = input_name

    #if the external input is "diet" or "stress" then the microbiota bacteria decrease in good bacteria classes and increase in pathogenic ones.
    #otherwise it only decreases the good bacteria classes.
    #random percentage to change the params of the microbiota
    def step(self):
         if model.barrier_impermeability < model.barrier_permeability_threshold_start:
            if self.input_name == params["treatment_input"]["diet"]:
                to_add = int((params["microbiota_good_bacteria_class"] * np.random.uniform(0, 3))/100)
                model.microbiota_good_bacteria_class = model.microbiota_good_bacteria_class + to_add
                to_remove = int((model.microbiota_pathogenic_bacteria_class * np.random.uniform(0, 2))/100)
                model.microbiota_pathogenic_bacteria_class = model.microbiota_pathogenic_bacteria_class - to_remove
            elif self.input_name == params["treatment_input"]["probiotics"]:
                to_add = int((params["microbiota_good_bacteria_class"] * np.random.uniform(0, 4))/100)
                model.microbiota_good_bacteria_class = model.microbiota_good_bacteria_class + to_add
                to_remove = int((model.microbiota_pathogenic_bacteria_class * np.random.uniform(0, 4))/100)
                model.microbiota_pathogenic_bacteria_class = model.microbiota_pathogenic_bacteria_class - to_remove


agent_cache = {}


#restore agents: Enzyme, Protein, CleavedProtein and Oligomer
def restore_agent(agent_data: Tuple):
    #uid: 0 id, 1 type, 2 rank
    uid = agent_data[0]
    pt_tuple = agent_data[2]
    pt = dpt(*pt_tuple)
    agent = None
    if uid[1] == AEP.TYPE:
        if uid in agent_cache:
            agent = agent_cache[uid]
        else: 
            agent = AEP(uid[0], uid[2], pt)
            agent_cache[uid] = agent        
        agent.state = agent_data[1]
        agent.pt = pt
    elif uid[1] == Protein.TYPE:
        if uid in agent_cache:
            agent = agent_cache[uid]
        else: 
            agent = Protein(uid[0], uid[2], agent_data[1], pt)
            agent_cache[uid] = agent        
        agent.toCleave = agent_data[3]
        agent.pt = pt
    elif uid[1] == CleavedProtein.TYPE:
        if uid in agent_cache:
            agent = agent_cache[uid]
        else: 
            agent = CleavedProtein(uid[0], uid[2], agent_data[1], pt)
            agent_cache[uid] = agent        
        agent.toAggregate = agent_data[3]
        agent.groupToAggregate = [restore_agent_from_pt(pt) for pt in agent_data[4]]
        agent.pt = pt
        agent.alreadyAggregate = agent_data[5]
    elif uid[1] == Oligomer.TYPE:
        if uid in agent_cache:
            agent = agent_cache[uid]
        else: 
            agent = Oligomer(uid[0], uid[2], agent_data[1], pt)
            agent_cache[uid] = agent        
        agent.pt = pt
    return agent

def restore_agent_from_pt(pt):
    agent_pt = dpt(*pt)
    return model.grid.get_agent(agent_pt) 



class Model():
    def __init__(self, comm: MPI.Intracomm, params: Dict):        
        self.comm = comm
        self.context = ctx.SharedContext(comm)
        self.rank = comm.Get_rank()

        box = space.BoundingBox(0,params['world.width']-1, 0, params['world.height']-1,0,0)
        self.grid = space.SharedGrid(name='grid', bounds=box, borders=space.BorderType.Sticky, occupancy=space.OccupancyType.Single,
                                     buffer_size=1, comm=comm)
        self.context.add_projection(self.grid)
        self.space = space.SharedCSpace('space', bounds=box, borders=space.BorderType.Sticky,
                                        occupancy=space.OccupancyType.Single,
                                        buffer_size=1, comm=comm,
                                        tree_threshold=100)    
        self.context.add_projection(self.space)
        self.ngh_finder = GridNghFinder(0,0,box.xextent,box.yextent)

        self.runner = schedule.init_schedule_runner(comm)
        self.runner.schedule_repeating_event(1, 1, self.step)  
        self.runner.schedule_repeating_event(1, 2, self.microbiota_dysbiosis_step) 
        self.runner.schedule_repeating_event(1, 5, self.move_cleaved_protein_step)     
        self.runner.schedule_stop(params['stop.at'])
        self.runner.schedule_end_event(self.at_end)

        repast4py.random.seed = params['seed']
        self.rng = repast4py.random.default_rng

        self.counts = Log()
        loggers = logging.create_loggers(self.counts, op=MPI.SUM, rank=self.rank)
        self.data_set = logging.ReducingDataSet(loggers, self.comm, params['agent_log_file'], buffer_size=1)

        #parallel distribution of the agents
        world_size = self.comm.Get_size()

        #dysbiosis inizialization
        self.microbiota_good_bacteria_class = params["microbiota_good_bacteria_class"]
        self.microbiota_pathogenic_bacteria_class = params["microbiota_pathogenic_bacteria_class"]
        self.microbiota_diversity_threshold = params["microbiota_diversity_threshold"]
        self.barrier_impermeability = params["barrier_impermeability"]
        self.barrier_permeability_threshold_stop = params["barrier_permeability_threshold_stop"]
        self.barrier_permeability_threshold_start = params["barrier_permeability_threshold_start"]

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

        total_ext_count = params['external_input_number']
        pp_ext_count = int(total_ext_count/world_size)
        if self.rank < total_ext_count % world_size:
            pp_ext_count += 1

        if params['treatment'] == True:
            total_treatment_count = params['treatment_input_number']
            pp_treatment_count = int(total_treatment_count/world_size)
            if self.rank < total_treatment_count % world_size:
                pp_treatment_count += 1

            self.added_agents_id = pp_aep_count + pp_alpha_p_count + pp_tau_p_count + pp_ext_count + pp_treatment_count
        else:
            self.added_agents_id = pp_aep_count + pp_alpha_p_count + pp_tau_p_count + pp_ext_count

        #print("Inizializzazione agenti processo ", self.rank, ": ")
        #add aep enzyme to the space
        for j in range(pp_aep_count):
            pt = self.grid.get_random_local_pt(self.rng)
            while(self.grid.get_agent(pt) is not None):
                pt = self.grid.get_random_local_pt(self.rng)
            aep_enzyme = AEP(j,self.rank,pt)
            self.context.add(aep_enzyme)
            self.move(aep_enzyme, pt)
            #print("Agent: ", aep_enzyme, "Position: ", aep_enzyme.pt)

        #add tau proteins to the space
        for x in range(pp_tau_p_count):
            pt = self.grid.get_random_local_pt(self.rng)
            while(self.grid.get_agent(pt) is not None):
                pt = self.grid.get_random_local_pt(self.rng)
            tau_p = Protein(x + params['aep_enzyme'], self.rank, pt, params["protein_name"]["tau"])
            self.context.add(tau_p)
            self.move(tau_p, pt)
            #print("Agent: ", tau_p, "Position: ", tau_p.pt)


        #add alpha syn proteins to the space
        for x in range(pp_alpha_p_count):
            pt = self.grid.get_random_local_pt(self.rng)
            while(self.grid.get_agent(pt) is not None):
                pt = self.grid.get_random_local_pt(self.rng)
            alpha_syn_p = Protein(x + params['aep_enzyme'] + params['tau_proteins'], self.rank, pt, params["protein_name"]["alpha_syn"])
            self.context.add(alpha_syn_p)
            self.move(alpha_syn_p, pt)
            #print("Agent: ", alpha_syn_p, "Position: ", alpha_syn_p.pt)

        #add external input to the space
        for x in range(pp_ext_count):
            pt = self.grid.get_random_local_pt(self.rng)
            while(self.grid.get_agent(pt) is not None):
                pt = self.grid.get_random_local_pt(self.rng)
            ext = ExternalInput(x + params['aep_enzyme'] + params['tau_proteins'] + params['alpha_syn_proteins'], self.rank, pt)
            self.context.add(ext)
            self.move(ext, pt)

        #add treatment input to the space
        if params["treatment"] == True:
            for x in range(pp_treatment_count):
                pt = self.grid.get_random_local_pt(self.rng)
                while(self.grid.get_agent(pt) is not None):
                    pt = self.grid.get_random_local_pt(self.rng)
                treatment = Treatment(x + params['aep_enzyme'] + params['tau_proteins'] + params['alpha_syn_proteins'] + params['external_input_number'], self.rank, pt)
                self.context.add(treatment)
                self.move(treatment, pt)
        
        #add alpha syn oligomers to the space
        for i in range(pp_alpha_ol_count):
            pt = self.grid.get_random_local_pt(self.rng)
            while(self.grid.get_agent(pt) is not None):
                pt = self.grid.get_random_local_pt(self.rng)
            alpha_syn_oligomer = Oligomer(i + params['aep_enzyme'] + params['tau_proteins'] + params['alpha_syn_proteins'] + params['external_input_number'] + params['treatment_input_number'], self.rank, pt, params["protein_name"]["alpha_syn"])
            self.context.add(alpha_syn_oligomer)
            self.move(alpha_syn_oligomer, pt)
            self.added_agents_id += 1 
            #print("Agent: ", alpha_syn_oligomer, "Position: ", alpha_syn_oligomer.pt)


        #add tau oligomers to the space
        for i in range(pp_tau_ol_count):
            pt = self.grid.get_random_local_pt(self.rng)
            while(self.grid.get_agent(pt) is not None):
                pt = self.grid.get_random_local_pt(self.rng)
            tau_oligomer = Oligomer(i + params['aep_enzyme'] + params['tau_proteins'] + params['alpha_syn_proteins'] + params['alpha_syn_oligomers'], self.rank, pt, params["protein_name"]["tau"])
            self.context.add(tau_oligomer)
            self.move(tau_oligomer, pt)
            self.added_agents_id += 1 
            #print("Agent: ", tau_oligomer, "Position: ", tau_oligomer.pt)

    def move_cleaved_protein_step(self):
        for agent in self.context.agents():
            if type(agent) == CleavedProtein:
                #print("Cleaved protein con aggregate: ", agent.alreadyAggregate, " in pos: ", agent.pt)
                if agent.alreadyAggregate == False:
                    pt = self.grid.get_random_local_pt(self.rng)
                    while(self.grid.get_agent(pt) is not None):
                        pt = self.grid.get_random_local_pt(self.rng)
                    self.move(agent, pt)
                    #print("Cleaved protein con aggregate: ", agent.alreadyAggregate, " in pos: ", agent.pt)


    #increase the dysbiosis in the system and, after a certain threshold, it starts to hyperactivate the AEP enzymes.
    def microbiota_dysbiosis_step(self):
        if self.microbiota_good_bacteria_class - self.microbiota_pathogenic_bacteria_class <= self.microbiota_diversity_threshold:
            value_decreased = int((params["barrier_impermeability"]*np.random.randint(0,6))/100) 
            if self.barrier_impermeability - value_decreased <= 0:
                self.barrier_impermeability = 0
            else:
                self.barrier_impermeability = self.barrier_impermeability - value_decreased
            number_of_aep_to_hyperactivate = value_decreased
            cont = 0
            for agent in self.context.agents(agent_type=0):
                if agent.state == params["aep_state"]["active"] and cont < number_of_aep_to_hyperactivate:
                    agent.state = params["aep_state"]["hyperactive"]  
                    cont += 1
                elif cont == number_of_aep_to_hyperactivate:
                    break
        else:
            if self.barrier_impermeability < params["barrier_impermeability"]:
                value_increased = int((params["barrier_impermeability"]*np.random.randint(0,4))/100) 
                if (self.barrier_impermeability + value_increased) <= params["barrier_impermeability"]:
                    self.barrier_impermeability = self.barrier_impermeability + value_increased

    def step(self):
        #self.context.synchronize(restore_agent)
        print("Step in model")
        print("Process: ", self.rank)
        #print("Agents' Positions Before Step:")
        #for agent in self.context.agents():
        #    print(f"Agent ({agent.uid}) at position ({agent.pt.x}, {agent.pt.y})")
        #    if type(agent) == CleavedProtein:
        #        print(f"Agent ({agent.uid}) con stati ToAggregate e AlreadyAggregate ({agent.toAggregate}, {agent.alreadyAggregate})")
        
        for agent in self.context.agents():
            agent.step()

        protein_to_remove = []
        all_true_cleaved_aggregates = []
        
        #gets all the agents to remove
        for agent in self.context.agents():
            if(type(agent) == Protein and agent.toCleave == True):
                protein_to_remove.append(agent)
            elif(type(agent) == CleavedProtein and agent.toAggregate == True):
                #print("Cleaved: ", agent.uid, ", aggregate: ", agent.toAggregate, ", alreadyAggregate: ", agent.alreadyAggregate)
                all_true_cleaved_aggregates.append(agent)      
        #print("Elenco di tutte le cleaved da rimuovere: ", all_true_cleaved_aggregates)                

        #removes the protein that are not needed anymore
        for agent in protein_to_remove:
            protein_name = agent.name
            self.remove_agent(agent)                
            self.add_cleaved_protein(protein_name)
            self.add_cleaved_protein(protein_name)

        #removes cleavedProteins
        for agent in all_true_cleaved_aggregates:
            if (self.context.agent(agent.uid) is not None and agent.toAggregate == True):
                if agent.is_valid() == True:
                    protein_name = agent.name 
                    cont = 0 
                    #print("Agente: ", agent.uid, " Vicini porca la miseria: ", agent.groupToAggregate)
                    for x in agent.get_cleaved_nghs():
                        if x.alreadyAggregate == True:
                            if(cont < 3):
                                self.remove_agent(x)
                                cont +=1
                            else:
                                x.alreadyAggregate = False
                                x.toAggregate = False
                                cont += 1
                    self.add_oligomer_protein(protein_name)
                    self.remove_agent(agent)

        self.context.synchronize(restore_agent)      
        self.log_counts()        

    def remove_agent(self, agent):
        self.context.remove(agent)

    def add_cleaved_protein(self,protein_name):
        self.added_agents_id += 1
        pt = self.grid.get_random_local_pt(self.rng)
        while(self.grid.get_agent(pt) is not None):
            pt = self.grid.get_random_local_pt(self.rng)  
        cleaved_protein = CleavedProtein(self.added_agents_id, self.rank, pt, protein_name)   
        self.context.add(cleaved_protein)   
        self.move(cleaved_protein, cleaved_protein.pt)

    
    def add_oligomer_protein(self, oligomer_name):
        self.added_agents_id += 1 
        pt = self.grid.get_random_local_pt(self.rng) 
        while(self.grid.get_agent(pt) is not None):
            pt = self.grid.get_random_local_pt(self.rng)
        oligomer_protein = Oligomer(self.added_agents_id, self.rank, pt, oligomer_name)
        self.context.add(oligomer_protein)            
        self.move(oligomer_protein, oligomer_protein.pt)        
    
    def move(self, agent, dpt):
        self.space.move(agent, cpt(dpt.x, dpt.y)) 
        self.grid.move(agent, dpt) 
        agent.pt =  dpt

    def log_counts(self):
        tick = self.runner.schedule.tick

        aep_active = 0
        aep_hyperactive = 0
        alpha_protein = 0
        tau_protein = 0
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
            elif(type(agent) == Protein): 
                if (agent.name == params["protein_name"]["alpha_syn"]): 
                    alpha_protein += 1 
                else: 
                    tau_protein += 1 
            elif(type(agent) == AEP): 
                if(agent.state == params["aep_state"]["active"]): 
                    aep_active += 1 
                else:  
                    aep_hyperactive += 1
        
        self.counts.aep_active = aep_active
        self.counts.aep_hyperactive = aep_hyperactive
        self.counts.alpha_protein = alpha_protein
        self.counts.tau_protein = tau_protein
        self.counts.alpha_cleaved = alpha_cleaved
        self.counts.tau_cleaved = tau_cleaved
        self.counts.alpha_oligomer = alpha_oligomer
        self.counts.tau_oligomer = tau_oligomer
        self.counts.microbiota_good_bacteria_class = self.microbiota_good_bacteria_class
        self.counts.microbiota_pathogenic_bacteria_class = self.microbiota_pathogenic_bacteria_class
        self.counts.barrier_impermeability = self.barrier_impermeability
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