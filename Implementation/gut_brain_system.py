from typing import Dict, Tuple
from mpi4py import MPI
import numpy as np
from dataclasses import dataclass
from repast4py import core, space, schedule, logging, parameters
from repast4py import context as ctx
import repast4py
import repast4py.random
from repast4py.space import DiscretePoint as dpt
import numba
from numba import int32, int64
from numba.experimental import jitclass
import pygame

# Class to make the graphics of the simulation
class GUI:
    def __init__(self, width, height, gut_context, brain_context, grid_dimensions=(100, 100)):
        self.background_color = (202, 187, 185) 
        self.border_color = (255, 255, 255) 
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.font = pygame.font.Font(None, 36)
        self.running = True
        self.gut_context = gut_context
        self.brain_context = brain_context
        self.grid_width, self.grid_height = grid_dimensions
        self.paused = False
        self.button_rects = []

    # Function to update the screen after each tick
    def update(self, gut_context, brain_context):
        # Update contexts
        self.gut_context, self.brain_context = gut_context, brain_context
        
        # Fill background and draw border rectangle
        self.screen.fill(self.background_color)
        inner_rect = (50, 50, self.width - 100, self.height - 300)
        pygame.draw.rect(self.screen, self.border_color, inner_rect)
        
        # Draw section titles
        text_y_position = inner_rect[1] - 30
        self._draw_centered_text("Gut Environment", self.width // 4, text_y_position)
        self._draw_centered_text("Brain Environment", 3 * self.width // 4, text_y_position)

        # Draw separating line
        pygame.draw.line(self.screen, (0, 0, 0), 
                        (self.width // 2, inner_rect[1]), 
                        (self.width // 2, inner_rect[1] + inner_rect[3]), 4)

        # Draw buttons and legend
        self.draw_buttons()
        self.draw_legend()

        # Define areas and draw agents
        gut_area = (50, 50, self.width // 2 - 47, self.height - 300)
        brain_area = (self.width // 2 + 3, 50, self.width // 2 - 50, self.height - 300)
        self._draw_context_agents(self.gut_context, gut_area)
        self._draw_context_agents(self.brain_context, brain_area)

    def _draw_centered_text(self, text, x_center, y):
        rendered_text = self.font.render(text, True, (0, 0, 0))
        text_x_position = x_center - rendered_text.get_width() // 2
        self.screen.blit(rendered_text, (text_x_position, y))

    def _draw_context_agents(self, context, area):
        self.draw_agents(context.agents(), area)

    # Function to draw agents on the screen
    def draw_agents(self, agents, area):
        radius = 5

        for agent in agents:
            x_center = area[0] + (agent.pt.x / self.grid_width) * area[2]
            y_center = area[1] + (agent.pt.y / self.grid_height) * area[3]

            # Adjust x and y to keep the entire circle within the area
            x = max(area[0] + radius, min(x_center, area[0] + area[2] - radius))
            y = max(area[1] + radius, min(y_center, area[1] + area[3] - radius))

            
            color = self.get_agent_color(agent)
            pygame.draw.circle(self.screen, color, (int(x), int(y)), radius)

    # Function to get the color of an agent based on its type and state
    def get_agent_color(self, agent):
        if agent.uid[1] == AEP.TYPE:
            if agent.state == params["aep_state"]["active"]:
                color = (147, 112, 219)
            else:
                color = (128, 0, 128)
        elif agent.uid[1] == Protein.TYPE:
            if agent.name == params["protein_name"]["tau"]:
                color = (173, 216, 230)  # Light Blue
            else:
                color = (255, 255, 128)  # Light Yellow 
        elif agent.uid[1] == CleavedProtein.TYPE:
            if agent.name == params["protein_name"]["tau"]:
                color = (113, 166, 210)  # Darker Blue               
            else:
                color = (225, 225, 100)  # Darker Yellow
        elif agent.uid[1] == Oligomer.TYPE:
            if agent.name == params["protein_name"]["tau"]:
                color = (0, 0, 255) # Blue
            else:
                color = (255, 255, 0) # Yellow
        elif agent.uid[1] == ExternalInput.TYPE:
            color = (169, 169, 169)  # Dark Grey
        elif agent.uid[1] == Treatment.TYPE:
            color = (211, 211, 211)  # Light Grey
        elif agent.uid[1] == Microglia.TYPE:
            if agent.state == params["microglia_state"]["resting"]:
                color = (144, 238, 144)  # Light Green 
            else:
                color = (0, 100, 0)  # Dark Green
        elif agent.uid[1] == Neuron.TYPE:
            if agent.state == params["neuron_state"]["healthy"]:
                color = (255, 105, 180)  # Pink
            elif agent.state == params["neuron_state"]["damaged"]:
                color = (255, 69, 0) # Orange-Red
            else:
                color = (0, 0, 0) # Black
        elif agent.uid[1] == Cytokine.TYPE:
            if agent.state == params["cyto_state"]["pro_inflammatory"]:
                color = (255, 0, 0) # Red
            else:
                color = (0, 255, 255)  # Cyan
        return color

    # Function to draw the legend on the screen
    def draw_legend(self):
        # Define legend position and size
        legend_x = 60
        legend_y = self.height - 250  
        legend_radius = 6
        legend_spacing = 35  
        text_offset_x = 15
        row_spacing = 20

        # Define legend title
        legend_title_font = pygame.font.Font(None, 20)
        legend_title_surface = legend_title_font.render("Legend:", True, (0, 0, 0))
        title_width, title_height = legend_title_surface.get_size()

        # Define legend fonts
        legend_font = pygame.font.Font(None, 18)
        legend_color = (0, 0, 0)  # Black

        # Define legend items
        legend_items = {
            (147, 112, 219): "Active AEP",
            (128, 0, 128): "Hyperactive AEP",
            (173, 216, 230): "Tau Protein",
            (255, 255, 128): "Alpha-syn Protein",
            (113, 166, 210): "Tau Cleaved",
            (225, 225, 100): "Alpha-syn Cleaved",
            (0, 0, 255): "Tau Oligomer",
            (255, 255, 0): "Alpha-syn Oligomer",
            (169, 169, 169): "External Input",
            (211, 211, 211): "Treatment",
            (144, 238, 144): "Resting Microglia",
            (0, 100, 0): "Active Microglia",
            (255, 105, 180): "Healthy Neuron",
            (255, 69, 0): "Damaged Neuron",
            (0, 0, 0): "Dead Neuron",
            (255, 0, 0): "Pro-inflammatory Cytokine",
            (0, 255, 255): "Anti-inflammatory Cytokine"
        }

        # Calculate number of items per column
        items_per_column = len(legend_items) // 2 + len(legend_items) % 2

        # Calculate legend background size
        legend_width = 400  
        legend_height = items_per_column * row_spacing + 70 

        # Calculate legend title position
        title_x = legend_x + (legend_width - title_width) // 2
        title_y = legend_y + 10  

        # Draw legend title
        self.screen.blit(legend_title_surface, (title_x, title_y))

        # Draw legend items
        row = 0
        for color, label in legend_items.items():
            col = 0 if row < items_per_column else 1
            circle_x = legend_x + col * (legend_width // 2) + legend_radius  
            circle_y = legend_y + 35 + (row % items_per_column) * row_spacing + legend_radius  
            pygame.draw.circle(self.screen, color, (circle_x, circle_y), legend_radius)
            legend_text_surface = legend_font.render(label, True, legend_color)
            text_width, text_height = legend_text_surface.get_size()
            text_x = legend_x + col * (legend_width // 2) + 2 * legend_radius + text_offset_x 
            text_y = circle_y - text_height // 2  
            self.screen.blit(legend_text_surface, (text_x, text_y))
            row += 1

    # Function to draw the play and stop buttons on the screen
    def draw_buttons(self):
        button_font = pygame.font.Font(None, 30)
        buttons = ["Play", "Stop"]
        button_width = 120  
        button_height = 40 
        button_spacing = 10 
        
        total_width = (button_width * len(buttons)) + (button_spacing * (len(buttons) - 1))
        start_x = (self.width - total_width) // 2
        
        for i, button_text in enumerate(buttons):
            button_x = start_x + i * (button_width + button_spacing)
            button_rect = pygame.Rect(button_x, self.height - 150, button_width, button_height)
            pygame.draw.rect(self.screen, (137, 106, 103), button_rect)
            button_surface = button_font.render(button_text, True, (0, 0, 0))
            # Center the text on the button
            button_text_rect = button_surface.get_rect(center=button_rect.center)
            self.screen.blit(button_surface, button_text_rect.topleft)

            self.button_rects.append((button_rect, button_text))

    # Function to handle mouse clicks on the buttons
    def handle_button_click(self, mouse_pos):
        for button_rect, button_text in self.button_rects:
            if button_rect.collidepoint(mouse_pos):
                self.on_button_click(button_text)

    # Function to handle button clicks
    def on_button_click(self, button_text):
        if button_text == "Play":
            self.paused = False
        elif button_text == "Stop":
            self.paused = True

@dataclass
class Log:
    # Gut variables
    aep_active: int = 0
    aep_hyperactive: int = 0
    alpha_protein_gut: int = 0
    tau_protein_gut: int = 0
    alpha_cleaved_gut: int = 0
    tau_cleaved_gut: int = 0
    alpha_oligomer_gut: int = 0
    tau_oligomer_gut: int = 0
    barrier_impermeability : int = 0
    microbiota_good_bacteria_class : int = 0
    microbiota_pathogenic_bacteria_class : int = 0
    # Brain variables
    resting_microglia: int = 0
    active_microglia: int = 0
    healthy_neuron: int = 0
    damaged_neuron: int = 0
    dead_neuron: int = 0
    cleaved_alpha_syn_brain: int = 0
    alpha_syn_oligomer_brain: int = 0
    cleaved_tau_brain: int = 0
    tau_oligomer_brain: int = 0
    cytokine_pro_inflammatory: int = 0
    cytokine_non_inflammatory: int = 0

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
    
# Manages the communication between the gut and brain contexts
class GutBrainInterface():
    def __init__(self, gut_context, brain_context):
        self.gut_context = gut_context
        self.brain_context = brain_context
        self.gut_grid = self.gut_context.get_projection("gut_grid")
        self.brain_grid = self.brain_context.get_projection("brain_grid")
        repast4py.random.seed = params['seed']
        self.rng = repast4py.random.default_rng

    # Unidirectional channel from gut to brain
    def transfer_from_gut_to_brain(self, agent):
        self.brain_context.add(agent)
        pt = self.brain_grid.get_random_local_pt(self.rng)
        self.brain_grid.move(agent, pt)
        agent.context = 'brain'
        agent.toRemove = False
        agent.toMove = False
        self.gut_context.remove(agent)

class AEP(core.Agent):
    TYPE = 0

    def __init__(self, local_id: int, rank: int, pt: dpt, context):
        super().__init__(id=local_id, type=AEP.TYPE, rank=rank)
        self.state = params["aep_state"]["active"]
        self.pt = pt
        self.context = context
        
    def save(self) -> Tuple:
        return (self.uid, self.state, self.pt.coordinates, self.context)
    
    # returns True if the agent is hyperactive, False otherwise
    def is_hyperactive(self):
        if self.state == params["aep_state"]["active"]:
            return False
        else:
            return True

    # AEP step function   
    def step(self):
        if self.pt is None:
            return
        nghs_coords = model.ngh_finder.find(self.pt.x, self.pt.y)
        protein = self.percepts(nghs_coords)
        if protein is not None:
            if(self.is_hyperactive() == True):
                self.cleave(protein)
        else: 
            random_index = np.random.randint(0, len(nghs_coords))
            model.move(self, dpt(nghs_coords[random_index][0], nghs_coords[random_index][1]), self.context)

    # returns the protein agent in the neighborhood of the agent
    def percepts(self, nghs_coords):
        for ngh_coords in nghs_coords:
            nghs_array = model.gut_grid.get_agents(dpt(ngh_coords[0], ngh_coords[1]))
            for ngh in nghs_array:
                if type(ngh) == Protein:
                    return ngh  
        return None 
    
    # cleaves the protein agent
    def cleave(self, protein):
        protein.change_state()

class Protein(core.Agent):

    TYPE = 1

    def __init__(self, local_id: int, rank: int, protein_name, pt: dpt, context):
        super().__init__(id=local_id, type=Protein.TYPE, rank=rank)
        self.name = protein_name
        self.pt = pt
        self.toCleave = False
        self.toRemove = False
        self.context =  context

    def save(self) -> Tuple: 
        return (self.uid, self.name, self.pt.coordinates, self.toCleave, self.toRemove, self.context)
    
    # Protein step function
    def step(self):
        if self.pt is None:
            return
        else: 
            nghs_coords = model.ngh_finder.find(self.pt.x, self.pt.y)
            random_index = np.random.randint(0, len(nghs_coords))
            chosen_dpt = dpt(nghs_coords[random_index][0], nghs_coords[random_index][1])
            model.move(self, chosen_dpt, self.context)

    # changes the state of the protein agent
    def change_state(self):
        if self.toCleave == False:
            self.toCleave = True

class CleavedProtein(core.Agent):
    TYPE = 2

    def __init__(self, local_id: int, rank: int, cleaved_protein_name, pt: dpt, context):
        super().__init__(id=local_id, type=CleavedProtein.TYPE, rank=rank)
        self.name = cleaved_protein_name
        self.toAggregate = False
        self.alreadyAggregate = False
        self.toRemove = False
        self.pt = pt
        self.context = context
        
    def save(self) -> Tuple: 
        return (self.uid, self.name, self.pt.coordinates, self.toAggregate, self.alreadyAggregate, self.toRemove, self.context)
    
    def step(self):
        if self.alreadyAggregate == True or self.toAggregate == True or self.pt is None:
            pass
        else:
            cleaved_nghs_number, _, nghs_coords = self.check_and_get_nghs()
            if cleaved_nghs_number == 0:
                random_index = np.random.randint(0, len(nghs_coords))
                model.move(self, dpt(nghs_coords[random_index][0], nghs_coords[random_index][1]), self.context)
            elif cleaved_nghs_number >= 4:               
                self.change_state()
            else:
                self.change_group_aggregate_status()
                
    def change_group_aggregate_status(self):
        nghs_coords = model.ngh_finder.find(self.pt.x, self.pt.y)
        for ngh_coords in nghs_coords:
            if self.context == 'brain':
                nghs_array = model.brain_grid.get_agents(dpt(ngh_coords[0], ngh_coords[1]))
            else:
                nghs_array = model.gut_grid.get_agents(dpt(ngh_coords[0], ngh_coords[1]))
            for ngh in nghs_array:
                if ngh is not None:
                    ngh.alreadyAggregate = False

    def is_valid(self):
        cont = 0
        _, nghs_cleaved, _ = self.check_and_get_nghs()
        for agent in nghs_cleaved:
            if (agent.alreadyAggregate == True):
                cont += 1
        if cont >= 4:
            return True
        else: 
            return False
    
    def change_state(self):
        if self.toAggregate == False:
            self.toAggregate = True
    
    def check_and_get_nghs(self):
        nghs_coords = model.ngh_finder.find(self.pt.x, self.pt.y)
        cont = 0
        cleavedProteins = []
        for ngh_coords in nghs_coords:
            if self.context == 'brain':
                ngh_array = model.brain_grid.get_agents(dpt(ngh_coords[0], ngh_coords[1]))
            else: 
                ngh_array = model.gut_grid.get_agents(dpt(ngh_coords[0], ngh_coords[1]))
            for ngh in ngh_array:
                if (type(ngh) == CleavedProtein and self.name == ngh.name):
                    cleavedProteins.append(ngh)
                    if ngh.toAggregate == False and ngh.alreadyAggregate == False:
                        ngh.alreadyAggregate = True
                        cont += 1
        return cont, cleavedProteins, nghs_coords

class Oligomer(core.Agent):

    TYPE = 3

    def __init__(self, local_id: int, rank: int, oligomer_name, pt: dpt, context):
        super().__init__(id=local_id, type=Oligomer.TYPE, rank=rank)
        self.name = oligomer_name
        self.pt = pt
        self.toRemove = False
        self.toMove = False
        self.context = context

    def save(self) -> Tuple: 
        return (self.uid, self.name, self.pt.coordinates, self.toRemove, self.context)
    
    # Oligomer step function
    def step(self):
        if self.pt is None:
            return
        else: 
            nghs_coords = model.ngh_finder.find(self.pt.x, self.pt.y)
            random_index = np.random.randint(0, len(nghs_coords))
            chosen_dpt = dpt(nghs_coords[random_index][0], nghs_coords[random_index][1])
            model.move(self, chosen_dpt, self.context)
            if len(nghs_coords) <= 6 and self.context == 'gut':
                if model.barrier_impermeability < params["barrier_impermeability"]:
                    percentage_threshold = int((model.barrier_impermeability * params["barrier_impermeability"])/100)
                    choice = np.random.randint(0, 100)
                    if choice > percentage_threshold:
                        self.toMove = True

class ExternalInput(core.Agent):

    TYPE = 4

    def __init__(self, local_id: int, rank: int, pt: dpt, context):
        super().__init__(id=local_id, type=ExternalInput.TYPE, rank=rank)
        possible_types = [params["external_input"]["diet"],params["external_input"]["antibiotics"],params["external_input"]["stress"]]
        random_index = np.random.randint(0, len(possible_types))
        input_name = possible_types[random_index]
        self.input_name = input_name
        self.pt = pt
        self.context = context

    def save(self) -> Tuple:
        return (self.uid, self.input_name, self.pt.coordinates, self.context)

    # External input step function
    def step(self):
        if model.barrier_impermeability >= model.barrier_permeability_threshold_stop:
            def adjust_bacteria(good_bacteria_factor, pathogenic_bacteria_factor):
                to_remove = int((model.microbiota_good_bacteria_class * np.random.uniform(0, good_bacteria_factor)) / 100)
                model.microbiota_good_bacteria_class -= to_remove
                to_add = int((params["microbiota_pathogenic_bacteria_class"] * np.random.uniform(0, pathogenic_bacteria_factor)) / 100)
                model.microbiota_pathogenic_bacteria_class += to_add

            if self.input_name == params["external_input"]["diet"]:
                adjust_bacteria(3, 3)
            elif self.input_name == params["external_input"]["antibiotics"]:
                adjust_bacteria(5, 2)
            else:
                adjust_bacteria(3, 3)

class Treatment(core.Agent):

    TYPE = 5

    def __init__(self, local_id: int, rank: int, pt: dpt, context):
        super().__init__(id=local_id, type=Treatment.TYPE, rank=rank)
        possible_types = [params["treatment_input"]["diet"],params["treatment_input"]["probiotics"]]
        random_index = np.random.randint(0, len(possible_types))
        input_name = possible_types[random_index]
        self.pt = pt
        self.input_name = input_name
        self.context = context

    def save(self) -> Tuple:
        return (self.uid, self.input_name, self.pt.coordinates, self.context)

    # Treatment step function
    def step(self):
        if model.barrier_impermeability < model.barrier_permeability_threshold_start:
            def adjust_bacteria(good_bacteria_factor, pathogenic_bacteria_factor):
                to_add = int((params["microbiota_good_bacteria_class"] * np.random.uniform(0, good_bacteria_factor)) / 100)
                model.microbiota_good_bacteria_class += to_add
                to_remove = int((model.microbiota_pathogenic_bacteria_class * np.random.uniform(0, pathogenic_bacteria_factor)) / 100)
                model.microbiota_pathogenic_bacteria_class -= to_remove

            if self.input_name == params["treatment_input"]["diet"]:
                adjust_bacteria(3, 2)
            elif self.input_name == params["treatment_input"]["probiotics"]:
                adjust_bacteria(4, 4)

class Microglia(core.Agent):

    TYPE = 6

    def __init__(self, local_id: int, rank: int, initial_state, pt: dpt, context):
        super().__init__(id=local_id, type=Microglia.TYPE, rank=rank)
        self.state = initial_state
        self.pt = pt
        self.context = context
    
    def save(self) -> Tuple:
        return (self.uid, self.state, self.pt.coordinates, self.context)

    # Microglia step function
    def step(self):
        nghs_coords = model.ngh_finder.find(self.pt.x, self.pt.y)
        ngh = self.check_oligomer_nghs(nghs_coords)
        if ngh is not None: 
            if self.state == params["microglia_state"]["resting"]:
                self.state = params["microglia_state"]["active"]
            else: 
                ngh.toRemove = True

    # returns the oligomer agent in the neighborhood of the agent     
    def check_oligomer_nghs(self, nghs_coords):
        for ngh_coord in nghs_coords:
            ngh_array = model.brain_grid.get_agents(dpt(ngh_coord[0], ngh_coord[1]))
            for ngh in ngh_array:
                if (type(ngh) == Oligomer):
                        return ngh
        return None

class Neuron(core.Agent): 

    TYPE = 7

    def __init__(self, local_id: int, rank: int, initial_state, pt: dpt, context: str):
        super().__init__(id=local_id, type=Neuron.TYPE, rank=rank)
        self.state = initial_state
        self.pt = pt
        self.toRemove = False
        self.context = context

    def save(self) -> Tuple:
        return (self.uid, self.state, self.pt.coordinates, self.toRemove, self.context)
    
    # Neuron step function
    def step(self): 
        difference_pro_anti_cytokine = model.pro_cytokine - model.anti_cytokine
        if difference_pro_anti_cytokine > 0: 
            level_of_inflammation = (difference_pro_anti_cytokine * 100)/(model.pro_cytokine + model.anti_cytokine)
            if np.random.randint(0,100) < level_of_inflammation: 
                self.change_state()
        else:
            pass
    
    # changes the state of the neuron agent
    def change_state(self):
        if self.state == params["neuron_state"]["healthy"]:
            self.state = params["neuron_state"]["damaged"]
        elif self.state == params["neuron_state"]["damaged"]:
            self.state = params["neuron_state"]["dead"]
            self.toRemove = True
            model.dead_neuron += 1

class Cytokine(core.Agent): 

    TYPE = 8

    def __init__(self, local_id: int, rank: int, pt: dpt, context):
        super().__init__(id=local_id, type=Cytokine.TYPE, rank=rank)
        self.pt = pt
        self.context = context
        possible_types = [params["cyto_state"]["pro_inflammatory"],params["cyto_state"]["non_inflammatory"]]
        random_index = np.random.randint(0, len(possible_types))
        self.state = possible_types[random_index]
        if self.state == params["cyto_state"]["pro_inflammatory"]:
            model.pro_cytokine += 1
        else: 
            model.anti_cytokine += 1

    def save(self) -> Tuple:
        return (self.uid, self.state, self.pt.coordinates, self.context)
    
    # Cytokine step function
    def step(self): 
        if self.pt is None:
            return
        microglie_nghs, nghs_coords = self.get_microglie_nghs()
        if len(microglie_nghs) == 0:
            random_index = np.random.randint(0, len(nghs_coords))
            model.move(self, dpt(nghs_coords[random_index][0], nghs_coords[random_index][1]), self.context)
        else:               
            ngh_microglia = microglie_nghs[0]
            if self.state == params["cyto_state"]["pro_inflammatory"] and ngh_microglia.state == params["microglia_state"]["resting"]:
                ngh_microglia.state = params["microglia_state"]["active"]
            elif self.state == params["cyto_state"]["non_inflammatory"] and ngh_microglia.state == params["microglia_state"]["active"]:
                ngh_microglia.state = params["microglia_state"]["resting"]

    # returns the microglia agents in the neighborhood of the agent
    def get_microglie_nghs(self):
        nghs_coords = model.ngh_finder.find(self.pt.x, self.pt.y)
        microglie = []
        for ngh_coords in nghs_coords:
            nghs_array = model.brain_grid.get_agents(dpt(ngh_coords[0], ngh_coords[1]))
            for ngh in nghs_array:
                if (type(ngh) == Microglia):
                    microglie.append(ngh)
        return microglie, nghs_coords

agent_cache = {}

# Function to restore the agents in the brain context from saved data
def restore_agent_brain(agent_data: Tuple):
    uid = agent_data[0]
    pt_array = agent_data[2]
    pt = dpt(pt_array[0], pt_array[1], 0)

    if uid[1] == Microglia.TYPE:
        agent_state = agent_data[1]
        context = agent_data[3]
        if uid in agent_cache:
            agent = agent_cache[uid]
        else:
            agent = Microglia(uid[0], uid[2], agent_state, pt, context)
            agent_cache[uid] = agent
        agent.state = agent_state
        agent.context = context
        agent.pt = pt
    elif uid[1] == Neuron.TYPE:
        context = agent_data[4]
        agent_state = agent_data[1]
        if uid in agent_cache:
            agent = agent_cache[uid]
        else:
            agent = Neuron(uid[0], uid[2], agent_state, pt, context)
            agent_cache[uid] = agent
        agent.state = agent_state
        agent.context = context
        agent.toRemove = agent_data[3]
        agent.pt = pt
    elif uid[1] == CleavedProtein.TYPE:
        agent_name = agent_data[1]
        context = agent_data[6]
        if uid in agent_cache:
            agent = agent_cache[uid]
        else:
            agent = CleavedProtein(uid[0], uid[2], agent_name, pt, context)
            agent_cache[uid] = agent
        agent.name = agent_name
        agent.toAggregate = agent_data[3]
        agent.alreadyAggregate = agent_data[4]
        agent.toRemove = agent_data[5]
        agent.context = context
        agent.pt = pt
    elif uid[1] == Oligomer.TYPE:
        context = agent_data[4]
        agent_name = agent_data[1]
        toRemove = agent_data[3]
        if uid in agent_cache:
            agent = agent_cache[uid]
        else:
            agent = Oligomer(uid[0], uid[2], agent_name, pt, context)
            agent_cache[uid] = agent
        agent.name = agent_name
        agent.pt = pt
        agent.context = context
        agent.toRemove = toRemove
    elif uid[1] == Cytokine.TYPE:
        context = agent_data[3]
        if uid in agent_cache:
            agent = agent_cache[uid]
        else:
            agent = Cytokine(uid[0], uid[2], pt, context)
            agent_cache[uid] = agent
        agent.pt = pt
        agent.context = context
        agent.state = agent_data[1]
    return agent

# Function to restore the agents in the gut context from saved data
def restore_agent_gut(agent_data: Tuple):
    uid = agent_data[0]
    pt_array = agent_data[2]
    pt = dpt(pt_array[0], pt_array[1], 0)

    if uid[1] == AEP.TYPE:
        context = agent_data[3]
        if uid in agent_cache:
            agent = agent_cache[uid]
        else:
            agent = AEP(uid[0], uid[2], pt, context)
            agent_cache[uid] = agent
        agent.state = agent_data[1]
        agent.context = context
        agent.pt = pt
    elif uid[1] == Protein.TYPE:
        context = agent_data[5]
        protein_name = agent_data[1]
        if uid in agent_cache:
            agent = agent_cache[uid]
        else:
            agent = Protein(uid[0], uid[2], protein_name, pt, context)
            agent_cache[uid] = agent
        agent.toCleave = agent_data[3]
        agent.toRemove = agent_data[4]
        agent.context = agent_data[5]
        agent.pt = pt
    elif uid[1] == CleavedProtein.TYPE:
        agent_name = agent_data[1]
        context = agent_data[6]
        if uid in agent_cache:
            agent = agent_cache[uid]
        else:
            agent = CleavedProtein(uid[0], uid[2], agent_name, pt, context)
            agent_cache[uid] = agent
        agent.toAggregate = agent_data[3]
        agent.alreadyAggregate = agent_data[4]
        agent.toRemove = agent_data[5]
        agent.context = context
        agent.pt = pt
    elif uid[1] == Oligomer.TYPE:
        context = agent_data[4]
        agent_name = agent_data[1]
        if uid in agent_cache:
            agent = agent_cache[uid]
        else:
            agent = Oligomer(uid[0], uid[2], agent_name, pt, context)
            agent_cache[uid] = agent
        agent.pt = pt
        agent.context = context
        agent.toRemove = agent_data[3]
    elif uid[1] == ExternalInput.TYPE:
        context = agent_data[3]
        if uid in agent_cache:
            agent = agent_cache[uid]
        else:
            agent = ExternalInput(uid[0], uid[2], pt, context)
            agent_cache[uid] = agent
        agent.pt = pt
        agent.context = context
    elif uid[1] == Treatment.TYPE:
        context = agent_data[3]
        if uid in agent_cache:
            agent = agent_cache[uid]
        else:
            agent = Treatment(uid[0], uid[2], pt, context)
            agent_cache[uid] = agent
        agent.pt = pt
        agent.context = context
    return agent

class Model():

    # Initialize the model
    def __init__(self, comm: MPI.Intracomm, params: Dict):
        self.comm = comm
        self.rank = comm.Get_rank()
        # Create shared contexts for the brain and the gut
        self.gut_context = ctx.SharedContext(comm)
        self.brain_context = ctx.SharedContext(comm)
        # Create shared grids for the brain and the gut
        box = space.BoundingBox(0, params['world.width'] - 1, 0, params['world.height'] - 1, 0, 0)
        self.gut_grid = self.init_grid('gut_grid', box, self.gut_context)
        self.brain_grid = self.init_grid('brain_grid', box, self.brain_context)

        self.ngh_finder = GridNghFinder(0, 0, box.xextent, box.yextent)
        self.gutBrainInterface = GutBrainInterface(self.gut_context, self.brain_context)
        # Initialize the schedule runner
        self.runner = schedule.init_schedule_runner(comm)
        self.init_schedule(params)
        # Set the seed for the random number generator
        repast4py.random.seed = params['seed']
        self.rng = repast4py.random.default_rng
        # Initialize the log
        self.counts = Log()
        loggers = logging.create_loggers(self.counts, op=MPI.SUM, rank=self.rank)
        self.data_set = logging.ReducingDataSet(loggers, self.comm, params['log_file'], buffer_size=1)
        # Initialize the model parameters
        self.init_microbiota_params(params)
        self.world_size = self.comm.Get_size()
        self.added_agents_id = 0
        self.pro_cytokine = 0
        self.anti_cytokine = 0
        self.dead_neuron = self.calculate_partitioned_count(params['neuron_dead.count'])
        # Initialize the agents
        agent_types_gut = [
            ('aep_enzyme.count', AEP, None),
            ('tau_proteins.count', Protein, params["protein_name"]["tau"]),
            ('alpha_syn_proteins.count', Protein, params["protein_name"]["alpha_syn"]),
            ('external_input.count', ExternalInput, None),
            ('treatment_input.count', Treatment, None),
            ('alpha_syn_oligomers_gut.count', Oligomer, params["protein_name"]["alpha_syn"]),
            ('tau_oligomers_gut.count', Oligomer, params["protein_name"]["tau"]),
        ]
        agent_types_brain = [
            ('neuron_healthy.count', Neuron, 'healthy'),
            ('neuron_damaged.count', Neuron, 'damaged'),
            ('neuron_dead.count', Neuron, 'dead'),
            ('resting_microglia.count', Microglia, 'resting'),
            ('active_microglia.count', Microglia, 'active'),
            ('alpha_syn_cleaved_brain.count', CleavedProtein, params["protein_name"]["alpha_syn"]),
            ('tau_cleaved_brain.count', CleavedProtein, params["protein_name"]["tau"]),
            ('alpha_syn_oligomer_brain.count', Oligomer, params["protein_name"]["alpha_syn"]),
            ('tau_oligomer_brain.count', Oligomer, params["protein_name"]["tau"]),
            ('cytokine.count', Cytokine, None)
        ]
        self.distribute_all_agents(agent_types_gut, self.gut_context, self.gut_grid, 'gut')
        self.distribute_all_agents(agent_types_brain, self.brain_context, self.brain_grid, 'brain')
        # Synchronize the contexts
        self.gut_context.synchronize(restore_agent_gut)
        self.brain_context.synchronize(restore_agent_brain)
        # Initialize Pygame and gui object
        pygame.init()
        self.screen = GUI(width=1600, height=800, gut_context=self.gut_context, brain_context=self.brain_context, grid_dimensions=(params['world.width'], params['world.height']))
        pygame.display.set_caption("Gut-Brain Axis Model")
        self.screen.update(gut_context=self.gut_context, brain_context=self.brain_context)

    # Function to initialize the shared grid
    def init_grid(self, name, box, context):
        grid = space.SharedGrid(name=name, bounds=box, borders=space.BorderType.Sticky, occupancy=space.OccupancyType.Multiple, buffer_size=1, comm=self.comm)
        context.add_projection(grid)
        return grid
    
    # Function to initialize the schedule
    def init_schedule(self, params):
        self.runner.schedule_repeating_event(1, 1, self.gut_step)
        self.runner.schedule_repeating_event(1, 2, self.microbiota_dysbiosis_step)
        self.runner.schedule_repeating_event(1, 5, self.move_cleaved_protein_step)
        self.runner.schedule_repeating_event(1, 1, self.brain_step, priority_type=0)
        self.runner.schedule_repeating_event(1, 1, self.pygame_update, priority_type=1)
        self.runner.schedule_repeating_event(1, 1, self.log_counts, priority_type=1)
        self.runner.schedule_stop(params['stop.at'])
        self.runner.schedule_end_event(self.at_end)

    # Function to initialize the microbiota parameters
    def init_microbiota_params(self, params):
        self.microbiota_good_bacteria_class = params["microbiota_good_bacteria_class"]
        self.microbiota_pathogenic_bacteria_class = params["microbiota_pathogenic_bacteria_class"]
        self.microbiota_diversity_threshold = params["microbiota_diversity_threshold"]
        self.barrier_impermeability = params["barrier_impermeability"]
        self.barrier_permeability_threshold_stop = params["barrier_permeability_threshold_stop"]
        self.barrier_permeability_threshold_start = params["barrier_permeability_threshold_start"]

    # Function to distribute all agents through the different ranks
    def distribute_all_agents(self, agent_types, context, grid, region):
        for agent_type in agent_types:
            total_count = params[agent_type[0]]
            pp_count = self.calculate_partitioned_count(total_count)
            self.create_agents(agent_type[1], pp_count, agent_type[2], context, grid, region)

    # Function to create agents in the different ranks based on the total count
    def create_agents(self, agent_class, pp_count, state_key, context, grid, region):
        for j in range(pp_count):
            pt = grid.get_random_local_pt(self.rng)
            if agent_class in [Neuron, Microglia]:
                agent = agent_class(self.added_agents_id + j, self.rank, params[f"{agent_class.__name__.lower()}_state"][state_key], pt, region)
            elif agent_class in [CleavedProtein, Oligomer, Protein]:
                agent = agent_class(self.added_agents_id + j, self.rank, params["protein_name"][state_key], pt, region)
            else:  
                # For agents without special state keys
                agent = agent_class(self.added_agents_id + j, self.rank, pt, region)
            context.add(agent)
            self.move(agent, pt, agent.context)
        self.added_agents_id += pp_count

    # Function to get the total count of agents to create in that rank
    def calculate_partitioned_count(self, total_count):
        pp_count = int(total_count / self.world_size)
        if self.rank < total_count % self.world_size:
            pp_count += 1
        return pp_count

    # Function to update the interface
    def pygame_update(self):   
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # If the 'X' button is clicked, stop the simulation
                print("Ending the simulation.")
                self.at_end()
                self.comm.Abort()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.screen.handle_button_click(event.pos)

        while self.screen.paused:
            for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        # If the 'X' button is clicked, stop the simulation
                        print("Ending the simulation.")
                        self.at_end()
                        self.comm.Abort()
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        self.screen.handle_button_click(event.pos)
        
        # Updates the Pygame GUI based on the current state of the Repast simulation
        self.screen.update(gut_context=self.gut_context, brain_context=self.brain_context)
        pygame.display.flip()        

    # Brain steps    
    def brain_step(self):
        self.brain_context.synchronize(restore_agent_brain)

        def gather_agents_to_remove():
            return [agent for agent in self.brain_context.agents() if isinstance(agent, (Oligomer, CleavedProtein, Neuron)) and agent.toRemove]

        # Remove agents marked for removal
        remove_agents = gather_agents_to_remove()
        removed_ids = set()
        for agent in remove_agents:
            if self.brain_context.agent(agent.uid) is not None:
                self.remove_agent(agent)
                removed_ids.add(agent.uid)

        self.brain_context.synchronize(restore_agent_brain)

        # Let each agent perform its step
        for agent in self.brain_context.agents():
            agent.step()

        # Collect data and perform operations based on agent states
        oligomer_to_remove = []
        active_microglia = 0
        damaged_neuron = 0
        all_true_cleaved_aggregates = []

        for agent in self.brain_context.agents():
            if isinstance(agent, Oligomer) and agent.toRemove:
                oligomer_to_remove.append(agent)
            elif isinstance(agent, Microglia) and agent.state == params["microglia_state"]["active"]:
                active_microglia += 1
            elif isinstance(agent, Neuron) and agent.state == params["neuron_state"]["damaged"]:
                damaged_neuron += 1
            elif isinstance(agent, CleavedProtein) and agent.toAggregate:
                all_true_cleaved_aggregates.append(agent)
                agent.toRemove = True

        for _ in range(active_microglia):
            self.add_cytokine()
        for _ in range(damaged_neuron):
            self.brain_add_cleaved_protein()
        for oligomer in oligomer_to_remove:
            if self.brain_context.agent(oligomer.uid) is not None:
                self.remove_agent(oligomer)
                removed_ids.add(oligomer.uid)

        self.brain_context.synchronize(restore_agent_brain)

        for agent in all_true_cleaved_aggregates:
            if agent.uid in removed_ids:
                continue
            if agent.toAggregate and agent.is_valid():
                cont = 0
                _, agent_nghs_cleaved, _ = agent.check_and_get_nghs()
                for x in agent_nghs_cleaved:
                    if x.alreadyAggregate and x.uid != agent.uid:
                        if cont < 3:
                            if self.brain_context.agent(x.uid) is not None:
                                self.remove_agent(x)
                                removed_ids.add(x.uid)
                            cont += 1
                        else:
                            x.alreadyAggregate = False
                            x.toAggregate = False
                            cont += 1
                self.add_oligomer_protein(agent.name, agent.context)
                self.remove_agent(agent)
                removed_ids.add(agent.uid)

        self.brain_context.synchronize(restore_agent_brain)

        # Remove agents marked for removal after all processing
        remove_agents = gather_agents_to_remove()
        for agent in remove_agents:
            if agent.uid not in removed_ids:
                if self.brain_context.agent(agent.uid) is not None:
                    self.remove_agent(agent)
                    removed_ids.add(agent.uid)

    # Function to remove an agent from the context and the grid
    def remove_agent(self, agent):
        if agent.context == 'brain':
            self.brain_context.remove(agent)
        else:
            self.gut_context.remove(agent)   

    # Function to add a cleaved protein agent to the brain context
    def brain_add_cleaved_protein(self):
        self.added_agents_id += 1
        possible_types = [params["protein_name"]["alpha_syn"], params["protein_name"]["tau"]]
        random_index = np.random.randint(0, len(possible_types))
        cleaved_protein_name = possible_types[random_index]
        pt = self.brain_grid.get_random_local_pt(self.rng)
        cleaved_protein = CleavedProtein(self.added_agents_id, self.rank, cleaved_protein_name, pt, 'brain')   
        self.brain_context.add(cleaved_protein)   
        self.move(cleaved_protein, cleaved_protein.pt, cleaved_protein.context)
    
    # Function to add a cleaved protein agent to the gut context
    def gut_add_cleaved_protein(self,cleaved_protein_name):
        self.added_agents_id += 1
        pt = self.gut_grid.get_random_local_pt(self.rng)
        cleaved_protein = CleavedProtein(self.added_agents_id, self.rank, cleaved_protein_name, pt, 'gut')   
        self.gut_context.add(cleaved_protein)   
        self.move(cleaved_protein, cleaved_protein.pt, 'gut')

    # Function to add an oligomer protein agent to the brain or gut context
    def add_oligomer_protein(self, oligomer_name, context):
        self.added_agents_id += 1 
        if context == 'brain':
            pt = self.brain_grid.get_random_local_pt(self.rng) 
            oligomer_protein = Oligomer(self.added_agents_id, self.rank, oligomer_name, pt, 'brain')
            self.brain_context.add(oligomer_protein)        
            self.move(oligomer_protein, oligomer_protein.pt, 'brain') 
        else:
            pt = self.gut_grid.get_random_local_pt(self.rng) 
            oligomer_protein = Oligomer(self.added_agents_id, self.rank, oligomer_name, pt, 'gut')
            self.gut_context.add(oligomer_protein)        
            self.move(oligomer_protein, oligomer_protein.pt, 'gut') 
             
    # Function to move an agent to a new location
    def move(self, agent, pt: dpt, context):
        if context == 'brain':
            self.brain_grid.move(agent, pt)
        else:
            self.gut_grid.move(agent, pt)
        agent.pt = pt

    # Function to add a cytokine agent to the brain context
    def add_cytokine(self):
        self.added_agents_id += 1
        pt = self.brain_grid.get_random_local_pt(self.rng)
        cytokine= Cytokine(self.added_agents_id, self.rank, pt, 'brain')   
        self.brain_context.add(cytokine)   
        self.move(cytokine, cytokine.pt, 'brain')
    
    # Gut steps 
    # Function to move the cleaved protein agents 
    def move_cleaved_protein_step(self):
        for agent in self.gut_context.agents():
            if type(agent) == CleavedProtein:
                if agent.alreadyAggregate == False:
                    pt = self.gut_grid.get_random_local_pt(self.rng)
                    self.move(agent, pt, agent.context)
        for agent in self.brain_context.agents():
            if type(agent) == CleavedProtein:
                if agent.alreadyAggregate == False:
                    pt = self.brain_grid.get_random_local_pt(self.rng)
                    self.move(agent, pt, agent.context)

    # Function to check if the microbiota is dysbiotic and adjust the barrier impermeability 
    def microbiota_dysbiosis_step(self):
        if self.microbiota_good_bacteria_class - self.microbiota_pathogenic_bacteria_class <= self.microbiota_diversity_threshold:
            value_decreased = int((params["barrier_impermeability"]*np.random.randint(0,6))/100) 
            if self.barrier_impermeability - value_decreased <= 0:
                self.barrier_impermeability = 0
            else:
                self.barrier_impermeability = self.barrier_impermeability - value_decreased
            number_of_aep_to_hyperactivate = value_decreased
            cont = 0
            for agent in self.gut_context.agents(agent_type=0):
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

    def gut_step(self):
        self.gut_context.synchronize(restore_agent_gut)

        def gather_agents_to_remove():
            return [agent for agent in self.gut_context.agents() if isinstance(agent, (Oligomer, CleavedProtein, Protein)) and agent.toRemove]

        remove_agents = gather_agents_to_remove()
        removed_ids = set()
        for agent in remove_agents:
            if self.gut_context.agent(agent.uid) is not None:
                self.remove_agent(agent)
                removed_ids.add(agent.uid)

        self.gut_context.synchronize(restore_agent_gut)

        for agent in self.gut_context.agents():
            agent.step()

        protein_to_remove = []
        all_true_cleaved_aggregates = []
        oligomers_to_move = []
        
        for agent in self.gut_context.agents():
            if(type(agent) == Protein and agent.toCleave == True):
                protein_to_remove.append(agent)
                agent.toRemove = True
            elif(type(agent) == CleavedProtein and agent.toAggregate == True):
                all_true_cleaved_aggregates.append(agent)  
                agent.toRemove = True 
            elif(type(agent) == Oligomer and agent.toMove == True):
                oligomers_to_move.append(agent)
                agent.toRemove = True

        for agent in oligomers_to_move:
            self.gutBrainInterface.transfer_from_gut_to_brain(agent)

        for agent in protein_to_remove:
            if agent.uid in removed_ids:
                continue
            protein_name = agent.name
            self.remove_agent(agent)
            removed_ids.add(agent.uid)                
            self.gut_add_cleaved_protein(protein_name)
            self.gut_add_cleaved_protein(protein_name)

        self.gut_context.synchronize(restore_agent_gut)

        for agent in all_true_cleaved_aggregates:
            if agent.uid in removed_ids:
                continue
            if agent.toAggregate and agent.is_valid():
                cont = 0
                _, agent_nghs_cleaved, _ = agent.check_and_get_nghs()
                for x in agent_nghs_cleaved:
                    if x.alreadyAggregate and x.uid != agent.uid:
                        if cont < 3:
                            if self.gut_context.agent(x.uid) is not None:
                                self.remove_agent(x)
                                removed_ids.add(x.uid)
                            cont += 1
                        else:
                            x.alreadyAggregate = False
                            x.toAggregate = False
                            cont += 1
                self.add_oligomer_protein(agent.name, 'gut')
                self.remove_agent(agent)
                removed_ids.add(agent.uid)

        self.gut_context.synchronize(restore_agent_gut)  

        remove_agents = gather_agents_to_remove()
        for agent in remove_agents:
            if agent.uid not in removed_ids:
                if self.gut_context.agent(agent.uid) is not None:
                    self.remove_agent(agent)
                    removed_ids.add(agent.uid) 

    # Function to log the counts of the agents
    def log_counts(self):
        tick = self.runner.schedule.tick

        counts = {
            "aep_active" : 0,
            "aep_hyperactive" : 0,
            "alpha_protein_gut" : 0,
            "tau_protein_gut" : 0,
            "alpha_cleaved_gut": 0,
            "tau_cleaved_gut": 0,
            "alpha_oligomer_gut": 0,
            "tau_oligomer_gut": 0,
            "microglia_resting": 0,
            "microglia_active": 0,
            "neuron_healthy": 0,
            "neuron_damaged": 0,
            "alpha_cleaved_brain": 0,
            "tau_cleaved_brain": 0,
            "alpha_oligomer_brain": 0,
            "tau_oligomer_brain": 0
        }

        for agent in self.brain_context.agents():
            if isinstance(agent, Oligomer):
                if agent.name == params["protein_name"]["alpha_syn"]:
                    counts["alpha_oligomer_brain"] += 1
                else:
                    counts["tau_oligomer_brain"] += 1
            elif isinstance(agent, CleavedProtein):
                if agent.name == params["protein_name"]["alpha_syn"]:
                    counts["alpha_cleaved_brain"] += 1
                else:
                    counts["tau_cleaved_brain"] += 1
            elif isinstance(agent, Neuron):
                if agent.state == params["neuron_state"]["healthy"]:
                    counts["neuron_healthy"] += 1
                elif agent.state == params["neuron_state"]["damaged"]:
                    counts["neuron_damaged"] += 1
            elif isinstance(agent, Microglia):
                if agent.state == params["microglia_state"]["active"]:
                    counts["microglia_active"] += 1
                else:
                    counts["microglia_resting"] += 1

        for agent in self.gut_context.agents(): 
            if(type(agent) == Oligomer): 
                if (agent.name == params["protein_name"]["alpha_syn"]): 
                    counts["alpha_oligomer_gut"] += 1 
                else: 
                    counts["tau_oligomer_gut"] += 1 
            if(type(agent) == CleavedProtein): 
                if (agent.name == params["protein_name"]["alpha_syn"]): 
                    counts["alpha_cleaved_gut"] += 1 
                else: 
                    counts["tau_cleaved_gut"] += 1 
            elif(type(agent) == Protein): 
                if (agent.name == params["protein_name"]["alpha_syn"]): 
                    counts["alpha_protein_gut"] += 1 
                else: 
                    counts["tau_protein_gut"] += 1 
            elif(type(agent) == AEP): 
                if(agent.state == params["aep_state"]["active"]): 
                    counts["aep_active"] += 1 
                else:  
                    counts["aep_hyperactive"] += 1            

        #brain
        self.counts.healthy_neuron = counts["neuron_healthy"]
        self.counts.damaged_neuron = counts["neuron_damaged"]
        self.counts.dead_neuron = self.dead_neuron
        self.counts.cytokine_pro_inflammatory = self.pro_cytokine
        self.counts.cytokine_non_inflammatory = self.anti_cytokine
        self.counts.cleaved_alpha_syn_brain = counts["alpha_cleaved_brain"]
        self.counts.cleaved_tau_brain = counts["tau_cleaved_brain"]
        self.counts.alpha_syn_oligomer_brain = counts["alpha_oligomer_brain"]
        self.counts.tau_oligomer_brain = counts["tau_oligomer_brain"]
        self.counts.resting_microglia = counts["microglia_resting"]
        self.counts.active_microglia = counts["microglia_active"]

        #gut
        self.counts.aep_active = counts["aep_active"]
        self.counts.aep_hyperactive = counts["aep_hyperactive"]
        self.counts.alpha_protein_gut = counts["alpha_protein_gut"]
        self.counts.tau_protein_gut = counts["tau_protein_gut"]
        self.counts.alpha_cleaved_gut = counts["alpha_cleaved_gut"]
        self.counts.tau_cleaved_gut = counts["tau_cleaved_gut"]
        self.counts.alpha_oligomer_gut = counts["alpha_oligomer_gut"]
        self.counts.tau_oligomer_gut = counts["tau_oligomer_gut"]
        self.counts.microbiota_good_bacteria_class = self.microbiota_good_bacteria_class
        self.counts.microbiota_pathogenic_bacteria_class = self.microbiota_pathogenic_bacteria_class
        self.counts.barrier_impermeability = self.barrier_impermeability

        self.data_set.log(tick)

    # Function to close the data set and quit Pygame
    def at_end(self):
        self.data_set.close()
        pygame.quit()

    # Function to start the simulation
    def start(self):
        self.runner.execute()

# Function to run the simulation
def run(params: Dict):
    global model      
    model = Model(MPI.COMM_WORLD, params)
    model.start()

if __name__ == "__main__":
    parser = parameters.create_args_parser()
    args = parser.parse_args()
    params = parameters.init_params(args.parameters_file, args.parameters)
    run(params)
    