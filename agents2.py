import agentpy as ap
import random
import pickle
import os

# Trash Container Agent
class TrashContainerAgent(ap.Agent):
    def setup(self):
        self.position = None
        self.capacity = self.p.container_limit
        self.current_fill = 0
        self.last_emptied_by = None  # rastrea qué camión lo vació
        self.last_emptied_step = -1  # rastrea cuándo fue vaciado

    def step(self):
        if random.uniform(0, 1) < self.p.population_density:
            if self.p.population_density >= 0.3:
                basura_generada = random.randint(2, 5)
            else:
                basura_generada = random.randint(1, 3)
            self.current_fill = min(self.current_fill + basura_generada, self.capacity * 2)

    def collect_trash(self, amount, truck_id=None):
        # Si no se especifica un camión, no permitir recolección
        if truck_id is None:
            return 0
            
        # Verificar si hay un camión en la misma posición
        truck_at_position = False
        for truck in self.model.trucks:
            if truck.truck_id == truck_id and truck.position == self.position:
                truck_at_position = True
                break
                
        if not truck_at_position:
            return 0  # No permitir recolección si no hay un camión en la posición
        
        # Proceder con la recolección normal
        collected = min(self.current_fill, amount)
        self.current_fill -= collected
        self.last_emptied_by = truck_id
        self.last_emptied_step = self.model.t
        return collected

    def is_critical(self):
        return self.current_fill >= 0.9 * self.capacity

    def is_overflowing(self):
        return self.current_fill >= self.capacity

# Trash Truck Agent
class TrashTruckAgent(ap.Agent):
    def setup(self):
        self.capacity = self.p.capacity
        self.load = 0
        self.position = (0, 0)
        self.q_table = {}
        self.epsilon = self.p.epsilon
        self.alpha = self.p.alpha
        self.gamma = self.p.gamma
        self.truck_id = 0
        self.load_q_table()

    def load_q_table(self):
        filename = f"q_table_truck_{self.truck_id}.pkl"
        if os.path.exists(filename):
            try:
                with open(filename, 'rb') as f:
                    saved_data = pickle.load(f)
                    self.q_table = saved_data['q_table']
                    self.epsilon = max(0.2, saved_data.get('epsilon', self.epsilon) * 0.98)
            except Exception as e:
                print(f"Error cargando Q-table para camión {self.truck_id}: {e}")

    def save_q_table(self):
        filename = f"q_table_truck_{self.truck_id}.pkl"
        try:
            training_runs = 1
            if os.path.exists(filename):
                with open(filename, 'rb') as f:
                    old_data = pickle.load(f)
                    training_runs = old_data.get('training_runs', 0) + 1
            with open(filename, 'wb') as f:
                pickle.dump({
                    'q_table': self.q_table,
                    'epsilon': self.epsilon,
                    'training_runs': training_runs
                }, f)
        except Exception as e:
            print(f"Error guardando Q-table para camion {self.truck_id}: {e}")

    def state(self):
        return (self.position, self.load)

    def possible_actions(self):
        return ["up", "down", "left", "right", "collect", "change_route"]

    def choose_action(self, state):
        # 1. Verificar si hay contenedores críticos y estamos en camino a uno
        critical_containers = self.model.get_critical_containers()
        current_target = self.model.get_target_for_truck(self.truck_id)
        
        # Si nuestro objetivo actual no es crítico pero hay contenedores críticos, cambiar de ruta
        if critical_containers and current_target not in critical_containers:
            return "change_route"  
        
        # 2. Si estamos en un contenedor, recoger basura
        container_at_position = self.model.get_container_at_position(self.position)
        if (container_at_position and 
            container_at_position.current_fill > 0 and 
            self.load < self.capacity):
            return "collect"

        # 3. Si estamos casi llenos, ir al vertedero
        if self.load >= self.capacity * 0.8:
            return self.move_to_dump()

        # 4. Si tenemos un objetivo, movernos hacia él
        if current_target:
            tx, ty = current_target
            x, y = self.position
            if x < tx: return "right"
            elif x > tx: return "left"
            elif y < ty: return "up"
            elif y > ty: return "down"
            else: 
                # Ya estamos en la posición objetivo, verificar si hay contenedor
                container_here = self.model.get_container_at_position(self.position)
                if container_here and container_here.current_fill > 0:
                    return "collect"
                else:
                    return "change_route"  
        
        # 5. Si no hay objetivo, explorar aleatoriamente
        return random.choice(["up", "down", "left", "right"])


    def move_to_dump(self):
        x, y = self.position
        dump_points = self.model.dump_points
        closest_dump = min(dump_points, key=lambda p: abs(x - p[0]) + abs(y - p[1]))
        tx, ty = closest_dump
        if x < tx: return "right"
        elif x > tx: return "left"
        elif y < ty: return "up"
        elif y > ty: return "down"
        else:
            self.load = 0
            return "collect"

    def update_q(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = {a: 0 for a in self.possible_actions()}
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0 for a in self.possible_actions()}
        old_value = self.q_table[state][action]
        next_max = max(self.q_table[next_state].values())
        self.q_table[state][action] = old_value + self.alpha * (reward + self.gamma * next_max - old_value)

    def step(self):
        state = self.state()
        action = self.choose_action(state)
        reward, next_state = self.execute(action)
        self.update_q(state, action, reward, next_state)

    def execute(self, action):
        x, y = self.position
        next_pos = self.position
        reward = 0

        if action == "up" and y < 7:
            next_pos = (x, y + 1)
        elif action == "down" and y > 0:
            next_pos = (x, y - 1)
        elif action == "left" and x > 0:
            next_pos = (x - 1, y)
        elif action == "right" and x < 7:
            next_pos = (x + 1, y)
        elif action == "change_route":
            # Liberar la asignación actual para que se busque un nuevo objetivo
            if self.truck_id in self.model.assignments:
                self.model.assignments.pop(self.truck_id)
                
            possible_moves = []
            if x > 0: possible_moves.append((x-1, y))
            if x < 7: possible_moves.append((x+1, y))
            if y > 0: possible_moves.append((x, y-1))
            if y < 7: possible_moves.append((x, y+1))
            
            if possible_moves:
                next_pos = random.choice(possible_moves)

            # Penalización por tener que cambiar de ruta
            reward -= 5  

        if action == "collect":
            container_at_position = self.model.get_container_at_position(self.position)
            if container_at_position and self.load < self.capacity:
                if container_at_position.current_fill > 0:
                    truck_space = self.capacity - self.load
                    amount_to_collect = min(container_at_position.current_fill, truck_space, 10)
                    collected = container_at_position.collect_trash(amount_to_collect, self.truck_id)
                    # Recompensas por recoger basura
                    reward += 30 * collected
                    if container_at_position.is_critical():
                    # Recompensa adicional por recoger de un contenedor crítico
                        reward += 100 * collected
                    self.load += collected
                else:
                    # Penalización por intentar recoger de un contenedor vacío
                    reward -= 2

                # liberar asignación si ya está vacío
                if container_at_position.current_fill <= 0:
                    self.model.assignments.pop(self.truck_id, None)
            else:
                # Penalización por intentar recoger con camión lleno
                reward -= 2

        overflowing_containers = self.model.get_overflowing_containers()
        # Penalización por contenedores desbordados
        reward -= 30 * len(overflowing_containers)

        self.position = next_pos
        return reward, self.state()
class GarbageEnvironment(ap.Model):
    def setup(self):
        import itertools
        grid_size = self.p.get('grid_size', 8)
        self.grid = ap.Grid(self, (grid_size, grid_size), track_empty=True)
        self.dump_points = [(0, 0), (grid_size-1, 0), (0, grid_size-1), (grid_size-1, grid_size-1)]

        num_containers = self.p.get('num_containers', 8)
        num_trucks = self.p.get('num_trucks', 2)

        # Generar todas las posiciones posibles, quitando los dumps
        all_positions = set(itertools.product(range(grid_size), range(grid_size)))
        forbidden = set(self.dump_points)
        available_positions = list(all_positions - forbidden)
        random.shuffle(available_positions)

        # Asignar posiciones aleatorias a los contenedores
        container_positions = available_positions[:num_containers]
        self.containers = ap.AgentList(self, num_containers, TrashContainerAgent)
        for c, pos in zip(self.containers, container_positions):
            c.position = pos
            c.current_fill = random.randint(5, 20)

        # Quitar posiciones ya usadas por contenedores
        remaining_positions = available_positions[num_containers:]
        random.shuffle(remaining_positions)
        truck_positions = remaining_positions[:num_trucks]
        self.trucks = ap.AgentList(self, num_trucks, TrashTruckAgent)
        for i, (t, pos) in enumerate(zip(self.trucks, truck_positions)):
            t.position = pos
            t.truck_id = i

        self.assignments = {}

    def step(self):
        self.assign_containers_to_trucks()
        self.containers.step()
        self.trucks.step()

    def assign_containers_to_trucks(self):
        # Identificar contenedores críticos
        criticals = [c for c in self.containers if c.is_critical()]

        # Si hay contenedores críticos, mantener asignaciones a ellos
        if criticals:
            critical_positions = {c.position for c in criticals}
            for truck_id, target_pos in list(self.assignments.items()):
                if target_pos not in critical_positions:
                    self.assignments.pop(truck_id)
        
        # Determinar qué contenedores considerar para asignación
        candidates = criticals if criticals else list(self.containers)
        used = set()

        # Ordenar camiones por carga 
        sorted_trucks = sorted(self.trucks, key=lambda t: t.load)
        
        for truck in sorted_trucks:
            # Si ya tiene un objetivo asignado a un contenedor crítico, mantenerlo
            if truck.truck_id in self.assignments:
                tgt = self.assignments[truck.truck_id]
                c = next((x for x in self.containers if x.position == tgt), None)
                
                # Solo mantener asignaciones a contenedores críticos no vacíos
                if c and c.is_critical() and c.current_fill > 5:
                    if truck.position == tgt and truck.load < truck.capacity * 0.9:
                        continue
                else:
                    # Liberar asignación si no es un contenedor crítico o está casi vacío
                    self.assignments.pop(truck.truck_id, None)
            
            # Buscar el contenedor más cercano para este camión
            best = None
            best_dist = 999
            
            for c in candidates:
                # Ignorar contenedores ya asignados o casi vacíos
                if c.position in used or c.current_fill <= 5:
                    continue
                    
                # Calcular distancia Manhattan
                dist = abs(truck.position[0] - c.position[0]) + abs(truck.position[1] - c.position[1])
                
                # Favorecer contenedores críticos reduciendo su distancia
                if c.is_critical():
                    dist *= 0.5  # Hacer que los contenedores críticos parezcan más cercanos
                    
                # Considerar la carga del camión (priorizar contenedores grandes para camiones vacíos)
                if truck.load < truck.capacity * 0.5:
                    # Camión vacío: priorizar contenedores más llenos
                    fullness_factor = c.current_fill / c.capacity
                    dist *= (1.0 - 0.3 * fullness_factor)  # Reducir distancia hasta un 30% para contenedores llenos
                    
                if dist < best_dist:
                    best = c
                    best_dist = dist
                    
            if best:
                self.assignments[truck.truck_id] = best.position
                used.add(best.position)
            # Si no hay contenedores disponibles, asignar un punto aleatorio para explorar
            elif truck.truck_id not in self.assignments:
                available = [(x, y) for x in range(self.p.grid_size) for y in range(self.p.grid_size) 
                        if (x, y) not in used and (x, y) != truck.position]
                if available:
                    explore_point = random.choice(available)
                    self.assignments[truck.truck_id] = explore_point
    def get_target_for_truck(self, truck_id):
        return self.assignments.get(truck_id, None)

    def get_container_at_position(self, position):
        for container in self.containers:
            if container.position == position:
                return container
        return None

    def get_critical_containers(self):
        return [c.position for c in self.containers if c.is_critical()]

    def get_overflowing_containers(self):
        return [c.position for c in self.containers if c.is_overflowing()]

    def get_dump_points(self):
        return list(self.dump_points)
        
    def is_truck_near_container(self, container_pos, threshold=1.0):
        """Verifica si hay algún camión cerca del contenedor"""
        for truck in self.trucks:
            dx = abs(truck.position[0] - container_pos[0])
            dy = abs(truck.position[1] - container_pos[1])
            manhattan_distance = dx + dy
            if manhattan_distance <= threshold:
                return True, truck.truck_id
        return False, None

    def end(self):
        for truck in self.trucks:
            truck.save_q_table()

# Parametros
parameters = {
    'steps': 100,
    'capacity': 500,
    'epsilon': 0.15, # Tasa de exploración
    'alpha': 0.15, # Tasa de aprendizaje
    'gamma': 0.95, # Importancia de las recompensas futuras
    'container_limit': 40,
    'population_density': 0.15,
    'num_trucks': 2,
    'num_containers': 12,
    'grid_size': 8
}

# Funciones de estadistica 
def simple_status(model):
    critical = sum(1 for c in model.containers if c.is_critical())
    overflow = sum(1 for c in model.containers if c.is_overflowing())
    total_trash = sum(c.current_fill for c in model.containers)
    total_load = sum(t.load for t in model.trucks)

    print(f"Críticos: {critical}, Desbordados: {overflow}, "
          f"Basura: {total_trash}, Carga total: {total_load}")


def show_truck_stats(model):
    """Muestra estadísticas separadas por camión"""
    print("\n--- Estadísticas por Camión ---")
    for i, truck in enumerate(model.trucks):
        efficiency = (truck.load / truck.capacity) * 100
        print(f"Camión {i}: Basura={truck.load}/{truck.capacity} ({efficiency:.1f}%) "
              f"| Estados Q={len(truck.q_table)} | Posición={truck.position}")

    total_collected = sum(t.load for t in model.trucks)
    total_remaining = sum(c.current_fill for c in model.containers)
    overall_efficiency = (total_collected / max(1, total_collected + total_remaining)) * 100
    print(f"\n📊 Eficiencia global: {overall_efficiency:.1f}%")

# Main
if __name__ == "__main__":
    model = GarbageEnvironment(parameters)
    print(f"Config: {parameters['steps']} pasos, {parameters['num_trucks']} camiones, {parameters['num_containers']} contenedores, grid {parameters['grid_size']}x{parameters['grid_size']}")

    results = model.run()
    simple_status(model)
    show_truck_stats(model)


