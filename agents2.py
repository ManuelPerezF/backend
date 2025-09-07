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

    def step(self):
        if random.uniform(0, 1) < self.p.population_density:
            if self.p.population_density >= 0.3:
                basura_generada = random.randint(2, 5)
            else:
                basura_generada = random.randint(1, 3)
            self.current_fill = min(self.current_fill + basura_generada, self.capacity * 2)

    def collect_trash(self, amount):
        collected = min(self.current_fill, amount)
        self.current_fill -= collected
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
                print(f"⚠️ Error cargando Q-table para camión {self.truck_id}: {e}")

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
            print(f"Error guardando Q-table para camión {self.truck_id}: {e}")

    def state(self):
        return (self.position, self.load)

    def possible_actions(self):
        return ["up", "down", "left", "right", "collect", "change_route"]

    def choose_action(self, state):
        container_at_position = self.model.get_container_at_position(self.position)
        if (container_at_position and 
            container_at_position.current_fill > 0 and 
            self.load < self.capacity):
            return "collect"

        if self.load >= self.capacity * 0.8:
            return self.move_to_dump()

        target = self.model.get_target_for_truck(self.truck_id)
        if target:
            tx, ty = target
            x, y = self.position
            if x < tx: return "right"
            elif x > tx: return "left"
            elif y < ty: return "up"
            elif y > ty: return "down"
            else: return "collect"

        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.possible_actions())
        else:
            return max(
                self.q_table.get(state, {}),
                key=self.q_table.get(state, {}).get,
                default=random.choice(self.possible_actions())
            )

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

        if action == "collect":
            container_at_position = self.model.get_container_at_position(self.position)
            if container_at_position and self.load < self.capacity:
                if container_at_position.current_fill > 0:
                    truck_space = self.capacity - self.load
                    amount_to_collect = min(container_at_position.current_fill, truck_space, 10)
                    collected = container_at_position.collect_trash(amount_to_collect)
                    reward += 30 * collected
                    if container_at_position.is_critical():
                        reward += 100 * collected
                    self.load += collected
                else:
                    reward -= 2

                # liberar asignación si ya está vacío
                if container_at_position.current_fill <= 0:
                    self.model.assignments.pop(self.truck_id, None)
            else:
                reward -= 2

        overflowing_containers = self.model.get_overflowing_containers()
        reward -= 30 * len(overflowing_containers)

        self.position = next_pos
        return reward, self.state()



# Garbage Environment
class GarbageEnvironment(ap.Model):
    def setup(self):
        import itertools
        grid_size = self.p.get('grid_size', 8)
        self.grid = ap.Grid(self, (grid_size, grid_size), track_empty=True)
        self.dump_points = [(0, 0), (grid_size-1, 0), (0, grid_size-1), (grid_size-1, grid_size-1)]

        num_containers = self.p.get('num_containers', 8)
        num_trucks = self.p.get('num_trucks', 2)

        # Generar todas las posiciones posibles, quitando los vertederos (dumps)
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
        criticals = [c for c in self.containers if c.is_critical()]
        candidates = criticals if criticals else list(self.containers)
        used = set()

        for truck in self.trucks:
            if truck.truck_id in self.assignments:
                tgt = self.assignments[truck.truck_id]
                c = next((x for x in self.containers if x.position == tgt), None)

                #  liberar si vacío o casi vacío
                if not c or c.current_fill <= 5:
                    self.assignments.pop(truck.truck_id, None)
                else:
                    continue

            best = None
            best_dist = 999
            for c in candidates:
                if c.position in used:
                    continue
                dist = abs(truck.position[0] - c.position[0]) + abs(truck.position[1] - c.position[1])
                if dist < best_dist:
                    best = c
                    best_dist = dist
            if best:
                self.assignments[truck.truck_id] = best.position
                used.add(best.position)

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

    def end(self):
        for truck in self.trucks:
            truck.save_q_table()




# Parametros
parameters = {
    'steps': 1000,
    'capacity': 200,
    'epsilon': 0.3,
    'alpha': 0.15,
    'gamma': 0.9,
    'container_limit': 30,
    'population_density': 0.2,
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
