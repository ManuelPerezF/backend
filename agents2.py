import agentpy as ap
import random
import pickle
import os
import matplotlib.pyplot as plt
import time
import sys


class TrashContainerAgent(ap.Agent):
    def setup(self):
        self.position = None
        self.capacity = self.p.container_limit
        self.current_fill = 0   
        self.generation_rate = 0
    
    def step(self):
        # Generación de basura controlada por densidad
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


class TrashTruckAgent(ap.Agent):
    def setup(self):
        self.capacity = self.p.capacity
        self.load = 0
        self.position = (0, 0)
        self.q_table = {}
        self.epsilon = self.p.epsilon
        self.alpha = self.p.alpha
        self.gamma = self.p.gamma
        self.truck_id = 0  # 
        
    def load_q_table(self):
        # Carga la Q-table desde archivo si existe
        filename = f"q_table_truck_{self.truck_id}.pkl"
        if os.path.exists(filename):
            try:
                with open(filename, 'rb') as f:
                    saved_data = pickle.load(f)
                    self.q_table = saved_data['q_table']
                    # Mantener epsilon alto para seguir explorando (decae lento)
                    self.epsilon = max(0.2, saved_data.get('epsilon', self.epsilon) * 0.98)
                    # print(f"Camión {self.truck_id}: Q-table cargada con {len(self.q_table)} estados, epsilon={self.epsilon:.3f}")
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
            print(f"Error guardando Q-table para camión {self.truck_id}: {e}")

    def state(self):
        return (self.position, self.load)

    def possible_actions(self):
        return ["up", "down", "left", "right", "collect", "change_route"]

    def choose_action(self, state):
        # Prioridad 1: si hay contenedor cerca y hay espacio, recolectar
        container_at_position = self.model.get_container_at_position(self.position)
        if (container_at_position and 
            container_at_position.current_fill > 0 and 
            self.load < self.capacity):
            return "collect"
        
        # Prioridad 2: si casi lleno, ir a descargar 
        if self.load >= self.capacity * 0.8:
            return self.move_to_dump()
        
        # Prioridad 3: moverse hacia el contenedor crítico más cercano
        critical_containers = self.model.get_critical_containers()
        if critical_containers:
            return self.move_to_critical(critical_containers)
        
        # Q-learning 
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.possible_actions())
        else:
            return max(
                self.q_table.get(state, {}),
                key=self.q_table.get(state, {}).get,
                default=random.choice(self.possible_actions())
            )
    

    # Funcion para moverse hacia el punto de descarga
    def move_to_dump(self):
        x, y = self.position
        dump_points = [(0, 0), (7, 0), (0, 7), (7, 7)]
        closest_dump = min(dump_points, key=lambda p: abs(x - p[0]) + abs(y - p[1]))
        
        target_x, target_y = closest_dump
        if x < target_x: return "right"
        elif x > target_x: return "left"
        elif y < target_y: return "up"
        elif y > target_y: return "down"
        else:
            # En punto de descarga, vaciar carga
            self.load = 0
            return "collect"


    # Funcion para moverse hacia el contenedor crítico más cercano
    def move_to_critical(self, critical_containers):
        x, y = self.position
        closest_critical = min(critical_containers, key=lambda p: abs(x - p[0]) + abs(y - p[1]))
        target_x, target_y = closest_critical
        if x < target_x: return "right"
        elif x > target_x: return "left"
        elif y < target_y: return "up"
        elif y > target_y: return "down"
        else:
            return "collect"

    def update_q(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = {a: 0 for a in self.possible_actions()}
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0 for a in self.possible_actions()}

        old_value = self.q_table[state][action]
        next_max = max(self.q_table[next_state].values())
        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        self.q_table[state][action] = new_value

    def step(self):
        state = self.state()
        action = self.choose_action(state)
        reward, next_state = self.execute(action)
        self.update_q(state, action, reward, next_state)

    def execute(self, action):
        x, y = self.position
        next_pos = self.position
        reward = 0

        # Movimiento dentro del grid 8x8
        if action == "up" and y < 7:
            next_pos = (x, y + 1)
        elif action == "down" and y > 0:
            next_pos = (x, y - 1)
        elif action == "left" and x > 0:
            next_pos = (x - 1, y)
        elif action == "right" and x < 7:
            next_pos = (x + 1, y)

        # Recompensa por acercarse a críticos
        critical_containers = self.model.get_critical_containers()
        if critical_containers:
            dist_before = min(abs(x - pos[0]) + abs(y - pos[1]) for pos in critical_containers)
            dist_after  = min(abs(next_pos[0] - pos[0]) + abs(next_pos[1] - pos[1]) for pos in critical_containers)
            if dist_after < dist_before:
                reward += 2

        # Recolectar
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
            else:
                reward -= 2

        # Cambio de ruta
        if action == "change_route":
            critical_containers = self.model.get_critical_containers()
            if critical_containers:
                cx, cy = self.position
                tx, ty = min(critical_containers, key=lambda pos: abs(cx - pos[0]) + abs(cy - pos[1]))
                if cx < tx and cx < 7:   next_pos = (cx + 1, cy)
                elif cx > tx and cx > 0: next_pos = (cx - 1, cy)
                elif cy < ty and cy < 7: next_pos = (cx, cy + 1)
                elif cy > ty and cy > 0: next_pos = (cx, cy - 1)
                reward += 10
            else:
                reward -= 2

        # Penalizaciones por capacidad
        if self.load >= self.capacity:
            reward -= 20
        overflowing_containers = self.model.get_overflowing_containers()
        reward -= 30 * len(overflowing_containers)

        self.position = next_pos
        return reward, self.state()


class GarbageEnvironment(ap.Model):
    def setup(self):
        self.grid = ap.Grid(self, (8, 8), track_empty=True)

        # Posiciones de los contenedores
        container_positions = [
            (1, 1), (6, 1), (2, 5), (5, 6),
            (3, 3), (6, 5), (1, 6), (4, 2)
        ]
        self.containers = ap.AgentList(self, len(container_positions), TrashContainerAgent)
        for container, pos in zip(self.containers, container_positions):
            container.position = pos
            container.current_fill = random.randint(5, 20)

        # Solo 2 camiones
        start_positions = [(0, 0), (7, 7)]
        self.trucks = ap.AgentList(self, 2, TrashTruckAgent)
        for i, (truck, pos) in enumerate(zip(self.trucks, start_positions)):
            truck.position = pos
            truck.truck_id = i  
            truck.load_q_table() 

        # Basura inicial
        self.initial_trash = sum(container.current_fill for container in self.containers)

    def step(self):
        self.containers.step()
        self.trucks.step()

    def get_container_at_position(self, position):
        for container in self.containers:
            if container.position == position:
                return container
        return None
    
    def get_critical_containers(self):
        return [container.position for container in self.containers if container.is_critical()]
    
    def get_overflowing_containers(self):
        return [container.position for container in self.containers if container.is_overflowing()]

    def end(self):
        # Guardar las tablas Q de cada camión
        for truck in self.trucks:
            truck.save_q_table()
        
        # Mostrar estadísticas separadas por camión
        show_truck_stats(self)
        
        # Calcular estadísticas básicas  
        total_trash_remaining = sum(c.current_fill for c in self.containers)
        total_collected = sum(t.load for t in self.trucks)
        efficiency = (total_collected / max(1, total_trash_remaining + total_collected)) * 100
        
        # Mostrar resultado simple
        print(f"\nEficiencia final: {efficiency:.1f}% - {total_collected} unidades recolectadas")
        return efficiency


parameters = {
    'steps': 100,          
    'capacity': 350,       
    'epsilon': 0.3,         # exploración
    'alpha': 0.15,          # tasa de aprendizaje
    'gamma': 0.9,           # descuento futuro
    'container_limit': 30,  
    'population_density': 0.2
}



def simple_status(model):
    critical = sum(1 for c in model.containers if c.is_critical())
    overflow = sum(1 for c in model.containers if c.is_overflowing())
    total_trash = sum(c.current_fill for c in model.containers)
    total_load = sum(t.load for t in model.trucks)
    
    print(f"Críticos: {critical}, Desbordados: {overflow}, Basura: {total_trash}, Carga: {total_load}")


def show_truck_stats(model):
    """Muestra estadísticas separadas por camión"""
    print("\n--- Estadísticas por Camión ---")
    for i, truck in enumerate(model.trucks):
        efficiency = (truck.load / truck.capacity) * 100
        print(f"Camión {i}: Basura={truck.load}/{truck.capacity} ({efficiency:.1f}%) | Estados Q={len(truck.q_table)} | Posición={truck.position}")
    
    total_collected = sum(t.load for t in model.trucks)
    total_remaining = sum(c.current_fill for c in model.containers)
    overall_efficiency = (total_collected / max(1, total_collected + total_remaining)) * 100


if __name__ == "__main__":
    print("Iniciando simulación")
    model = GarbageEnvironment(parameters)
    print(f"Config: {parameters['steps']} pasos, 2 camiones, 8 contenedores")
    
    results = model.run()

