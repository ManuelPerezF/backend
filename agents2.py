# agents2.py
import agentpy as ap
import random
import pickle
import os
import matplotlib.pyplot as plt
import time
import sys

# --------------------------
# Trash Container Agent
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


# --------------------------
# Trash Truck Agent (con Q-learning)
class TrashTruckAgent(ap.Agent):

    def setup(self):
        self.capacity = self.p.capacity
        self.load = 0
        self.position = (0, 0)
        self.q_table = {}
        self.epsilon = self.p.epsilon
        self.alpha = self.p.alpha
        self.gamma = self.p.gamma
        self.truck_id = 0  # Se asignará en el modelo
        
        # Cargar Q-table si existe
        self.load_q_table()

    def load_q_table(self):
        """Carga la Q-table desde archivo si existe"""
        filename = f"q_table_truck_{self.truck_id}.pkl"
        if os.path.exists(filename):
            try:
                with open(filename, 'rb') as f:
                    saved_data = pickle.load(f)
                    self.q_table = saved_data['q_table']
                    # Mantener epsilon alto para seguir explorando (decae lento)
                    self.epsilon = max(0.2, saved_data.get('epsilon', self.epsilon) * 0.98)
                    print(f"🔄 Camión {self.truck_id}: Q-table cargada con {len(self.q_table)} estados, epsilon={self.epsilon:.3f}")
            except Exception as e:
                print(f"⚠️ Error cargando Q-table para camión {self.truck_id}: {e}")
                
    def save_q_table(self):
        """Guarda la Q-table en archivo"""
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
            print(f"💾 Camión {self.truck_id}: Q-table guardada con {len(self.q_table)} estados (ejecución #{training_runs})")
        except Exception as e:
            print(f"⚠️ Error guardando Q-table para camión {self.truck_id}: {e}")

    def state(self):
        return (self.position, self.load)

    def possible_actions(self):
        return ["up", "down", "left", "right", "collect", "change_route"]

    def choose_action(self, state):
        # Prioridad 1: si hay contenedor aquí y hay espacio, recolectar
        container_at_position = self.model.get_container_at_position(self.position)
        if (container_at_position and 
            container_at_position.current_fill > 0 and 
            self.load < self.capacity):
            return "collect"
        
        # Prioridad 2: si casi lleno, ir a descargar (esquinas)
        if self.load >= self.capacity * 0.8:
            return self.move_to_dump()
        
        # Prioridad 3: moverse hacia el contenedor crítico más cercano
        critical_containers = self.model.get_critical_containers()
        if critical_containers:
            return self.move_to_critical(critical_containers)
        
        # Q-learning (exploración/explotación)
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.possible_actions())
        else:
            return max(
                self.q_table.get(state, {}),
                key=self.q_table.get(state, {}).get,
                default=random.choice(self.possible_actions())
            )
    
    def move_to_dump(self):
        """Moverse hacia el punto de descarga más cercano (esquinas)"""
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
            return "collect"  # acción dummy para reforzar estado
    
    def move_to_critical(self, critical_containers):
        """Moverse hacia el contenedor crítico más cercano"""
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

        # Redirección rápida hacia crítico
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

        # Penalizaciones suaves
        if self.load >= self.capacity:
            reward -= 20
        overflowing_containers = self.model.get_overflowing_containers()
        reward -= 30 * len(overflowing_containers)

        self.position = next_pos
        return reward, self.state()


# --------------------------
# Garbage Environment (Modelo del entorno)
class GarbageEnvironment(ap.Model):

    def setup(self):
        self.grid = ap.Grid(self, (8, 8), track_empty=True)

        # ✅ 8 contenedores distribuidos (puedes cambiarlos si tu mapa lo requiere)
        container_positions = [
            (1, 1), (6, 1), (2, 5), (5, 6),
            (3, 3), (6, 5), (1, 6), (4, 2)
        ]
        self.containers = ap.AgentList(self, len(container_positions), TrashContainerAgent)
        for container, pos in zip(self.containers, container_positions):
            container.position = pos
            container.current_fill = random.randint(5, 20)

        # ✅ SOLO 2 camiones (esquinas opuestas para cubrir la ciudad)
        start_positions = [(0, 0), (7, 7)]
        self.trucks = ap.AgentList(self, 2, TrashTruckAgent)
        for i, (truck, pos) in enumerate(zip(self.trucks, start_positions)):
            truck.position = pos
            truck.truck_id = i  # Asignar ID único

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
        # Guardar Q-tables al final de la simulación
        for truck in self.trucks:
            truck.save_q_table()
        
        # Estadísticas
        total_trash_remaining = sum(c.current_fill for c in self.containers)
        total_collected = sum(t.load for t in self.trucks)
        efficiency = (total_collected / max(1, total_trash_remaining + total_collected)) * 100
        
        print(f"\n🎯 RESULTADOS FINALES:")
        print(f"   • Eficiencia de recolección: {efficiency:.1f}%")
        print(f"   • Basura recolectada: {total_collected} unidades")
        print(f"   • Basura restante en contenedores: {total_trash_remaining} unidades")
        print(f"   • Estados aprendidos por camión: {[len(t.q_table) for t in self.trucks]}")
        print(f"   • Epsilon final por camión: {[f'{t.epsilon:.3f}' for t in self.trucks]}")
        print(f"   • 🚀 ¡La próxima ejecución será más eficiente!")
        return efficiency


# --------------------------
# Parámetros del modelo (usados por el backend)
parameters = {
    'steps': 1000,            # pasos por episodio
    'capacity': 35,         # capacidad de cada camión
    'epsilon': 0.3,         # exploración
    'alpha': 0.15,          # tasa de aprendizaje
    'gamma': 0.9,           # descuento futuro
    'container_limit': 30,  # capacidad de contenedores
    'population_density': 0.2
}


# --------------------------
# Visualización local opcional
def realtime_simulation(model, steps=20, delay=0.5):
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    for step in range(steps):
        ax1.clear()
        ax1.grid(True)
        ax1.set_xlim(-0.5, 7.5)
        ax1.set_ylim(-0.5, 7.5)
        ax1.set_xticks(range(8))
        ax1.set_yticks(range(8))
        ax1.set_title(f"Simulación de Basura - Paso {step}")
        
        # Contenedores
        critical_count = 0
        overflow_count = 0
        total_trash = 0
        for i, c in enumerate(model.containers):
            x, y = c.position
            color = 'red' if c.is_critical() else ('orange' if c.is_overflowing() else 'green')
            if c.is_critical(): critical_count += 1
            if c.is_overflowing(): overflow_count += 1
            total_trash += c.current_fill
            
            ax1.scatter(x, y, s=300, c=color, marker='s', edgecolors='black', alpha=0.8)
            ax1.text(x, y+0.15, f"{c.current_fill}/{c.capacity}", ha='center', fontsize=7, weight='bold')
            ax1.text(x, y-0.3, f"C{i}", ha='center', fontsize=6, color='black')
        
        # Camiones
        active_trucks = 0
        total_load = 0
        for i, t in enumerate(model.trucks):
            x, y = t.position
            total_load += t.load
            
            q_size = len(t.q_table)
            truck_color = 'lightblue'
            if q_size > 50:   truck_color = 'darkblue'
            elif q_size > 20: truck_color = 'blue'
            
            # cuenta activos si ya se movió o lleva carga
            if t.load > 0 or t.position not in [(0,0), (7,7)]:
                active_trucks += 1
                
            ax1.scatter(x, y, s=250, c=truck_color, marker='o', edgecolors='black', alpha=0.9)
            ax1.text(x, y-0.35, f"{t.load}", ha='center', fontsize=8, color='white', weight='bold')
            ax1.text(x+0.3, y+0.3, f"T{i}", ha='center', fontsize=6, color='black')
            ax1.text(x+0.3, y-0.3, f"ε:{t.epsilon:.2f}", ha='center', fontsize=5, color='purple')
        
        # Panel lateral
        ax2.clear()
        ax2.axis('off')
        ax2.set_title("Estadísticas de Entrenamiento", fontsize=12, weight='bold')
        
        n_trucks = len(model.trucks)
        n_conts = len(model.containers)
        stats_text = f"""
ESTADO DE LA SIMULACIÓN (Paso {step})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚛 CAMIONES:
  • Activos: {active_trucks}/{n_trucks}
  • Carga total: {total_load}
  • Capacidad total: {n_trucks * model.trucks[0].capacity}

📦 CONTENEDORES:
  • Críticos: {critical_count}/{n_conts}
  • Desbordados: {overflow_count}/{n_conts}
  • Basura total (visible): {total_trash}
"""
        for i, truck in enumerate(model.trucks):
            q_size = len(truck.q_table)
            avg_q = (sum(sum(actions.values()) for actions in truck.q_table.values()) / 
                     max(1, q_size * 6)) if q_size > 0 else 0
            if q_size > 50:   level = "🟢 EXPERTO"
            elif q_size > 20: level = "🟡 INTERMEDIO"
            elif q_size > 5:  level = "🟠 NOVATO"
            else:             level = "🔴 SIN ENTRENAR"
            stats_text += f"""
Camión {i} ({level}):
  • Q-Table: {q_size} estados
  • Q-valor promedio: {avg_q:.2f}
  • Epsilon: {truck.epsilon:.3f}
  • Posición: {truck.position}
  • Carga: {truck.load}/{truck.capacity}
"""
        ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes, fontsize=8, 
                 verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.pause(delay)
        model.step()
    plt.ioff()
    plt.close(fig)


# --------------------------
# Entrenamiento local opcional
if __name__ == "__main__":
    print("🚛 INICIANDO SIMULACIÓN DE RECOLECCIÓN DE BASURA")
    model = GarbageEnvironment(parameters)
    print(f"🎮 Simulación configurada: {parameters['steps']} pasos, {2} camiones, {8} contenedores")
    # Visual (usar --visual) o entrenamiento rápido
    if len(sys.argv) > 1 and sys.argv[1] == "--visual":
        model.setup()
        realtime_simulation(model, steps=50, delay=0.3)
    else:
        results = model.run()
    print("\n✅ Simulación completada.")
