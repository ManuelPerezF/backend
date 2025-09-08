from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from collections import deque
import threading, time

from agents2 import GarbageEnvironment, parameters

# Estado global
model = GarbageEnvironment(parameters)
model.setup()


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Variables de simulación
step_queues = {}
_t_counter = 0
_stop_event = threading.Event()
_step_lock = threading.Lock()
_producer_thread = None
_producer_delay = 0.5

# Funciones de produccion de pasos para los camiones
def _enqueue_step_for_truck(truck_id: int, done: bool = False):
    global _t_counter
    truck = model.trucks[truck_id]
    step = {
        "t": _t_counter,
        "x": int(truck.position[0]),
        "y": int(truck.position[1]),
        "carrying": int(truck.load),
        "action": "",
        "done": done
    }
    step_queues.setdefault(truck_id, deque()).append(step)

# Genera automáticamente pasos
def _auto_producer():
    global _t_counter
    for i, _ in enumerate(model.trucks):
        step_queues[i] = deque()
    while not _stop_event.is_set():
        with _step_lock:
            if _t_counter >= parameters['steps']:
                for i, _ in enumerate(model.trucks):
                    _enqueue_step_for_truck(i, done=True)
                break
            model.step() # Avanza la simulación un paso
            _t_counter += 1
            for i, _ in enumerate(model.trucks):
                _enqueue_step_for_truck(i, done=False)
        time.sleep(_producer_delay)


@app.on_event("startup")
def startup_event():
    global _producer_thread, _stop_event
    _stop_event = threading.Event()
    _producer_thread = threading.Thread(target=_auto_producer, daemon=True)
    _producer_thread.start()


@app.on_event("shutdown")
def shutdown_event():
    global _stop_event
    _stop_event.set()
    if _producer_thread and _producer_thread.is_alive():
        _producer_thread.join()

# Endpoint para obtener el estado de la simulacion
@app.get("/session")
def get_session():
    return {
        "gridX": parameters.get("grid_size", 8),
        "gridY": parameters.get("grid_size", 8),
        "totalSteps": parameters['steps'],
        "currentStep": _t_counter,
        "trucks": [{"id": i, "pos": t.position, "load": t.load} for i, t in enumerate(model.trucks)],
        "containers": [{"pos": c.position, "fill": c.current_fill} for c in model.containers],
        "dumps": [{"pos": [int(p[0]), int(p[1])]} for p in model.get_dump_points()],
        
    }

# Endpoint para obtener el siguiente paso de un camión
@app.get("/step/next")
def get_step(robot_id: int = 0):
    q = step_queues.setdefault(robot_id, deque())
    if not q:
        return Response(status_code=204)
    return q.popleft()

# Endpoint para reiniciar la simulacion
@app.post("/simulation/reset")
async def reset_simulation(request: Request):
    global model, parameters, _t_counter, _stop_event, _producer_thread
    _stop_event.set()

    body = await request.json()
    # Actualiza parámetros con lo que mande Unity
    parameters.update(body)

    model = GarbageEnvironment(parameters)
    model.setup()
    _t_counter = 0

    _stop_event = threading.Event()
    _producer_thread = threading.Thread(target=_auto_producer, daemon=True)
    _producer_thread.start()

    return {"ok": True, "parameters": parameters}
