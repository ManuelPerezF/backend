from fastapi import FastAPI, Response, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Tuple, Dict, Deque
from collections import deque
import threading
import time

from agents2 import GarbageEnvironment, parameters

# ---- Modelo y parámetros ----
model = GarbageEnvironment(parameters)
model.setup()

# TOTAL_STEPS ahora se obtiene dinámicamente de los parámetros actuales

# ---- FastAPI ----
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ---- DTOs ----
class TruckDTO(BaseModel):
    id: int
    pos: Tuple[int, int]
    load: int

class ContainerDTO(BaseModel):
    pos: Tuple[int, int]
    fill: int

class StepDTO(BaseModel):
    t: int
    x: int
    y: int
    carrying: int
    action: str
    done: bool = False

class StateDTO(StepDTO):
    pass

class ResetDTO(BaseModel):
    start: Tuple[int, int] = (0, 0)
    robot_id: int = 0

# ---- Estado ----
step_queues: Dict[int, Deque[StepDTO]] = {0: deque()}
current_state: Dict[int, StateDTO] = {}
_producer_thread: threading.Thread | None = None
_stop_event = threading.Event()
_step_lock = threading.Lock()
_t_counter = 0
_producer_delay = 0.8  # seconds between steps


def _ensure_queue(robot_id: int) -> Deque[StepDTO]:
    return step_queues.setdefault(robot_id, deque())


def _enqueue_step_for_truck(truck_id: int, done: bool = False):
    """Helper to enqueue the current state of a given truck."""
    q = _ensure_queue(truck_id)
    truck = model.trucks[truck_id]
    step = StepDTO(
        t=_t_counter,
        x=int(truck.position[0]),
        y=int(truck.position[1]),
        carrying=int(truck.load),
        action="MOVE",
        done=done,
    )
    q.append(step)


def _auto_producer():
    global _t_counter
    # Pre-create queues for each truck id present in the model
    for i, _ in enumerate(model.trucks):
        _ensure_queue(i)

    while not _stop_event.is_set():
        with _step_lock:
            if _t_counter >= parameters['steps']:
                # Mark done once and stop
                for i, _ in enumerate(model.trucks):
                    _enqueue_step_for_truck(i, done=True)
                break

            # Advance the simulation one step
            model.step()
            _t_counter += 1

            # Enqueue a step per truck reflecting its new position/load
            for i, _ in enumerate(model.trucks):
                _enqueue_step_for_truck(i, done=False)

        # pacing
        time.sleep(_producer_delay)

    # Save Q-tables at end
    try:
        model.end()
    except Exception:
        pass


def _stop_producer(join_timeout: float = 2.0):
    _stop_event.set()
    if _producer_thread and _producer_thread.is_alive():
        _producer_thread.join(timeout=join_timeout)


def _start_producer():
    global _producer_thread
    _stop_event.clear()
    if _producer_thread is None or not _producer_thread.is_alive():
        _producer_thread = threading.Thread(target=_auto_producer, name="auto_producer", daemon=True)
        _producer_thread.start()


def _reset_model_and_counters():
    """Reinicia completamente la simulación: modelo, contador y colas."""
    global model, _t_counter, parameters
    with _step_lock:
        # Recargar parámetros actualizados
        import importlib
        import agents2
        importlib.reload(agents2)
        # Traer los símbolos actualizados desde agents2 y actualizar el estado global
        from agents2 import GarbageEnvironment, parameters as new_parameters

        # Actualizar la variable de parámetros a la versión recargada
        parameters = new_parameters

        # recrear modelo con parámetros actualizados
        model = GarbageEnvironment(parameters)
        model.setup()
        _t_counter = 0
        # limpiar colas y crear una por camión con un paso inicial t=0
        for k in list(step_queues.keys()):
            try:
                step_queues[k].clear()
            except Exception:
                step_queues[k] = deque()
        for i, _ in enumerate(model.trucks):
            _ensure_queue(i)
            _enqueue_step_for_truck(i, done=False)

        print(f"🔄 Modelo reseteado con {parameters['steps']} pasos configurados")

# ---- Endpoints ----
@app.get("/session")
def get_session(reset: bool = False):
    # Opción para resetear cuando Unity solicite la sesión
    if reset:
        _stop_producer()
        _reset_model_and_counters()
        _start_producer()
    print(f"HTTP GET /session called (reset={reset}) -> totalSteps={parameters.get('steps','<missing>')} currentStep={_t_counter}")
    resp = {
        "gridX": 8,
        "gridY": 8,
        "totalSteps": parameters['steps'],  #  Dinámico desde parameters
        "currentStep": _t_counter,
        "trucks": [TruckDTO(id=i, pos=truck.position, load=truck.load) for i, truck in enumerate(model.trucks)],
        "containers": [ContainerDTO(pos=c.position, fill=c.current_fill) for c in model.containers]
    }
    print(f"HTTP GET /session response: totalSteps={resp['totalSteps']} currentStep={resp['currentStep']} trucks={len(resp['trucks'])} containers={len(resp['containers'])}")
    return resp

@app.get("/step/next", response_model=StepDTO, responses={204: {"description": "No Content"}})
def get_next_step(robot_id: int = 0):
    q = _ensure_queue(robot_id)
    if not q:
        return Response(status_code=204)
    return q.popleft()

@app.post("/step")
def post_step(step: StepDTO = Body(...), robot_id: int = 0):
    if step.t >= parameters['steps'] - 1:  #  cuando llega al límite dinámico
        step.done = True
    _ensure_queue(robot_id).append(step)
    return {"ok": True}

@app.post("/episode/reset")
def episode_reset(req: ResetDTO):
    step_queues[req.robot_id] = deque()
    step_queues[req.robot_id].append(
        StepDTO(t=0, x=req.start[0], y=req.start[1], carrying=0, action="RESET", done=False)
    )
    return {"ok": True}

@app.get("/state", response_model=StateDTO, responses={204: {"description": "No Content"}})
def get_state(robot_id: int = 0):
    st = current_state.get(robot_id)
    if not st:
        return Response(status_code=204)
    return st

@app.post("/state", response_model=StateDTO)
def post_state(state: StateDTO = Body(...), robot_id: int = 0):
    current_state[robot_id] = state
    return state


@app.on_event("startup")
def _on_startup():
    global _producer_thread, _t_counter
    _t_counter = 0
    _stop_event.clear()
    print(f"🚀 Backend iniciado con {parameters['steps']} pasos configurados")
    if _producer_thread is None or not _producer_thread.is_alive():
        _producer_thread = threading.Thread(target=_auto_producer, name="auto_producer", daemon=True)
        _producer_thread.start()


@app.on_event("shutdown")
def _on_shutdown():
    _stop_producer()


@app.post("/simulation/reset")
def simulation_reset():
    """Reinicia la simulación, contador de pasos y productor."""
    _stop_producer()
    _reset_model_and_counters()
    _start_producer()
    resp = {
        "ok": True,
        "totalSteps": parameters['steps'],
        "currentStep": _t_counter,
        "trucks": len(model.trucks),
    }
    print(f"HTTP POST /simulation/reset called -> returning: {resp}")
    return resp
