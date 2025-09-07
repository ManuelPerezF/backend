from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Tuple, Dict, Deque
from collections import deque
import threading
import time

from agents2 import GarbageEnvironment, parameters

model = GarbageEnvironment(parameters)
model.setup()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# DTOs
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

# Estado
step_queues: Dict[int, Deque[StepDTO]] = {}
_t_counter = 0
_stop_event = threading.Event()
_step_lock = threading.Lock()
_producer_thread = None
_producer_delay = 0.5

def _enqueue_step_for_truck(truck_id: int, done: bool = False):
    global _t_counter
    truck = model.trucks[truck_id]
    step = StepDTO(
        t=_t_counter,
        x=int(truck.position[0]),
        y=int(truck.position[1]),
        carrying=int(truck.load),
        action="",
        done=done
    )
    step_queues.setdefault(truck_id, deque()).append(step)

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
            model.step()
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

@app.get("/session")
def get_session():
    return {
        "gridX": 8,
        "gridY": 8,
        "totalSteps": parameters['steps'],
        "currentStep": _t_counter,
        "trucks": [TruckDTO(id=i, pos=t.position, load=t.load) for i, t in enumerate(model.trucks)],
        "containers": [ContainerDTO(pos=c.position, fill=c.current_fill) for c in model.containers],
        "dumps": [{"pos": [int(p[0]), int(p[1])]} for p in model.get_dump_points()],
    }

@app.get("/step/next")
def get_step(robot_id: int = 0):
    q = step_queues.setdefault(robot_id, deque())
    if not q:
        return Response(status_code=204)
    return q.popleft()

@app.post("/simulation/reset")
def reset_simulation():
    global model, _t_counter, _stop_event, _producer_thread
    _stop_event.set()

    model = GarbageEnvironment(parameters)
    model.setup()
    _t_counter = 0

    # reiniciar evento e hilo productor
    _stop_event = threading.Event()
    _producer_thread = threading.Thread(target=_auto_producer, daemon=True)
    _producer_thread.start()

    return {"ok": True}
