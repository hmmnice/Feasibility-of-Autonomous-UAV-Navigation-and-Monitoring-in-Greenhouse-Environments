"""Small-N task ordering and battery feasibility for multi-crop extensions."""

import itertools
import numpy as np


def route_distance(start, ordered_points, return_point=None):
    start = np.asarray(start, dtype=float)
    points = [np.asarray(point, dtype=float) for point in ordered_points]
    chain = [start] + points
    if return_point is not None:
        chain.append(np.asarray(return_point, dtype=float))
    return float(sum(
        np.linalg.norm(end - begin)
        for begin, end in zip(chain, chain[1:])))


def nearest_neighbour_order(start, tasks):
    """Return stable nearest-neighbour ordering for task dictionaries."""
    remaining = list(tasks)
    current = np.asarray(start, dtype=float)
    ordered = []
    while remaining:
        index = min(
            range(len(remaining)),
            key=lambda item: (
                float(np.linalg.norm(
                    np.asarray(remaining[item]["position"], dtype=float)
                    - current)),
                str(remaining[item].get("id", item)),
            ))
        chosen = remaining.pop(index)
        ordered.append(chosen)
        current = np.asarray(chosen["position"], dtype=float)
    return ordered


def exact_small_n_order(start, tasks, return_point=None, maximum_tasks=8):
    """Return globally shortest task order for a deliberately small set."""
    tasks = tuple(tasks)
    if len(tasks) > int(maximum_tasks):
        raise ValueError("exact ordering is limited to %d tasks" % maximum_tasks)
    if not tasks:
        return []
    return list(min(
        itertools.permutations(tasks),
        key=lambda order: (
            route_distance(
                start, [task["position"] for task in order], return_point),
            tuple(str(task.get("id", "")) for task in order),
        ),
    ))


def battery_budget(
        route_length_m, cruise_speed_mps, fixed_task_time_s,
        task_count, usable_flight_time_s, return_reserve_fraction=0.20):
    """Estimate whether a multi-target plan preserves the return reserve."""
    if cruise_speed_mps <= 0.0 or usable_flight_time_s <= 0.0:
        raise ValueError("speed and usable flight time must be positive")
    travel_time = float(route_length_m) / float(cruise_speed_mps)
    inspection_time = float(fixed_task_time_s) * int(task_count)
    required = travel_time + inspection_time
    available = float(usable_flight_time_s) * (
        1.0 - float(return_reserve_fraction))
    return {
        "travel_time_s": travel_time,
        "inspection_time_s": inspection_time,
        "required_time_s": required,
        "available_before_reserve_s": available,
        "return_reserve_fraction": float(return_reserve_fraction),
        "feasible": bool(required <= available + 1e-9),
        "time_margin_s": available - required,
    }
