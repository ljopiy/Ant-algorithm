"""
Карты городов (пунктов доставки) для лабораторной работы.

Каждая карта — набор координат (x, y). Расстояния считаются
как евклидовы (прямая линия), что удобно для учебной TSP / курьерской задачи.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

Point = Tuple[float, float]


# ---------------------------------------------------------------------------
# Варианты карт. Студент выбирает карту по номеру варианта (см. методичку).
# ---------------------------------------------------------------------------
MAPS: Dict[str, List[Point]] = {
    # Компактный учебный пример (6 пунктов)
    "demo": [
        (0, 0),
        (1, 3),
        (4, 3),
        (5, 1),
        (3, 0),
        (2, 2),
    ],
    # «Кольцо» — пункты примерно по окружности
    "ring": [
        (5.0, 0.0),
        (3.5, 3.5),
        (0.0, 5.0),
        (-3.5, 3.5),
        (-5.0, 0.0),
        (-3.5, -3.5),
        (0.0, -5.0),
        (3.5, -3.5),
    ],
    # «Кластеры» — два района города
    "clusters": [
        (0, 0),
        (0.5, 0.8),
        (1.2, 0.3),
        (0.8, 1.1),
        (8, 8),
        (8.5, 7.2),
        (7.4, 8.6),
        (9.0, 8.3),
        (4, 4),  # промежуточный пункт между кластерами
    ],
    # Случайная карта фиксируется seed'ом при генерации
}


def generate_random_map(n_cities: int, seed: int, span: float = 10.0) -> List[Point]:
    """Случайные координаты в квадрате [0, span] x [0, span]."""
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0.0, span, size=(n_cities, 2))
    return [(float(x), float(y)) for x, y in coords]


def distance_matrix(cities: List[Point]) -> np.ndarray:
    """Матрица расстояний D[i, j] = расстояние между городами i и j."""
    pts = np.asarray(cities, dtype=float)
    # (n, 1, 2) - (1, n, 2) -> (n, n, 2) -> нормы по последней оси
    diff = pts[:, None, :] - pts[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))


def get_map(name: str, n_cities: int = 12, seed: int = 42) -> List[Point]:
    """
    Вернуть список городов по имени карты.

    name="random" — генерирует новую карту по seed (удобно для вариантов).
    """
    key = name.lower().strip()
    if key == "random":
        return generate_random_map(n_cities=n_cities, seed=seed)
    if key not in MAPS:
        known = ", ".join(sorted(MAPS) + ["random"])
        raise ValueError(f"Неизвестная карта '{name}'. Доступны: {known}")
    return list(MAPS[key])
