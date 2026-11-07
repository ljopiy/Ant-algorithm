"""
Муравьиный алгоритм (Ant System) для задачи коммивояжера (TSP).

Идея (кратко):
  - Муравьи строят маршруты, выбирая следующий город вероятностно.
  - Вероятность зависит от феромона на ребре (τ) и эвристики η = 1/d.
  - После итерации феромон испаряется и усиливается на пройденных рёбрах.

Основные формулы Ant System (Dorigo et al.):
  P_ij = (τ_ij^α * η_ij^β) / Σ_k (τ_ik^α * η_ik^β)   — выбор следующего города
  τ_ij ← (1 - ρ) * τ_ij + Σ_a Δτ_ij^a                   — обновление феромона
  Δτ_ij^a = Q / L_a , если муравей a прошёл ребро (i,j); иначе 0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class ACOParams:
    """Параметры, которые студент может менять в эксперименте."""

    n_ants: int = 20          # число муравьёв за итерацию
    n_iterations: int = 100   # число итераций
    alpha: float = 1.0        # влияние феромона τ
    beta: float = 5.0         # влияние эвристики η = 1/d
    rho: float = 0.5          # скорость испарения феромона (0..1)
    q: float = 100.0          # константа отложения феромона
    tau0: float = 1.0         # начальный феромон на всех рёбрах
    seed: Optional[int] = 42  # для воспроизводимости


@dataclass
class ACOResult:
    best_tour: List[int]
    best_length: float
    history: List[float] = field(default_factory=list)  # лучшая длина по итерациям
    pheromone: Optional[np.ndarray] = None


class AntColonyTSP:
    """
    Реализация классического Ant System для замкнутого маршрута (TSP).

    Интерпретация для транспортной / курьерской задачи:
      города = пункты доставки, длина маршрута = суммарный пробег курьера,
      который должен объехать все пункты и вернуться на склад (город 0).
    """

    def __init__(self, distances: np.ndarray, params: ACOParams):
        if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
            raise ValueError("distances должна быть квадратной матрицей")
        self.d = np.asarray(distances, dtype=float)
        self.n = self.d.shape[0]
        self.p = params
        self.rng = np.random.default_rng(params.seed)

        # Эвристика: чем короче ребро, тем привлекательнее.
        # На диагонали расстояние 0 — ставим η=0, чтобы не выбирать «остаться».
        with np.errstate(divide="ignore"):
            self.eta = 1.0 / self.d
        np.fill_diagonal(self.eta, 0.0)

        self.tau = np.full((self.n, self.n), params.tau0, dtype=float)
        np.fill_diagonal(self.tau, 0.0)

    def _tour_length(self, tour: Sequence[int]) -> float:
        """Длина замкнутого маршрута (включая возврат в старт)."""
        length = 0.0
        for i in range(len(tour) - 1):
            length += self.d[tour[i], tour[i + 1]]
        length += self.d[tour[-1], tour[0]]
        return float(length)

    def _choose_next(self, current: int, unvisited: List[int]) -> int:
        """Вероятностный выбор следующего города по правилу Ant System."""
        # Привлекательности рёбер current -> j
        tau = self.tau[current, unvisited] ** self.p.alpha
        eta = self.eta[current, unvisited] ** self.p.beta
        scores = tau * eta

        total = scores.sum()
        if total <= 0 or not np.isfinite(total):
            # Запасной вариант: равновероятный выбор
            return int(self.rng.choice(unvisited))

        probs = scores / total
        return int(self.rng.choice(unvisited, p=probs))

    def _build_tour(self, start: int) -> List[int]:
        """Один муравей строит полный маршрут, начиная из start."""
        unvisited = list(range(self.n))
        unvisited.remove(start)
        tour = [start]
        current = start
        while unvisited:
            nxt = self._choose_next(current, unvisited)
            tour.append(nxt)
            unvisited.remove(nxt)
            current = nxt
        return tour

    def _update_pheromone(self, tours: List[List[int]], lengths: List[float]) -> None:
        """Испарение + отложение феромона всеми муравьями итерации."""
        # 1) Испарение
        self.tau *= 1.0 - self.p.rho

        # 2) Отложение: чем короче маршрут, тем больше феромона
        for tour, length in zip(tours, lengths):
            if length <= 0:
                continue
            deposit = self.p.q / length
            for i in range(len(tour) - 1):
                a, b = tour[i], tour[i + 1]
                self.tau[a, b] += deposit
                self.tau[b, a] += deposit  # симметричная TSP
            # замыкание цикла
            a, b = tour[-1], tour[0]
            self.tau[a, b] += deposit
            self.tau[b, a] += deposit

    def run(self) -> ACOResult:
        """Основной цикл алгоритма."""
        best_tour: Optional[List[int]] = None
        best_length = float("inf")
        history: List[float] = []

        for _ in range(self.p.n_iterations):
            tours: List[List[int]] = []
            lengths: List[float] = []

            for _ant in range(self.p.n_ants):
                # Случайный старт делает исследование разнообразнее
                start = int(self.rng.integers(0, self.n))
                tour = self._build_tour(start)
                length = self._tour_length(tour)
                tours.append(tour)
                lengths.append(length)

                if length < best_length:
                    best_length = length
                    best_tour = tour.copy()

            self._update_pheromone(tours, lengths)
            history.append(best_length)

        assert best_tour is not None
        return ACOResult(
            best_tour=best_tour,
            best_length=best_length,
            history=history,
            pheromone=self.tau.copy(),
        )


def nearest_neighbor_length(distances: np.ndarray, start: int = 0) -> Tuple[List[int], float]:
    """
    Жадный базовый алгоритм «ближайший сосед» — для сравнения с ACO.
    Студенту полезно увидеть, что метаэвристика часто лучше жадного старта.
    """
    n = distances.shape[0]
    unvisited = set(range(n))
    unvisited.remove(start)
    tour = [start]
    current = start
    while unvisited:
        nxt = min(unvisited, key=lambda j: distances[current, j])
        tour.append(nxt)
        unvisited.remove(nxt)
        current = nxt
    length = 0.0
    for i in range(n - 1):
        length += distances[tour[i], tour[i + 1]]
    length += distances[tour[-1], tour[0]]
    return tour, float(length)
