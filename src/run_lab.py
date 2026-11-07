"""
Точка входа лабораторной работы.

Примеры запуска из папки Е86_муравьиный_алгоритм:

  python src/run_lab.py --map demo
  python src/run_lab.py --map random --cities 15 --seed 7 --alpha 1 --beta 5 --rho 0.5
  python src/run_lab.py --map clusters --ants 30 --iters 150 --no-show
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Чтобы можно было запускать и как скрипт, и как модуль
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aco import ACOParams, AntColonyTSP, nearest_neighbor_length  # noqa: E402
from maps import distance_matrix, get_map  # noqa: E402


def plot_results(cities, best_tour, history, out_dir: Path, show: bool) -> None:
    """Две картинки: маршрут и сходимость лучшей длины."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pts = np.asarray(cities, dtype=float)

    # --- Маршрут ---
    fig1, ax1 = plt.subplots(figsize=(7, 6))
    ax1.scatter(pts[:, 0], pts[:, 1], c="tab:blue", s=60, zorder=3)
    for i, (x, y) in enumerate(pts):
        ax1.annotate(str(i), (x, y), textcoords="offset points", xytext=(5, 5))

    order = list(best_tour) + [best_tour[0]]
    path = pts[order]
    ax1.plot(path[:, 0], path[:, 1], "r-", lw=2, zorder=2)
    ax1.set_title("Лучший маршрут ACO (TSP / курьер)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    route_path = out_dir / "best_route.png"
    fig1.savefig(route_path, dpi=140)

    # --- Сходимость ---
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.plot(history, lw=2)
    ax2.set_title("Сходимость: лучшая длина маршрута по итерациям")
    ax2.set_xlabel("Итерация")
    ax2.set_ylabel("Длина лучшего маршрута")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    hist_path = out_dir / "convergence.png"
    fig2.savefig(hist_path, dpi=140)

    print(f"Сохранено: {route_path}")
    print(f"Сохранено: {hist_path}")

    if show:
        plt.show()
    else:
        plt.close("all")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Лабораторная: муравьиный алгоритм (TSP)")
    p.add_argument("--map", default="demo", help="demo|ring|clusters|random")
    p.add_argument("--cities", type=int, default=12, help="число городов для random")
    p.add_argument("--seed", type=int, default=42, help="seed карты/ГПСЧ")
    p.add_argument("--ants", type=int, default=20, help="число муравьёв")
    p.add_argument("--iters", type=int, default=100, help="число итераций")
    p.add_argument("--alpha", type=float, default=1.0, help="влияние феромона")
    p.add_argument("--beta", type=float, default=5.0, help="влияние эвристики 1/d")
    p.add_argument("--rho", type=float, default=0.5, help="испарение феромона (0..1)")
    p.add_argument("--q", type=float, default=100.0, help="константа отложения феромона")
    p.add_argument("--out", default="results", help="папка для графиков и JSON")
    p.add_argument("--no-show", action="store_true", help="не открывать окна matplotlib")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cities = get_map(args.map, n_cities=args.cities, seed=args.seed)
    dist = distance_matrix(cities)

    params = ACOParams(
        n_ants=args.ants,
        n_iterations=args.iters,
        alpha=args.alpha,
        beta=args.beta,
        rho=args.rho,
        q=args.q,
        seed=args.seed,
    )

    print("=" * 60)
    print("Муравьиный алгоритм (Ant System) для TSP / курьерской задачи")
    print("=" * 60)
    print(f"Карта: {args.map}, городов: {len(cities)}")
    print(
        f"Параметры: ants={params.n_ants}, iters={params.n_iterations}, "
        f"alpha={params.alpha}, beta={params.beta}, rho={params.rho}, Q={params.q}"
    )

    nn_tour, nn_len = nearest_neighbor_length(dist, start=0)
    print(f"Базовый жадный (nearest neighbor): длина = {nn_len:.4f}, маршрут = {nn_tour}")

    aco = AntColonyTSP(dist, params)
    result = aco.run()

    print(f"ACO: лучшая длина = {result.best_length:.4f}")
    print(f"ACO: лучший маршрут = {result.best_tour}")
    improve = 100.0 * (nn_len - result.best_length) / nn_len if nn_len > 0 else 0.0
    print(f"Улучшение относительно жадного: {improve:.2f}%")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "map": args.map,
        "n_cities": len(cities),
        "params": {
            "n_ants": params.n_ants,
            "n_iterations": params.n_iterations,
            "alpha": params.alpha,
            "beta": params.beta,
            "rho": params.rho,
            "q": params.q,
            "seed": params.seed,
        },
        "nearest_neighbor": {"tour": nn_tour, "length": nn_len},
        "aco": {"tour": result.best_tour, "length": result.best_length},
        "improvement_percent": improve,
        "history": result.history,
    }
    json_path = out_dir / "summary.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Сохранено: {json_path}")

    plot_results(cities, result.best_tour, result.history, out_dir, show=not args.no_show)


if __name__ == "__main__":
    main()
