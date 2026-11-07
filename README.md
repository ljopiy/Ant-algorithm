# Е86 — Лабораторная работа: муравьиный алгоритм (TSP / курьер)

Учебный проект к заданию **2025_Е86**: изучение Ant System на задаче коммивояжера с транспортной интерпретацией (маршрут курьера по пунктам доставки).

## Быстрый старт

```bash
python -m pip install -r requirements.txt
python src/run_lab.py --map demo --no-show
```

Результаты: папка `results/` (`best_route.png`, `convergence.png`, `summary.json`).

## Документы

- **`Методические_указания.md`** — постановка, теория, пошаговые инструкции, варианты заданий, требования к мини-отчёту.
- **`src/aco.py`** — реализация алгоритма с комментариями к формулам.
- **`src/maps.py`** — карты пунктов (`demo`, `ring`, `clusters`, `random`).
- **`src/run_lab.py`** — запуск экспериментов из командной строки.

## Примеры вариативности

```bash
python src/run_lab.py --map ring --beta 8 --no-show --out results_B
python src/run_lab.py --map clusters --rho 0.8 --no-show --out results_C
python src/run_lab.py --map random --cities 12 --seed 142 --ants 30 --iters 120 --no-show
```

Подробности и правила выбора варианта — в методичке.
