# Analysis

The first section below covers baseline behavior on small weighted grids.
The sections after that cover what a small grid cannot show: a real city's
street network, the difference between a heuristic that is admissible and
one that is actually useful, and what happens when the map changes while a
route is being driven.

Every number here comes from code in this repository and can be
regenerated:

```bash
python examples/grid_demo.py --map maps/map3.txt --algorithm classic
python examples/build_graph.py
python examples/replanning_demo.py
python -m search_algorithms.benchmark.run_benchmark --trials 20
```

---

## Weighted grids

The algorithms are implemented against a generalized `Problem` interface
(cost per edge rather than per cell, heuristic supplied by the problem).
Three small hand-built grids confirm that BFS, UCS, and A\* behave the way
the textbook predicts before moving to a real map.

These numbers are checked by
[`tests/test_algorithms.py`](../tests/test_algorithms.py); the test suite
fails if a future change alters them.

### Uniform-cost grid (`maps/map1.txt`)

| Algorithm | Path cost | Steps | Nodes expanded |
|---|---:|---:|---:|
| BFS | 11 | 11 | 25 |
| UCS | 11 | 11 | 25 |
| A\* | 11 | 11 | **19** |

When every step costs the same, BFS and UCS expand the same nodes in the
same way — they are effectively the same algorithm here. A\* finds the same
path while expanding 24% fewer nodes.

### Weighted grid (`maps/map2.txt`)

| Algorithm | Path cost | Steps | Nodes expanded |
|---|---:|---:|---:|
| BFS | 57 | **8** | 15 |
| UCS | 12 | 12 | 13 |
| A\* | **12** | 12 | **12** |

BFS returns the path with the fewest steps, but that path costs almost
five times as much as the cheapest one. Fewest steps is not the same as
cheapest.

### Heuristic search (`maps/map3.txt`)

| Algorithm | Path cost | Steps | Nodes expanded |
|---|---:|---:|---:|
| BFS | 28 | 28 | 336 |
| UCS | 28 | 28 | 336 |
| A\* | 28 | 28 | **126** |

![Grid comparison](images/grid_comparison.png)

On the largest map, A\* expands 62% fewer nodes than BFS or UCS. This is
where the heuristic makes the biggest difference.

---

## Real city

Tuscaloosa's drivable street network: 4,630 intersections, 11,234 road
segments. Edge costs are in meters. The heuristic is great-circle distance
to the goal.

Route: Bryant-Denny Stadium to Tuscaloosa Amphitheater (1,764 m apart in a
straight line).

| Algorithm | Distance | Intersections | Nodes expanded | Time |
|---|---:|---:|---:|---:|
| BFS | 3,431 m | **19** | 702 | 1.3 ms |
| UCS | 2,271 m | 24 | 583 | 6.7 ms |
| A\* | **2,271 m** | 24 | **146** | 2.6 ms |

![Route comparison](images/route_comparison.png)

Two results match the grid experiments, one does not.

Matching the grid: BFS again minimizes the number of intersections, not
distance — 19 intersections but a 3.4 km drive, 51% longer than necessary.
UCS and A\* again return the same optimal distance.

Different from the grid: A\*'s advantage is bigger here — 4.0x fewer node
expansions, compared to 2.7x on the largest grid map. The next section
explains why.

---

## Admissible vs. informative

A\* is guaranteed to find the optimal path as long as the heuristic never
overestimates the remaining cost (this is called admissible). But how much
work A\* saves depends on how close the estimate is to the true cost, which
is a separate property. Random grids show this clearly.

20 random 10x10 to 40x40 grids per size, 25% obstacles:

| Terrain costs | Mean A\* vs UCS expansions | Same optimal cost? |
|---|---|---|
| all 1 (uniform) | 1.76x fewer | yes |
| 1-3 | 1.11x fewer | yes |
| 1-9 | 1.02x fewer | yes |

![Benchmark, uniform cost](images/benchmark_uniform.png)

Manhattan distance counts moves and assumes each one costs at least 1.
When terrain costs go up to 9, the true remaining cost can be nine times
higher than the estimate. The heuristic is still admissible — A\* still
returns the optimal path every time — but the estimate is so far below the
true cost that `f(n) = g(n) + h(n)` is dominated by `g(n)`, and A\* ends up
behaving almost exactly like UCS.

This also explains the city result. There, edge costs are in meters and
the heuristic is also a distance in meters, so the units match and the
estimate stays close to the true cost on roads that run roughly straight.
A heuristic that is well matched to the cost units on 4,630 nodes performs
better than a poorly matched one on 1,600 nodes.

The takeaway: admissibility guarantees correctness. How informative the
heuristic is determines speed. These are two separate properties, and
usually only the first one is checked.

---

## Replanning

A planned route assumes the world does not change. This experiment closes
three road segments along a planned route. The agent only finds out about
each closure when it reaches that point, and has to replan from wherever
it currently is.

| | Distance driven | Total nodes expanded |
|---|---:|---:|
| Knows all closures in advance | 2,533 m | 193 |
| Discovers closures on arrival | **4,109 m** | **387** |
| Difference | +1,576 m (+62%) | 2.01x |

Not knowing about the closures costs twice: the detour itself (1,576 m
extra driving), and the extra search work. Each time the agent has to
replan, it searches from scratch and re-expands territory the earlier
search had already covered — solving three medium-sized search problems
ends up costing about twice as much as solving one large one would.

This is the exact motivation for D\* Lite, which reuses the previous search
tree instead of discarding it. Implementing D\* Lite and measuring the same
two columns is the clearest next experiment for this project.

---

## Follow-up questions

**1. Why can BFS return a more expensive path than UCS on a weighted grid?**

BFS orders its frontier by depth, not cost, and returns the first path
that reaches the goal. That path has the fewest edges, but fewest edges
only means cheapest when every edge costs the same. The weighted grid
(`maps/map2.txt`) shows the gap at its widest: 8 steps costing 57, versus
12 steps costing 12. The same thing happens on the street network for the
same reason: BFS uses 19 intersections but drives 1.2 km further than
necessary.

**2. If UCS and A\* return the same cost, why does A\* expand fewer states?**

Both are best-first searches that differ only in how they order the
frontier. UCS uses `g(n)`, the cost already spent getting there, so it
expands outward in every direction equally, including away from the goal.
A\* uses `f(n) = g(n) + h(n)`, adding an estimate of the remaining cost,
which lowers the priority of states that are cheap to reach but point the
wrong way. With an admissible `h`, A\* never expands a state whose `f`
exceeds the optimal cost, so it never expands more states than UCS, and
usually expands far fewer. In the route figures, UCS's expanded region is
roughly circular; A\*'s is stretched toward the goal.

**3. What happens to A\* when h(n) = 0 everywhere?**

`f(n) = g(n) + 0 = g(n)`, which is exactly UCS's priority function. A\*
becomes UCS, including identical tie-breaking and an identical expansion
order. This is checked in
[`tests/test_problem_abc.py::test_zero_heuristic_makes_astar_equal_ucs`](../tests/test_problem_abc.py),
which compares the full `expansion_order` lists, not just the counts. The
zero heuristic is technically admissible, since it never overestimates —
it just never estimates anything, which is why it does not save any work.

**4. If Manhattan distance h(n) is replaced by 2·h(n), is A\* still optimal?**

Not always. Doubling the estimate can push it above the true remaining
cost, which makes the heuristic inadmissible. When that happens, A\* can
settle on a suboptimal path — it may return the goal while a cheaper route
is still sitting in the frontier, because that cheaper route looked worse
due to the inflated estimate.

There is still a bound, though. Weighted A\* with `f = g + w·h` returns a
path that costs at most `w` times the optimal cost. So `2h` gives a result
that is at most 2x the optimal cost, and in practice it is usually much
closer than that while expanding far fewer nodes. This is implemented as
`weighted_astar`, and the bound is checked in
[`tests/test_algorithms.py::test_weighted_astar_is_bounded_suboptimal`](../tests/test_algorithms.py).

One detail from the admissible-vs-informative section is worth noting here:
on the weighted grids used in
this project, Manhattan distance underestimates the true cost by so much
that doubling it (`2h`) usually still stays admissible in practice. Whether
doubling the heuristic actually breaks optimality depends on how much
slack the original heuristic had.

---

## Contraction Hierarchies

Parts 1-4 all use algorithms that search the graph fresh on every query,
including A\*.

[Contraction Hierarchies](contraction_hierarchies.md) removes that cost by
preprocessing the street network once and answering each query by
searching only a small part of it. On the same Tuscaloosa graph used
above, it expands 87 intersections where Dijkstra expands 2,378 — 13x
faster per query, and all 200 sampled routes returned exactly Dijkstra's
optimal distance.

That document covers the method, how correctness is verified, the
preprocessing cost involved, and the point at which that cost is worth
paying.
