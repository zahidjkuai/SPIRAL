<div align="center">

# SPIRAL-3D

### A physics-solved neural network for 3-D incompressible flow — no training data, no backpropagation

**S**tream-function **P**hysics-**I**nformed **RA**ndom-feature **L**earning, applied to the 3-D lid-driven cavity and passive-scalar mixing

![Python](https://img.shields.io/badge/python-3.10%2B-black)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-black)
![GPU](https://img.shields.io/badge/runs%20on-Colab%20T4-black)
![Data](https://img.shields.io/badge/training%20data-none-black)

<img src="docs/overview.png" width="900" alt="SPIRAL-3D overview">

</div>

---

## In one paragraph

SPIRAL represents the flow with a one-hidden-layer neural network whose hidden weights are **random and frozen**. Only the output weights are unknown, and they are not *trained*: they are **solved** with Newton's method so that the discretised Navier–Stokes equations hold at every grid point. The velocity is the curl of the network output (a vector potential), so **mass is conserved to machine precision by construction**. The result is a neural-network solver that behaves like a classic CFD code — deterministic, convergent under mesh refinement, independent of the random seed — while keeping a continuous, differentiable network representation of the field.

## Highlights

| | |
|---|---|
| **No data** | Inputs are grid positions; the only "loss" is the discretised PDE with its boundary conditions. No CFD results, measurements or pre-training. |
| **Exact continuity** | u = ∇×A on a staggered grid → max \|∇·u\| ≈ 10⁻²⁵ for *any* weights. |
| **Newton instead of SGD** | Kernel-form Levenberg–Marquardt with Cholesky (chord re-use): the residual falls 2–3 orders of magnitude per iteration; ~5 Newton iterations per time step. |
| **Seed-independent** | Two different random hidden layers give output weights with correlation 0.006 but velocities equal to 10⁻⁹: the physics fixes the answer. |
| **Verified** | Matches a classic finite-difference solve of the same equations to round-off; 2.1 % from Ghia et al. (1982) at Re = 100 on a 24×24 mesh, error falling as 1/N². |
| **Full pipeline** | Transient 3-D flow (BDF2) → passive-scalar transport (second network, conservative FV, linear upwind) → mixing index, centre of gravity, kinetic energy, cost report. |

## Benchmarking in Lid Driven Cavity Against CFD Result
<img width="4021" height="2604" alt="Fig_mesh_domain" src="https://github.com/user-attachments/assets/9c04872a-f9f8-4056-8ce6-0e7875703e48" />
<img width="3116" height="1436" alt="Fig_injection_setup" src="https://github.com/user-attachments/assets/591820b5-01bf-4a94-962a-690bac2eaa66" />
<img width="2973" height="1535" alt="Fig_3D_velocity_isometric" src="https://github.com/user-attachments/assets/cd78c541-19a1-4c6e-9f96-4342f61d2c56" />
<img width="3322" height="1450" alt="Fig_centrelines" src="https://github.com/user-attachments/assets/b5ccae07-989f-4495-9778-91b555aa20c7" />
<img width="3839" height="3603" alt="Fig_vortices_vectors_midsection" src="https://github.com/user-attachments/assets/d5e6019c-fdd8-4495-8540-a079a61a5091" />
<img width="5212" height="1754" alt="Fig_vortices_midsection" src="https://github.com/user-attachments/assets/13933a11-1766-4805-b202-e3854b8b457e" />
<img width="2528" height="1479" alt="phi_sequence" src="https://github.com/user-attachments/assets/8ca35d0c-d7dc-4eb1-a48f-2a21e9793d81" />
<img width="3928" height="1304" alt="Fig_computational_cost" src="https://github.com/user-attachments/assets/109cb6ff-10d9-4356-8ef3-affe97ec060e" />
<img width="2550" height="714" alt="mixing_index" src="https://github.com/user-attachments/assets/3a1b09f5-f3f0-4d13-ba1c-51df4b962589" />
<img width="986" height="951" alt="cog_trajectory" src="https://github.com/user-attachments/assets/d60bedda-2b50-4838-a097-d10c9ae8e59b" />





## How it works

<img src="docs/algorithm.png" width="900" alt="Algorithm and glossary">

1. **Mesh → inputs.** Every unknown value of the vector potential sits on a grid edge; its coordinates (x, y, z) are one input row. Boundary values are known and are *not* inputs.
2. **Frozen random layer.** `a_k(x) = tanh(w_k·x + b_k)`, with `w_k = s_k d_k` (log-uniform steepness, random direction) and `b_k = −w_k·x0_k`. Drawn once from a seed, never updated.
3. **Output.** `A = Φz` on the edges, `u = curl A` on the faces.
4. **Physics residual.** Curl of the momentum equations (pressure eliminated), BDF2 in time; no-slip walls and the lid speed enter here.
5. **Newton on z.** `J = SΦ`, solve `(JJᵀ + λI)α = −r`, `Δz = Jᵀα`; repeat until the relative update < 10⁻⁸.

| Network | Solver |
|---|---|
| <img src="docs/architecture.png" width="440"> | <img src="docs/convergence.png" width="440"> |

### Why a random network gives the CFD answer

The residual is the same set of discrete equations a CFD code solves. With at least as many hidden nodes as equations, the frozen layer can represent any field at the mesh points, so `r = 0` *is* the CFD solution — with the same mesh error, which falls as the mesh is refined. The random seed changes the weights, not the flow.

<img src="docs/why_it_matches_cfd.png" width="900" alt="Newton convergence, spin-up vs Ghia, mesh refinement, seed independence">

### See it by hand

A 1-D channel flow with 3 cells, 2 unknowns and 2 hidden nodes — every number can be checked with a calculator, and [`examples/spiral_1d_hand.py`](examples/spiral_1d_hand.py) reproduces them.

<img src="docs/worked_example_1d.png" width="700" alt="1-D worked example">

## Quick start

```bash
pip install -r requirements.txt
python examples/spiral_1d_hand.py          # the 1-D example, milliseconds
S3_QUICK=1 python ldc3d_spiral.py          # 12×12×2 pipeline check on a CPU
```

**Google Colab (recommended for the full case):** Runtime → *T4 GPU*, upload `ldc3d_spiral.py`, then

```python
%run ldc3d_spiral.py
```

### Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `S3_QUICK` | `0` | `1` = tiny mesh, pipeline check only |
| `S3_OUT` | `/content/ldc3d_results` | output folder |
| `S3_RE` | `nominal` | `nominal` → Re = 1000; `table` → Re from the property table |
| `S3_ZWALL` | `symmetry` | front/back faces: `symmetry` (2-D-equivalent) or `noslip` (truly 3-D) |
| `S3_CFG` | — | JSON overrides of `CFG`, e.g. `'{"N": 40, "Nz": 4}'` |
| `RCFD_DIR` | `/content/rcfd` | optional ANSYS Fluent reference CSV files for comparison |
| `S3_ANSYS_COST` | — | optional Fluent wall time/RAM for the cost table |

Main knobs in `CFG`: `N`, `Nz` (mesh), `ratio` (hidden nodes per equation, default 2), `dt`, `tol_psi` (Newton tolerance, 10⁻⁸), `seed`.

### Outputs

`Fig_centrelines` (vs Ghia et al. 1982), `Fig_vortices_midsection`, `Fig_flow_slices`, `Fig_3D_streamlines`, `Fig_3D_velocity_isometric`, `Fig_training_convergence`, `Fig_KE_MI_CoG`, `Fig_computational_cost`, `phi_mixing.gif`, plus `kinetic_energy.csv`, `mixing_index.csv`, `center_of_gravity.csv`, `flow_training_history.csv`, `computational_cost.txt`, `report.txt` and the final network weights `flow_network_final.pt`. Long runs checkpoint automatically and resume after a Colab disconnect.

## SPIRAL vs. a physics-informed neural network (PINN)

| | PINN | SPIRAL |
|---|---|---|
| Hidden weights | trained | random, frozen |
| Output weights | trained by gradient descent (10⁴–10⁶ iterations) | solved by Newton (a few iterations per time step) |
| Boundary conditions | penalty terms in the loss | written into the discrete equations |
| Continuity | penalty term, approximate | exact (curl of a vector potential) |
| Result vs random seed | varies | identical to 10⁻⁹ |
| Accuracy controlled by | network size, loss weights, optimiser | the mesh, like CFD |

SPIRAL belongs to the family of extreme-learning-machine / random-feature PDE solvers (e.g. PIELM, local ELM, the random feature method); its distinctive pieces are the vector-potential output with an exact discrete curl, the pressure-free curl-momentum residual, and the kernel-form Newton solve for 3-D transient Navier–Stokes.

## Limitations (honest)

- The kernel matrix `K = SGSᵀ` is **dense**: memory grows as n_eq² and the Cholesky factor as n_eq³. On a 15 GB T4, 30×30×3 runs in seconds per step, 40×40×4 needs a float32 factor, 50×50×5 (1 mm cells) is borderline. Production-size meshes need a sparse or domain-decomposed formulation.
- Accuracy is the accuracy of the second-order discretisation: at Re = 1000 the meshes above are still mesh-limited.
- Validated so far on the lid-driven cavity (Ghia et al. 1982, ANSYS Fluent reference) and scalar mixing in it.

## Repository

```
ldc3d_spiral.py            full 3-D solver: flow + scalar + figures + cost report
examples/spiral_1d_hand.py 1-D hand-calculation example
docs/                      figures used in this README
requirements.txt
```

## Citation

```bibtex
@mastersthesis{spiral3d,
  title  = {SPIRAL-3D: a physics-solved random-feature network for the 3-D lid-driven cavity and passive-scalar mixing},
  author = {<Your Name>},
  school = {Johannes Kepler University Linz},
  year   = {2026}
}
```

## References

- U. Ghia, K. N. Ghia, C. T. Shin, *High-Re solutions for incompressible flow using the Navier–Stokes equations and a multigrid method*, J. Comput. Phys. 48 (1982) 387–411.
- M. Raissi, P. Perdikaris, G. E. Karniadakis, *Physics-informed neural networks*, J. Comput. Phys. 378 (2019) 686–707.
- S. Dong, Z. Li, *Local extreme learning machines and domain decomposition for solving linear and nonlinear PDEs*, CMAME 387 (2021) 114129.

---

<div align="center"><sub>Built at Johannes Kepler University Linz · questions and issues welcome</sub></div>
