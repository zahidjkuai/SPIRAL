# Inside SPIRAL: Opening the Black Box of a Training-Free Physics Solver

> A step-by-step visual walkthrough of how SPIRAL turns a grid point into a physics-satisfying
> flow field — **without training a single network weight.**
> All pictures and numbers are from one real run: lid-driven cavity, Re = 1000, Δt = 0.1 s,
> time step at t = 0.4 s, query point x\* = (0.50, 0.85).

**Contents**

1. [The architecture at a glance](#the-architecture-at-a-glance)
2. [Step 1 · Domain → inputs](#step-1--domain--inputs)
3. [Step 2 · Hidden layer: the frozen random features](#step-2--hidden-layer-the-frozen-random-features)
4. [Step 3 · Output: the weighted sum](#step-3--output-the-weighted-sum)
5. [Step 4 · Residual: the physics check](#step-4--residual-the-physics-check)
6. [Step 5 · Newton: driving the residual to zero](#step-5--newton-driving-the-residual-to-zero)
7. [The whole pipeline in one view](#the-whole-pipeline-in-one-view)
8. [Variable glossary](#variable-glossary)

---

## The architecture at a glance

<img width="4488" height="3543" alt="SPIRAL_architecture" src="https://github.com/user-attachments/assets/9ddab51d-5566-422b-89b5-6847eb89408b" />


*Figure 1 — Overview of the SPIRAL solver.* **(a) Flow network.** Positions of staggered mesh edges,
x = (x, y, z), are mapped through a single hidden layer of frozen tanh nodes (drawn once from a fixed
seed, never updated) to a linear output — the vector potential **A** — parameterized only by the
output weights **z**, with no bias. A fixed discrete-curl operator built from the mesh converts A into
face velocities, so the flow is exactly divergence-free (∇·u = 0 **by construction**). The momentum
residual (convection, diffusion, time term, boundary conditions) is assembled on the edges, and a
Newton solver in kernel form updates z until r = 0 everywhere; then the next time step begins.
**(b) One hidden node.** Each node computes aₖ(x) = tanh(**w**ₖ·x + bₖ): a smooth spatial step whose
steepness and orientation are random, so the frozen layer is a multi-scale dictionary of smooth
features. **(c) Scalar network.** The same construction transports a passive scalar through the
velocity field from (a). *Legend: gray = frozen, never updated; white = the only solved unknowns (z);
hatched = fixed operators with no parameters.*

**The one-sentence summary:** what a normal PINN does with millions of trained weights and gradient
descent, SPIRAL does with *frozen random features + ~15 exact linear solves.*

---

## Step 1 · Domain → inputs

<table>
  <tr>
    <td><img alt="step1" src="https://github.com/user-attachments/assets/d1ab75ca-8fbf-40ff-9835-ac4b9553055b" /></td>
    <td><img alt="step11" src="https://github.com/user-attachments/assets/d1a8356d-e763-4c13-b545-e699e5cd6190" /></td>
  </tr>
  <tr>
    <td align="center"><b>Step 1 · Domain → inputs</b></td>
    <td align="center"><b>Step 2 · Hidden layer</b></td>
  </tr>
</table>


*Figure 2 — The lid-driven cavity and its input matrix.*

- The **only inputs** are the coordinates of interior points: one row per point, x = (x, y, z).
- **Wall points carry no unknowns** — the no-flow-through condition sets A = 0 there, so they are
  not inputs.
- Boundary conditions are **not** inputs either. They enter later, inside the residual, through
  ghost values (see Step 4).

---

## Step 2 · Hidden layer: the frozen random features

### Where do the weights come from?

<img width="1672" height="941" alt="step2" src="https://github.com/user-attachments/assets/8ae6c8f8-2b28-4ebc-b9a0-f9de80db4ed9" />


*Figure 3 — One node, start to finish: random weights → a line in the cavity → signed distance →
tanh “step” → one number → the feature row.*

- **wₖ and bₖ are drawn at random once** (fixed seed) and **frozen forever** — never trained.
  Each of the thousands of nodes has its own random (wₖ, bₖ).
- Geometrically, wₖ and bₖ place a **random straight line** in the cavity; the value
  s = wₖ·x\* + bₖ measures *which side of the line, and how far* the query point x\* is.
- tanh(s) is **“the step”**: far on the − side → ≈ −1, far on the + side → ≈ +1, on the line → 0,
  with a smooth blend whose steepness is set by |wₖ|.
- The input x\* = (x, y, z) is the **position of the query point** — not a property of the node.
  The node is a function; x\* is where we evaluate it.

### The final result of the hidden layer

<table>
  <tr>
    <td><img alt="step22" src="https://github.com/user-attachments/assets/fc21a2fb-d41a-4500-a30c-9f15863d4a88" /></td>
    <td><img alt="step21" src="https://github.com/user-attachments/assets/cb89440a-e0d1-4481-bde3-3afc3abdf465" /></td>
  </tr>
</table>


*Figure 4 — The hidden layer’s entire output: one big finished table Φ.*

Every node asked about every point → **Φ = tanh(XW + b)**, a table of *(number of points) ×
(number of nodes)* entries, **computed once and stored**. Nothing is trained.

![One matrix multiply](images/step2_one_matrix_multiply.png)

*Figure 5 — Why Φ is free: it is a single matrix multiply X × W (+ bias, + tanh), not a loop over
millions of node evaluations.*

---

## Step 3 · Output: the weighted sum

<img width="1672" height="941" alt="step3" src="https://github.com/user-attachments/assets/02acea18-b392-418e-8643-f8b1198fb790" />


*Figure 6 — Mixing the feature row with the unknowns z produces the physical field.*

- Take the row Φ(x\*) from Step 2 and form **A(x\*) = Σ zₖ aₖ** — one scalar per component of the
  vector potential.
- **z is the ONLY unknown** in the whole method (3 × P numbers for the flow network).
- Velocity comes from a **discrete curl, u = C·A**, so **∇·u = 0 exactly for any z** — the output
  is divergence-free by construction, and the pressure never appears.

---

## Step 4 · Residual: the physics check

<img width="1312" height="1199" alt="step4" src="https://github.com/user-attachments/assets/4f04c39e-8f1c-4816-affb-6b02b5884e59" />


*Figure 7 — The residual at one point: time + convection + viscous must sum to zero. Note the ghost
cell sitting above the lid — the mirror rule feeds the wall condition into the viscous term.*

The momentum balance (curl taken, pressure gone), at every point:

```
r  =  [u^new − u^history]/Δt   +   (u·∇)u   −   (1/Re)·∇²u   =  must be ZERO
       └──── time ────┘          └ convection ┘   └── viscous ──┘
```

- **Time:** how much the local velocity changed since the last time step.
- **Convection:** momentum carried by the flow itself (the nonlinear term).
- **Viscous:** friction spreading motion out, computed from the 5-point stencil.

### Ghost cells — how the boundary conditions get in

![Ghost cells explained](images/ghost_cells_correct_deep.png)

*Figure 8 — Wall condition → ghost value → stencil → viscous term → residual.*

- Near a wall the stencil needs a value **outside** the domain. The wall answers through the
  **mirror rule**: moving lid u_ghost = −u_inside + 2U; fixed wall u_ghost = −u_inside, so the
  average of inside and ghost equals the wall speed exactly.
- Ghost values are **computed, never learned** — they are the only way boundary conditions enter
  the solver, and they change together with z at every iteration.

With a first-guess z the three terms do not cancel (r ≠ 0). After Newton they cancel to ~10⁻¹¹.
Repeated at every point, r is a **vector of thousands of numbers** — target: all zero.

---

## Step 5 · Newton: driving the residual to zero

<img width="1448" height="1086" alt="step51" src="https://github.com/user-attachments/assets/5413b1bb-0147-4b5a-8970-3d3878fb4c92" />
<img width="1448" height="1086" alt="step5" src="https://github.com/user-attachments/assets/47621634-2fb4-450c-9897-c620ae354d81" />


*Figure 9 — The Newton computation line by line, with every variable explained.*

One iteration, in plain words:

1. **Compute r(z)** at all points (Step 4).
2. **Never build the Jacobian** J = S·Φ (huge and rectangular). Instead compress it:
   G = ΦΦᵀ, K = S·G·Sᵀ + λI — a small, symmetric, positive-definite system.
3. **Solve Kα = −r** by Cholesky — the one expensive step.
4. **Lift back: Δz = ΦᵀSᵀα**, update z ← z + Δz.
5. Recompute r and repeat — about **15 iterations** thanks to quadratic convergence.

No learning rate, no epochs, no training data — each iteration is one exact linear solve.

### What it looks like inside the domain

<img width="1448" height="1086" alt="step52" src="https://github.com/user-attachments/assets/e2b37fbb-6bb9-4f41-86d7-99ebda0b6faf" />


*Figure 10 — The flow field healing across Newton iterations: the misplaced vortex of the first
guess reorganizes into the correct cavity flow while the residual maps fade to zero.*

The first guess has the vortex misplaced and large residuals along the shear layer under the lid
(exactly where ghost cells feed the lid speed in). Each solve nudges **all z’s at once**, so the
whole field reorganizes together — vortex position, near-lid speed, corner eddies — until the time,
convection and viscous terms cancel **at every point, including the ghost-fed wall stencils.**

---

## The whole pipeline in one view

<img width="1916" height="821" alt="insight" src="https://github.com/user-attachments/assets/a30bf18f-6247-4459-a9a7-08bdb07d6f58" />


*Figure 11 — SPIRAL end to end: grid point → frozen features → weighted sum → residual → Newton.
The “network” is never trained; the physics residual defines the problem; Newton solves it.*

---

## Variable glossary

| Symbol | Meaning |
|---|---|
| **x\*** = (x, y, z) | Position of a query point — the only input |
| **wₖ, bₖ** | Frozen random weight and bias of hidden node k (drawn once, never trained) |
| **aₖ = tanh(wₖ·x + bₖ)** | Output of node k at point x — a smooth spatial step in (−1, +1) |
| **Φ = tanh(XW + b)** | Feature table: all nodes × all points, computed once |
| **A** | Vector potential; the network’s linear output, A(x) = Φ(x)·z |
| **z** | The output weights — the **only unknowns** of the method |
| **u = C·A** | Velocity from the discrete curl C → ∇·u = 0 exactly |
| **r** | Residual: momentum-balance violation at each point (target: all zero) |
| **S** | Sparse stencil matrix ∂r/∂A — built from the 5-point stencil + ghost rules |
| **G = ΦΦᵀ** | Point-to-point similarity through the features |
| **K = S·G·Sᵀ + λI** | Small symmetric positive-definite kernel matrix actually solved |
| **λ** | Tiny regularizer keeping K numerically safe |
| **α** | Helper vector of the kernel solve (no direct physical meaning) |
| **Δz** | Newton correction to the output weights |
| **Jacobian J = S·Φ** | Full sensitivity — exists mathematically, never stored |
| **h, Δt, Re** | Grid spacing, time step (0.1 s), Reynolds number (1000) |

---

*Generated as supplementary documentation for the SPIRAL code. Every figure traces the same run:
lid-driven cavity, Re = 1000, Δt = 0.1 s, time step at t = 0.4 s, query point x\* = (0.50, 0.85).*

