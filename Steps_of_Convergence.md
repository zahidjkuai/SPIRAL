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
    <td><img alt="domain and inputs" src="https://github.com/user-attachments/assets/d1ab75ca-8fbf-40ff-9835-ac4b9553055b" /></td>
    <td><img alt="ghost cells" src="https://github.com/user-attachments/assets/d1a8356d-e763-4c13-b545-e699e5cd6190" /></td>
  </tr>
  <tr>
    <td align="center"><b>Step 1 · Only inner points are inputs</b></td>
    <td align="center"><b>Trailer for Step 4 · Walls enter as "ghost" values</b></td>
  </tr>
</table>

> **The story so far:** the network is only handed one thing — the coordinates of points *inside* the cavity (left).
> The walls? They never appear as inputs. They reappear later as mirrored "ghost" values (right) that smuggle
> the lid speed and no-slip condition straight into the physics check — enforced exactly, never learned.

---

## Step 2 · Hidden layer: the frozen random features

### The twist: nobody trains these weights

<img alt="step2" src="https://github.com/user-attachments/assets/8ae6c8f8-2b28-4ebc-b9a0-f9de80db4ed9" />

*Figure 3 — One node's whole life: dice → a random line in the cavity → distance → the tanh "step" → one number into a row.*

- **wₖ and bₖ are rolled like dice once** (fixed seed) and **frozen forever**. Each of the 5,953 nodes
  gets its own random draw — and that is the entire "training" of the hidden layer.
- Each draw places a **random straight line** through the cavity: s = wₖ·x\* + bₖ tells
  *which side of the line, and how far* our point x\* is.
- tanh(s) is **"the step"**: deep on the − side → ≈ −1, deep on the + side → ≈ +1, on the line → 0.
  The slope |wₖ| decides how sharp the gradient between the two worlds is.
- Repeat for all 5,953 nodes and their answers fill one row — **that row is everything the hidden
  layer will ever say about x\*.**

### And the whole layer collapses into one table

<table>
  <tr>
    <td><img alt="one matrix multiply" src="https://github.com/user-attachments/assets/cb89440a-e0d1-4481-bde3-3afc3abdf465" /></td>
    <td><img alt="the finished table Phi" src="https://github.com/user-attachments/assets/fc21a2fb-d41a-4500-a30c-9f15863d4a88" /></td>
  </tr>
  <tr>
    <td align="center"><b>How it's made: one matrix multiply</b></td>
    <td align="center"><b>What you get: the finished table Φ</b></td>
  </tr>
</table>

*Figure 4 — 53 million numbers, and not one of them learned: Φ = tanh(XW + b), computed once at startup, stored forever.*

> **The story so far:** we handed the network only coordinates (Step 1). Out of pure randomness it built
> a dictionary of 5,953 smooth "steps" — every possible way to slice the cavity in two. Asking the whole
> dictionary about every point takes **a single matrix multiply**. The hidden layer's job is already done,
> and no gradient ever touched it. The only question left: *how much of each step do we need?* — that is z,
> and Step 3 is where it enters.

---

## Step 3 · Output: the weighted sum — where z takes the stage

<img alt="step3" src="https://github.com/user-attachments/assets/02acea18-b392-418e-8643-f8b1198fb790" />

*Figure 5 — One row, one set of weights, one sum: 5,953 ready numbers become the physical field A — and the only unknowns z finally appear.*

- Take the feature row Φ(x\*) from Step 2, multiply each entry by its own weight zₖ, and
  **add everything**: A(x\*) = Σ zₖ aₖ. That single sum is the stream-function-like potential at x\*.
- **z is the ONLY unknown in the entire method** — 3 × P = 17,859 numbers. We never chose them,
  trained them, or guessed them: Step 5 will solve for them.
- Do the same sum at every point and the **whole flow field A(x) emerges** — the vortex you see
  in the figure is literally built out of frozen random steps, mixed by z.
- The velocity is **not** another network output: it comes from a discrete curl, u = ∇ × A.
  Consequence: **∇·u = 0 exactly, for any z, always**. Incompressibility is baked into the
  architecture — no pressure equation, no penalty term, no constraint to enforce.

> **The story so far:** coordinates in (Step 1), a frozen dictionary of random steps (Step 2),
> and now a recipe — "take this much of each step" — written in 17,859 numbers called z.
> Any z already gives a divergence-free flow. But a *divergence-free* flow is not yet a *correct* flow:
> with the wrong z, the physics books don't balance. Checking whether they do is Step 4's job —
> and it has a verdict ready: **r = −0.18 ≠ 0. Wrong.**

---

## Step 4 · Residual: the physics check

<img alt="step4" src="https://github.com/user-attachments/assets/4f04c39e-8f1c-4816-affb-6b02b5884e59" />

*Figure 6 — The courtroom scene: three physical terms must balance to zero at every point. With the
first-guess z they don't (r = −0.178, guilty). After Newton they do (r = 1.9×10⁻¹¹, free to go).*

We take the flow field from Step 3 and ask Navier–Stokes one question, at **every point**:

**r = [ uⁿᵉʷ − uʰⁱˢᵗᵒʳʸ ] / Δt   +   (u·∇)u   −   (1/Re) ∇²u   ≟   0**

| term | name | what it asks |
|---|---|---|
| [uⁿᵉʷ − uʰⁱˢᵗᵒʳʸ] / Δt | **time** | is the flow speeding up or slowing down here since the last time step? |
| (u·∇)u | **convection** | the momentum fluid carries along with itself — the boss term in fast regions, and the only nonlinear one |
| − (1/Re) ∇²u | **viscous** | friction ironing out velocity differences between neighbors — computed from a 5-point stencil around the point |

*(Why no pressure? We evaluate the curl of the momentum equation — pressure has zero curl, so it simply vanishes from the courtroom.)*

- **Time:** is the flow speeding up or slowing down here since the last time step?
- **Convection:** the momentum fluid carries along with itself — the boss term in fast regions,
  and the only nonlinear one.
- **Viscous:** friction ironing out velocity differences between neighbors — computed from a
  5-point stencil around the point.

(Why no pressure? We evaluate the *curl* of the momentum equation — pressure has zero curl,
so it simply vanishes from the courtroom.)

### Ghost cells — how the walls sneak into the math

- Near a wall, that 5-point stencil asks for a value **outside the domain**. The wall answers
  through a **mirror rule**: moving lid → u_ghost = −u_inside + 2U; fixed wall → u_ghost = −u_inside.
  In both cases the average of inside and ghost equals the wall speed **exactly**.
- These ghost values are **computed by the rule, never learned** — and they are the *only* way
  boundary conditions enter the entire solver. No BC rows in the input, no penalty terms.
- Subtle and important: as z changes during Newton, u_inside changes, so **the ghost values change
  with it** — the boundary conditions are rebuilt inside r at every single iteration.

### The verdict

With the first-guess z (extrapolated from previous time steps): time −0.105, convection −0.560,
viscous +0.488 → **r = −0.178 ≠ 0. Physics violated.** After Newton: +0.025 − 0.416 + 0.390 →
**r = 1.9×10⁻¹¹ ≈ 0. Physics satisfied.** Repeat this audit at all 8,930 points, and r becomes a
*vector* of 8,930 accusations. The target: acquit them all.

> **The story so far:** we can generate any divergence-free flow with a z (Step 3), and now we have
> a lie detector that says exactly *how wrong* that flow is at every point — including at the walls,
> which testify through their ghost values. The only thing missing is a judge that corrects z until
> every accusation reads zero. That judge has a 400-year-old name: **Newton. Step 5.**

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

