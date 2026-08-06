# Project brief: semiclassical network model for moire magnetic Bloch bands

**Status: nothing here is implemented.**  This is a plan, written 2026-08-06
out of a design conversation.  No code in this repo does any of it.  Nothing
below describes existing behaviour except where it explicitly points at a
file that exists today.

Audience: whoever picks this up next.  Read sections 1 and 2 before anything
else — section 1 is the *reason*, and it should decide design arguments that
section 2 alone would leave open.

---

## 1. The goal, and why it is not "a better number"

Computing the magnetic Bloch bands (MBBs) at flux `1/q` is already easy here:
`main_v3.py` does it exactly, and the semiclassical Hofstadter mode
(`hofstadter_system.py` + `unfold.py`) gives them with `Oz`/`Lz` on a mesh.
Accuracy is not the problem.

The problem is that those bands are **un-intuitive**.  They fall out of a
diagonalization in a Landau-level basis, with Laguerre `F_nm` matrix elements
and an `LL_multiplier`/`Nmax` truncation.  You never see a Fermi contour, you
never see a gap.  The MBB is a black box.

The user's stated epistemics, which drive this whole project: **the zero-field
band structure is ground-level truth.**  A thing is "understood" when it is
explained in terms of zero-field quantities.  MBBs at present are not.

So the deliverable is an **explanatory reduction**: derive the MBB from

- the **bare dispersion** (which `extended_mode = centroid` already returns
  *identically* — it is `u_b^dag H_jj u_b`, the moire-free bilayer),
- the **moire gap at each Bragg plane** (already stored as `gap_K`), i.e. the
  Fourier components of the moire potential,
- the **velocities at each crossing** (from the same bare dispersion),
- and **one number**: the Aharonov-Bohm phase per moire plaquette, `2*pi/q`.

Nothing else.  No LL basis, no Laguerre functions, no truncation cutoff.

**Design implication, and it matters:** where a choice trades transparency for
accuracy, take transparency.  The exact answer already exists and is cheap at
moderate `q`.  This project is worthless if it becomes a second black box.

**Why the regime is not a niche.**  The construction needs a weak moire
potential and large orbits.  The user's observation, which is the strongest
part of the pitch: systems whose moire potential comes from **interlayer
hybridization** rather than a strong substrate potential are in that regime by
construction — small gaps, large orbits.  That is a class of systems, not one
material.  It is also exactly where exact diagonalization is most expensive,
because low field means large `q` means large matrices.

---

## 2. The physics

### The reframing

The current `breakdown.py` picture is: an LL is a closed orbit, breakdown makes
it leaky, so the level smears by `Gamma`.  Leakage as damage.

The network picture inverts it.  Breakdown is **hopping between orbits**: at a
Bragg plane the carrier continues onto the orbit centred one moire cell over.
So there is a *lattice of semiclassical orbits*, with hopping amplitude set by
the breakdown amplitude, and the LL on each site hybridizes into a band exactly
as an atomic level does in tight binding.  The bandwidth is the hopping — which
is why the incoherent `sum sqrt(1-P)` in `breakdown.py` gets the scale right at
all (median 0.83 against exact).

Then the phase: going around one plaquette of that orbit lattice picks up an AB
phase `2*pi * (flux per moire cell)/phi_0`.  A tight-binding lattice with a
flux-dependent phase per plaquette **is Harper's equation**.  So the splitting
of a bare LL into subbands is not a separate phenomenon — it is Hofstadter,
derived from the orbit lattice instead of the atomic lattice.

This makes the butterfly's `phi -> 1/phi` self-similarity manifest: the
strong-field limit is moire Bloch bands broadened into LLs, the weak-field
limit is LLs broadened into orbit-network bands, and they are the same equation
with the magnetic length and the moire period exchanged.

### The calculation

Network eigenproblem, a secular condition

```
det[ M(E, kappa) - 1 ] = 0
```

`M` = product of **arc propagators** (semiclassical phase along each arc
between junctions) and **junction scattering matrices** (2x2 Landau-Zener).
`kappa` is a Bloch quasimomentum conjugate to the orbit-centre lattice.
Solving at each `kappa` gives `E(kappa)` over the magnetic BZ — the MBB
dispersion, with bandwidth as a byproduct rather than as the output.

Berry curvature comes from the eigenvectors (the arc amplitudes): a vector of
dimension (arcs per magnetic cell), so `A = i<psi|d_kappa psi>` and
`Omega = curl A` on a tiny matrix.  Wrinkle: it is a *nonlinear* eigenproblem
(arc phases depend on `E`), so the Berry connection needs left/right
eigenvectors with a `dM/dE` normalization.  Known technique; Chern numbers of
scattering networks are well-trodden (Chalker-Coddington is the canonical
example).

### Two rungs, and what each one gives

| Rung | Method | Output |
|---|---|---|
| 1 | network at flux `1/q` | full band structure: `E(kappa)`, `Omega(kappa)`, bandwidth, Chern number |
| 2 | Onsager on the MBB with `delta B` (`onsager_bfield`) | level **positions** only — *not* widths |

Level-2 widths require running the network a *second* time, on the MBB
dispersion with the MBB's own gaps.  That `onsager_bfield` currently rejects
`breakdown = 1` is exactly this missing piece, not an oversight.

**Each rung resolves one digit of the continued-fraction expansion of the
flux.**  That is Azbel's bookkeeping and the precise sense in which the
butterfly is self-similar.  Rung 2 is valid only for `delta phi` small enough
that many level-2 orbits fit inside the MBB — you are resolving structure
*locally around* `1/q`, not globally.  Each rung zooms in by one denominator.

The validity condition is evaluable with numbers already in hand: rung 2 needs
`hbar*omega_c^(2) << W_MBB`, and `W_MBB` has been measured at 60-100% of
`hbar*omega_c^(1)`.

**Is a third rung worth wanting?**  Probably not, and it is settleable with
existing tooling: level-3 structure lives inside a fraction of `W_MBB`, and
whether that survives disorder is a question `main_v3`'s SCBA already answers
quantitatively.  Compare the level-3 scale to the SCBA broadening at the
relevant fields.  Expectation (untested) is that two rungs covers everything
experimentally meaningful — which would make "two rungs" the whole story
rather than a truncation.

### One thing NOT to claim

It is tempting to say the MBB is the zero-field moire band structure continued
in flux.  **It is not.**  Arc phases go as `(hbar/eB)^2 * (k-area)`, so the
network is singular at `B -> 0` and does not reduce to the zero-field bands.

What is genuinely shared is the **scattering data**, not the propagator:

- zero-field moire bands = bare bands + Bragg scattering with *free* propagation
  (the nearly-free-electron construction);
- MBBs = bare bands + *the same* Bragg scattering, with *cyclotron* propagation
  and an AB phase.

Same junctions, different connector.  That is still the tie-back to zero field
the project is for, and it is the defensible form of it.

---

## 3. What already exists in this repo

The geometry machinery is largely built.  This is the main reason the project
is tractable.

| Asset | What it gives the network |
|---|---|
| `extended_zone.py`, `extended_mode = centroid` | the **bare dispersion**, identically (`u_b^dag H_jj u_b`).  Already the right decomposition: moire potential quarantined out of `E` |
| `gap_K` / `gap_Kp` (meV) | the **moire gap at each Bragg plane**, `2*abs(E_dominant - E_centroid)`, a ridge of height `Eg` on the plane |
| `breakdown.py: orbit_breakdown_fields` | **junction locations** — finds and dedups the Bragg crossings, computes `B0` at each.  Solved code, reuse directly |
| `breakdown.py: reciprocal_shells` | the Bragg-plane test that works.  A Wigner-Seitz partition was tried and is wrong; see `doc_technical.md` |
| `isoenergy.py`, `return_contours=True` | the **contours**, hence the arcs (segments between consecutive crossings) |
| `isoenergy.py: enclosedBC` | **loop Berry flux** from `Oz` — see the gauge trap in section 5 |
| `isoenergy_areas` | `A_ext` and `A_fold` on the same energy grid (both tabulated in the extended-zone section of `doc_technical.md`) |
| `onsager_bfield` | the rung-2 solver, already written.  `Blist` is `delta B` from the background flux baked into the bands |
| `main_v3.py` | **exact ground truth** at any `qq/pp`.  This is what makes every stage testable in isolation |
| `main_v3.py` SCBA | disorder broadening, for the "is there a third rung" question |

### The validation dataset (guard this)

`semiclassical/_exactwidth.json` (4.9 KB) holds **105 exact Landau-level widths**
from `main_v3` at `qq = 1`, and `_breakdown.json` the matching semiclassical
side.  Parameters: `pp = 12 ... 2`, `nk = 16`, window `[-170, -100]` meV,
`moire_0.5_Um20_Em0.5`, spanning 1.93-11.55 T.

At `qq = 1` the exact subband width **is** the MBB bandwidth — that identity is
what made the `breakdown.py` validation possible, and it makes this dataset the
ground truth for the network's `E(kappa)` too.  Stages 0 and 1 below depend on
it.

`_proto_exactwidth.py` extracts it; `_proto_compare.py` does the level-by-level
comparison.  **These files are untracked** (not gitignored — just never added).
Regenerating them costs real compute.  Commit them before doing anything else.

Diagnostics from that run, useful as sanity checks: 12 crossings at 114 of 120
levels; crossing turn-on between `A/A_BZ = 0.881` and `0.887` (circle-in-hexagon
value is `pi/(2*sqrt(3)) = 0.9069`); `m_c = 0.059-0.074 m_e`; `B0` median 0.93 T,
range 0.095-11.95 T.

---

## 4. The plan

Ground truth exists at every stage.  Do not reorder — stage 1 exists to
de-risk the single hardest-to-debug ingredient before any topology code is
written.

### Stage 0 — interference period from zero-field areas

**Cost: days.  No new machinery.  Data in hand.**

Under breakdown the classic signature is **magnetic interference**: combination
frequencies from linear combinations of the constituent orbit areas — the
transmit-everywhere orbit (`A_ext`) and the reflect-everywhere orbit
(`A_fold`).  This is the Falicov-Stachowiak Mg result, and it is the natural
candidate for the ~4x level-to-level oscillation `breakdown.py` cannot
reproduce.

Sketch of the phase: paths differ by `(hbar/eB) * dA` with `dA = A_ext - A_fold`,
and level spacing is set by `dA_ext/dE`, so levels per oscillation cycle:

```
N ~ (dA_ext/dE) / (d(dA)/dE)
```

Both areas are zero-field contour areas `isoenergy` already computes on the same
grid.  Test `N` against the measured oscillation in `_exactwidth.json`.

**Confidence: the prefactor is a quick sketch, treat it as a scaling argument.
The structure — period set by an area difference — is well established.**

Two known snags: the folded areas in the crossing-active range are the
pocket-switching artifacts documented in the extended-zone section of
`doc_technical.md`, so identifying the correct reflected orbit needs that
bookkeeping; and the oscillation is level-to-level at fixed `B` (i.e. in `E`),
not in `1/B`, so the standard dHvA framing needs translating.

**Pass criterion:** measured period matches `N`.  If it does, the MBB fine
structure is demonstrably a function of zero-field contour geometry and the
explanatory claim is established with no new code.  If it does not, find out
why *before* committing to the rest.

### Stage 1 — two-orbit Pippard network

**Cost: ~a week.  This is the de-risking step.  Do not skip it.**

Closed form, 2x2.  Predicts the oscillation's **amplitude** as well as its
period, and — the real point — **tests the Stokes phase convention** against
the 105-level dataset before any network topology code exists.

**Pass criterion:** reproduces the level-to-level oscillation in
`_exactwidth.json`, not just its envelope.  `breakdown.py` gets the envelope
(median 0.83) and the rank correlation within a field is near zero at low `B`;
beating that is the whole point.

### Stage 2 — full 2D network at flux `1/q`

The real thing.  Outputs `E(kappa)`, `Omega(kappa)` on a magnetic-BZ mesh.

**Pass criteria:**
- bandwidth matches `main_v3` **level by level**, not as an envelope;
- MBB Chern number satisfies the Diophantine relation against the parent band's
  Chern number.  The parent Chern numbers are already validated against
  `main_v3` sigma_xy plateaus (bands 5-11), so this closes independently of the
  widths.

### Stage 3 — feed rung 2

Emit `E_K`, `Oz_K`, `Lz_K`, `vol_M`, `_kmesh` in the standard format and hand to
`isoenergy` + `onsager_bfield`.

The recursion closes because the network stage's **output type equals its input
type**.  Stages read the mesh from the data via `_kmesh` and `load_data` unwraps
uniformly, so a network stage emitting those keys would be consumed unchanged.
Rung 3 is then the same code again.

**Pass criterion:** the rung-2 fan matches `main_v3` near flux `1/q`.

---

## 5. Traps

Difficulty assessment from the design conversation:

| Piece | Difficulty | Note |
|---|---|---|
| Locate junctions | **done** | reuse `orbit_breakdown_fields` |
| Identify arcs | trivial | contour segments between crossings |
| Solve `det[M-1]=0` | easy | track eigenvalues of `M` on the unit circle as `E` sweeps.  `M` is ~`12q x 12q`, so ~120x120 at `q=10` — nothing |
| Arc propagator phase | **medium** | gauge-dependent per arc; only loops are invariant |
| Junction Stokes phase | **medium, high silent-error risk** | 2x2 LZ connection matrix, the `arg Gamma(1-i*delta)` piece.  Conventions differ between sources by factors inside `delta`, and a wrong one shifts every gap without visibly breaking anything |
| Network topology / magnetic cell | **tedious, not deep** | which arc leaves which junction into which neighbouring orbit; `q` plaquettes per magnetic cell.  This is where the time actually goes |
| MBB Berry curvature | medium | nonlinear eigenproblem, left/right eigenvectors, `dM/dE` normalization |

The linear algebra is a non-issue.  The physics inputs are already computed.
What the project buys is bookkeeping.

### The gauge trap, and the design decision that removes it

**Do not compute Berry phases along arcs.**  They are gauge-dependent
(eigenvector phase), and chasing a smooth gauge along a contour is precisely
the class of thing that produced the documented Berry-curvature sign problem in
`onsager.py` (see "Term signs" in `doc_technical.md`).

Formulate in **loop** quantities instead: every independent loop of the network
encloses a region, and the Berry phase around it is the flux of `Oz` through
it — which `enclosedBC` already computes, just for different loops.

Same trick for the dynamical phase: distribute arc phases as triangle areas
from a chosen origin, so individual arc phases are convention-dependent but
every loop sum is the enclosed area regardless.

**Cheap decisive self-test:** verify `E(kappa)` is invariant under moving the
origin.

### The `Bmultiplier = 4` prerequisite, which is also an opportunity

`onsager_Bmultiplier` defaults to **4**, not 1, and nobody has traced where the
factor belongs — it rides on the rhs rather than being folded into `S(E)`.  See
"The Bmultiplier factor" in `doc_technical.md` and the
`project_onsager_lambda4_origin` memory.

The network uses the **same area normalization** in its arc phases, so it will
either inherit that factor or expose it.  Because the network builds the total
phase as a sum over arcs from first principles rather than quantizing a closed
area, a factor-of-4 error in the normalization has nowhere to hide.

**Treat "does the network reproduce `lambda = 1` naturally" as an early
checkpoint.**  If it does not, something upstream is wrong and should be fixed
before the results are trusted.

### Where the explanation runs out

The network is asymptotic.  Where the moire potential is strong or orbits are
small relative to the moire BZ, no zero-field account of this kind exists and
the MBB really is an irreducibly mixed object.  Say so rather than extrapolating.

---

## 6. Confidence register

Do not treat this document as established fact.  Explicitly:

**Established (measured, in the repo):**
- `Gamma` from `breakdown.py` matches exact widths with median 0.83 over 105
  levels, 1.93-11.55 T; it is an envelope, rank correlation near zero at low `B`.
- The exact widths oscillate ~4x level to level.
- One LL = `qq` magnetic subbands; at `qq = 1` subband width = MBB bandwidth.
- 12 crossings, turn-on at `A/A_BZ ~ 0.88`.
- `extended_mode = centroid` returns the bare dispersion identically.

**Standard physics, high confidence, not re-derived here:**
- Breakdown as orbit hopping -> Harper -> Hofstadter.
- Magnetic interference producing combination frequencies.
- The network secular condition and its solution by transfer matrix.
- Chern numbers of scattering networks.

**Sketches from conversation — verify before relying on:**
- `N ~ (dA_ext/dE)/(d(dA)/dE)` for the oscillation period.  Prefactor unchecked.
- The claim that the network exposes the `lambda = 4` factor.  Plausible
  mechanism, untested.
- The triangle-area gauge-fixing construction.  Standard in spirit, not
  verified in this geometry.
- "Two rungs covers everything that survives disorder."  Never computed;
  the SCBA comparison that would settle it has not been run.

**Not checked at all:**
- Whether anyone has already done exactly this for a moire system.  A literature
  search is stage -1.  Wilkinson's RG is the closest published thing found, and
  it appears genuinely under-cited relative to the Harper/Hofstadter numerics,
  but "nobody has done this" was not verified.

---

## 7. References

All bibliographic details below were verified against Crossref on 2026-08-06,
except where noted.  The annotated list mapping each breakdown-specific citation
to the piece of `breakdown.py` it underwrites is in the "Magnetic breakdown
broadening" section of `doc_technical.md`; this list is the *network* subset.

**The blueprint.**  Wilkinson derived the butterfly's self-similarity from
semiclassical tunnelling between orbits.  This is the closest published thing
to the whole project:

- M. Wilkinson, *An exact renormalisation group for Bloch electrons in a
  magnetic field*, J. Phys. A **20**, 4337 (1987).
- M. Wilkinson, *Critical properties of electron eigenstates in incommensurate
  systems*, Proc. R. Soc. Lond. A **391**, 305 (1984).
- M. Wilkinson, *Von Neumann lattices of Wannier functions for Bloch electrons
  in a magnetic field*, Proc. R. Soc. Lond. A **403**, 135 (1986).  The
  orbit-lattice basis made explicit.

**Network machinery:**

- A. B. Pippard, *Quantization of coupled orbits in metals*, Proc. R. Soc.
  Lond. A **270**, 1 (1962); *...II. The two-dimensional network, with special
  reference to the properties of zinc*, Phil. Trans. R. Soc. Lond. A **256**,
  317 (1964).  Paper II is the closest published setting to this geometry.
  **Stage 1 comes from paper I.**
- L. M. Falicov and H. Stachowiak, Phys. Rev. **147**, 505 (1966).  Coupled
  orbits and the combination frequencies stage 0 tests for.
- W. G. Chambers, *Linear-Network Model for Magnetic Breakdown in Two
  Dimensions*, Phys. Rev. **140**, A135 (1965).  The 1D warm-up.
- R. W. Stark and L. M. Falicov, *Magnetic Breakdown in Metals*, Prog. Low
  Temp. Phys. **5**, 235 (1967).  Review; best single entry point.

**Modern semiclassics with Berry phase, orbital moment and breakdown together
— i.e. what `onsager.py` and the network would do jointly:**

- A. Alexandradinata and L. Glazman, Phys. Rev. B **97**, 144422 (2018).

**Context:**

- D. R. Hofstadter, Phys. Rev. B **14**, 2239 (1976).  Worth rereading for the
  citation of Azbel, *Sov. Phys. JETP* **19**, 634 (1964), which is the original
  quasiclassical route to the continued-fraction structure and predates the
  numerics.  **Azbel details unverified** — Soviet-era JETP is patchy in
  Crossref.
- M. I. Kaganov and A. A. Slutskin, *Coherent magnetic breakdown*, Phys. Rep.
  (~vol. 98, 1983).  **Unverified**, a lead to check.
- E. I. Blount, Phys. Rev. **126**, 1636 (1962).  Source of the two-velocity
  `B0` form the code uses.
- J. R. Wallbank *et al.*, Phys. Rev. B **87**, 245408 (2013).  The miniband
  model.
- R. Krishna Kumar *et al.*, Science **357**, 181 (2017).  Breakdown measured
  in a graphene superlattice; the Brown-Zak framing.

---

## 8. Open questions

1. Has this been done for a moire system already?  Stage -1.
2. Does stage 0's period test pass?  Everything downstream is contingent on it.
3. Does the network reproduce `lambda = 1`, or inherit the 4?
4. Is there a third rung above disorder?  Answerable now with the SCBA.
5. Rung-2 widths need a second network pass.  Worth it, or is rung 2 at
   positions-only resolution enough?

## 9. Standing constraints

- Commit directly to `main`; no feature branches, no PRs.
- Any `.py` change requires the corresponding doc update
  (`doc_technical.md` / `doc_user_guide.md` / `CLAUDE.md`).
- Read the doc files before the source.
- Never bias an implementation toward the expected physics.  Report surprises
  as they are — the `lambda = 4` and Berry-sign histories in this repo are both
  cases where that mattered.
