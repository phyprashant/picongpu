.. _usage-plugins-atomicEnergy:

Atomic Energy
-------------

This species plugin exposes energy reservoirs that are not included in the
particle- and field-energy plugins. It is available for atomic-physics ion
species.

Configuration
^^^^^^^^^^^^^

For an ion species named ``N``, enable the diagnostic with::

   --N_atomicEnergyExchange.period 100

Replace ``N`` with the name of the atomic ion species. A period of ``100``
writes steps 0, 100, 200, and so on.

Process-resolved exchange
^^^^^^^^^^^^^^^^^^^^^^^^^

``N_atomic_energy_exchange_by_process.dat`` records the cumulative signed
atomic-energy change caused by each accepted transition. Upward transitions
are positive and downward transitions are negative. Bound-free processes use
the same dynamic IPD-shifted transition energy used by the solver, so this is
the only atomic quantity commensurate with the PIC energy diagnostics.

Besides one column per process the file carries

``externalReconciliation_J``
   internal energy moved by a charge-state change made *outside* atomicPhysics
   that left the atomic state index valid, i.e. one that changed
   ``boundElectrons`` alone. ``FixAtomicState`` reconciles it by moving the ion
   to the ground state of its new charge state and books the jump here, using
   unperturbed table energies because the module that ionized the ion applied
   no IPD either. No module in the current tree does this, so the column is
   zero in practice; it exists for one that would.

``externalIonizationWeight``
   macro-ion weight that arrived from an ionization module outside
   atomicPhysics. This is what the separate field-ionization modules actually
   do: BSI, ADK, Keldysh and ThomasFermi all go through ``SetChargeState``,
   which invalidates the atomic state index because it cannot reach the atomic
   data needed to choose a consistent state. The ion's previous atomic state is
   destroyed at that moment, so the internal energy it carried **cannot be
   recovered** by ``FixAtomicState`` and is not booked anywhere.

   .. warning::

      A non-zero value here means every energy column in this file is
      incomplete by an unknown amount, and neither ``total_atomic_exchange_J``
      nor ``internalEnergy_J`` may be used in an energy budget. The plugin also
      writes a warning to stderr the first time it happens. The column is
      exactly zero when no such module is active on the species.

``total_atomic_exchange_J``
   the sum of the process columns and ``externalReconciliation_J``.

``internalEnergy_J``
   the absolute atomic internal energy. The reference energy is evaluated once
   at the first output step and advanced from there by the ledger; on restart
   the value is recovered from the checkpointed file rather than re-anchored,
   which would step the curve by the reservoir accrued so far. **This** is the
   column to put in an energy budget.

``referenceEnergy_J``
   the unperturbed reference energy of the current ion population. Its excess
   over ``internalEnergy_J`` measures the accumulated continuum-lowering
   reservoir. This is a diagnostic comparison value, not an energy-budget
   term.

Escaped radiation
^^^^^^^^^^^^^^^^^

``N_radiated_energy.dat`` contains cumulative positive escaped photon energy
from spontaneous de-excitation and radiative recombination. PIConGPU does not
create photon particles for these channels. The radiative-recombination term
uses the selected electron-histogram bin energy plus the IPD-shifted released
binding energy, matching the atomic-physics model's bin-level description.

Energy budget
^^^^^^^^^^^^^

For an evolving-temperature simulation without an external driver, the budget
that closes is

.. math::

   E_{\mathrm{e,kin}} + E_{\mathrm{i,kin}} + E_{E+B}
   + U_{\mathrm{internal}} + E_{\mathrm{radiated}},

with :math:`U_{\mathrm{internal}}` taken from ``internalEnergy_J``. This
requires ``externalIonizationWeight`` to be zero; if it is not, the ledger is
incomplete and no budget can be formed from it.

.. warning::

   Substituting the reference energy for :math:`U_{\mathrm{internal}}` does
   not close, and the error is large rather than marginal: in Ar\ :sup:`2+` at
   100 eV with recombination enabled the reference energy rises about 60 %
   more than the ledger, turning a 0.4 % closure into 13 %. The gap is the
   continuum-lowering reservoir, which the reference energy implicitly
   includes and the solver never charged to the electrons.

Note that closure is bounded from below by the bare PIC energy drift of the
setup, which should be measured before any residual is attributed to the atomic
model. Measure it by removing the ``atomicPhysicsParticle<>`` flags from the
species in :ref:`speciesDefinition.param <usage-params-core>`, which is what
makes ``atomicPhysicsActive`` false and compiles the stage out.

.. warning::

   Setting the process switches in ``atomicPhysics.param`` to ``false`` is
   *not* equivalent and does not give a bare-PIC baseline. The switches decide
   only which transition types exist; whether the stage runs at all is gated on
   the species flags. With every switch false the full sub-step sequence still
   executes, and ``DecelerateElectrons`` still rescales the momentum of every
   in-range electron and stores the result back into a ``float_X`` attribute.
   That writeback is not the identity even when the bin's energy change is
   zero, so a small error accumulates once per sub-step for the whole run. In a
   lithium test case it produced +0.06 % of electron kinetic energy over 13 fs
   where the flags-removed control gave -0.01 %. Revisions that also evaluated
   the energy update itself in ``float_X``, rather than only the final
   momentum writeback, were far worse: +7.6 % on the same setup.
