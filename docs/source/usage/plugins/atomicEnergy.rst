.. _usage-plugins-atomicEnergy:

Atomic Energy
-------------

These species plugins expose energy reservoirs that are not included in the
particle- and field-energy plugins. They are available for atomic-physics ion
species.

Configuration
^^^^^^^^^^^^^

For an ion species named ``N``, enable both diagnostics with::

   --N_atomicInternalEnergy.period 100 \
   --N_atomicEnergyExchange.period 100

Replace ``N`` with the name of the atomic ion species. A period of ``100``
writes steps 0, 100, 200, and so on.

Absolute atomic internal energy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``N_atomic_internal_energy.dat`` contains::

   #step ionization_J excitation_J total_atomic_J

For each macro ion, the plugin maps ``atomicStateCollectionIndex`` to the
configured atomic state and sums

.. math::

   w\left(\sum_{q=0}^{Z_i-1} I_q + E_{\mathrm{exc},i}\right),

where :math:`w` is the macro-particle weighting, :math:`I_q` are the
ground-state ionization potentials, and :math:`E_{\mathrm{exc},i}` is the
configuration energy relative to the ground state of that charge state.
The neutral ground state is the zero of energy.

The result uses the unperturbed input-table energies. Dynamic ionization
potential depression (IPD) is not included.

Process-resolved exchange
^^^^^^^^^^^^^^^^^^^^^^^^^

``N_atomic_energy_exchange_by_process.dat`` records the cumulative signed
atomic-energy change caused by each accepted transition. Upward transitions
are positive and downward transitions are negative. Bound-free processes use
the same dynamic IPD-shifted transition energy used by the solver.

Thus, with IPD disabled, the sum of the process columns should agree with
``total_atomic_J(t) - total_atomic_J(0)``. With IPD enabled, their difference
measures the accumulated continuum-lowering contribution; the absolute and
event-based quantities intentionally use different energy definitions.

Escaped radiation
^^^^^^^^^^^^^^^^^

``N_radiated_energy.dat`` contains cumulative positive escaped photon energy
from spontaneous de-excitation and radiative recombination. PIConGPU does not
create photon particles for these channels. The radiative-recombination term
uses the selected electron-histogram bin energy plus the IPD-shifted released
binding energy, matching the atomic-physics model's bin-level description.

Energy budget
^^^^^^^^^^^^^

For an evolving-temperature simulation without an external driver, a useful
unperturbed budget is

.. math::

   E_{\mathrm{e,kin}} + E_{\mathrm{i,kin}} + E_{E+B}
   + U_{\mathrm{atomic}} + E_{\mathrm{radiated}}.

When dynamic IPD is enabled, strict thermodynamic closure additionally needs
a consistent continuum-lowering reservoir. The process ledger provides the
solver-side IPD-shifted exchange needed to diagnose that difference, but does
not by itself define an equilibrium IPD free energy.

