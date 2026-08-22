/* Copyright 2026 Prashant Sharma
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/atomicPhysics/ConvertEnum.hpp"
#include "picongpu/particles/atomicPhysics/enums/ProcessClass.hpp"

#include <cstdint>

namespace picongpu::particles::atomicPhysics::localHelperFields
{
    /** Cumulative atomic and escaped-radiation energy, separated by atomic process.
     *
     * All entries are macro-particle weighted eV. Atomic energy is signed: upward
     * transitions add energy and downward transitions remove it. Radiation energy
     * is positive energy that has left the simulated particle/field system.
     */
    class AtomicEnergyExchange
    {
    public:
        static constexpr uint32_t numberProcesses = 10u;

        template<enums::ProcessClass T_ProcessClass, typename T_Worker>
        HDINLINE void addAtomicEnergy(T_Worker const& worker, float_64 const energy)
        {
            alpaka::atomicAdd(
                worker.getAcc(),
                &atomicEnergy[u8(T_ProcessClass)],
                energy,
                ::alpaka::hierarchy::Threads{});
        }

        template<enums::ProcessClass T_ProcessClass, typename T_Worker>
        HDINLINE void addRadiatedEnergy(T_Worker const& worker, float_64 const energy)
        {
            alpaka::atomicAdd(
                worker.getAcc(),
                &radiatedEnergy[u8(T_ProcessClass)],
                energy,
                ::alpaka::hierarchy::Threads{});
        }

        /** add an already block-wide reduced contribution
         *
         * @attention must be called by exactly one worker of the block, see the atomicPhysics kernels for the
         *  accumulate-per-thread, reduce-once-per-block pattern. Calling the per-thread variants above once per
         *  macro ion serialises the kernel on a single global memory address.
         */
        //!@{
        template<enums::ProcessClass T_ProcessClass, typename T_Worker>
        HDINLINE void addAtomicEnergyBlockWide(T_Worker const& worker, float_64 const energy)
        {
            alpaka::atomicAdd(
                worker.getAcc(),
                &atomicEnergy[u8(T_ProcessClass)],
                energy,
                ::alpaka::hierarchy::Blocks{});
        }

        template<enums::ProcessClass T_ProcessClass, typename T_Worker>
        HDINLINE void addRadiatedEnergyBlockWide(T_Worker const& worker, float_64 const energy)
        {
            alpaka::atomicAdd(
                worker.getAcc(),
                &radiatedEnergy[u8(T_ProcessClass)],
                energy,
                ::alpaka::hierarchy::Blocks{});
        }
        //!@}

        /** book an internal energy jump made outside the atomicPhysics transition machinery
         *
         * Charge state changes applied by other modules, field ionization in particular, reach atomicPhysics only as
         * a changed boundElectrons attribute. FixAtomicState reconciles them by moving the ion to the ground state of
         * its new charge state, which changes internal energy without any transition being chosen or accepted, so no
         * ProcessClass applies. Kept separate from the per-process array for that reason.
         *
         * @attention unperturbed table energies, no IPD shift: the module that ionized the ion did not apply one
         *  either. See addReconciliationEnergyBlockWide callers.
         * @attention must be called by exactly one worker of the block, see addAtomicEnergyBlockWide
         */
        template<typename T_Worker>
        HDINLINE void addReconciliationEnergyBlockWide(T_Worker const& worker, float_64 const energy)
        {
            alpaka::atomicAdd(worker.getAcc(), &reconciliationEnergy, energy, ::alpaka::hierarchy::Blocks{});
        }

        /** count macro ion weight arriving from an ionization module outside atomicPhysics
         *
         * Their previous atomic state is destroyed by SetChargeState before atomicPhysics ever sees them, so the
         * internal energy they carried cannot be booked. A non-zero count means the ledger is incomplete by an
         * unknown amount, which the plugin reports; see FixAtomicStateKernel.
         *
         * @attention must be called by exactly one worker of the block, see addAtomicEnergyBlockWide
         */
        template<typename T_Worker>
        HDINLINE void addExternalIonizationWeightBlockWide(T_Worker const& worker, float_64 const weight)
        {
            alpaka::atomicAdd(worker.getAcc(), &externalIonizationWeight, weight, ::alpaka::hierarchy::Blocks{});
        }

        HDINLINE float_64 getExternalIonizationWeight() const
        {
            return externalIonizationWeight;
        }

        HDINLINE float_64 getAtomicEnergy(uint32_t const process) const
        {
            return atomicEnergy[process];
        }

        HDINLINE float_64 getRadiatedEnergy(uint32_t const process) const
        {
            return radiatedEnergy[process];
        }

        HDINLINE float_64 getReconciliationEnergy() const
        {
            return reconciliationEnergy;
        }

    private:
        float_64 atomicEnergy[numberProcesses] = {0.0};
        float_64 radiatedEnergy[numberProcesses] = {0.0};
        //! weighted eV, internal energy moved by charge state changes made outside atomicPhysics
        float_64 reconciliationEnergy = 0.0;
        //! weight of macro ions reconciled after an external ionization, whose energy could not be booked
        float_64 externalIonizationWeight = 0.0;
    };
} // namespace picongpu::particles::atomicPhysics::localHelperFields
