/* Copyright 2026 Prashant Sharma
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

//! @file kernel summing the unperturbed table energy stored in atomic ion states

#pragma once

#include "picongpu/defines.hpp"

#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/lockstep.hpp>
#include <pmacc/memory/shared/Allocate.hpp>
#include <pmacc/particles/algorithm/ForEach.hpp>

#include <cstdint>

namespace picongpu
{
    /** Sum unperturbed ionization and excitation energies for an atomic ion species.
     *
     * Atomic-state energies are relative to the ground state of the respective charge
     * state.  The absolute reference used here is the neutral ground state, therefore
     * the ground-state ionization potentials below the particle's charge state must be
     * added.  Values accumulated by the kernel are weighted eV.
     *
     * @attention no dynamic IPD shift is applied, the input tables are used as given. The result is therefore a
     *  reference energy, NOT a quantity that may be differenced in time and compared against the PIC energy
     *  diagnostics; see the AtomicEnergyExchange ledger for that. AtomicEnergyExchange uses this kernel to report
     *  the current reference value and to anchor its cumulative ledger to an absolute internal energy.
     */
    struct KernelAtomicInternalEnergy
    {
        template<
            typename T_Worker,
            typename T_ParBox,
            typename T_DBox,
            typename T_Mapping,
            typename T_AtomicStateDataBox,
            typename T_ChargeStateDataBox>
        DINLINE void operator()(
            T_Worker const& worker,
            T_ParBox particleBox,
            T_DBox globalEnergy,
            T_Mapping mapper,
            T_AtomicStateDataBox const atomicStateBox,
            T_ChargeStateDataBox const chargeStateBox) const
        {
            PMACC_SMEM(worker, sharedIonizationEnergy, float_64);
            PMACC_SMEM(worker, sharedExcitationEnergy, float_64);

            auto masterOnly = lockstep::makeMaster(worker);
            masterOnly(
                [&]()
                {
                    sharedIonizationEnergy = 0.0;
                    sharedExcitationEnergy = 0.0;
                });
            worker.sync();

            DataSpace<simDim> const superCellIdx(mapper.getSuperCellIndex(worker.blockDomIdxND()));
            auto forEachParticle
                = pmacc::particles::algorithm::acc::makeForEach(worker, particleBox, superCellIdx);
            if(!forEachParticle.hasParticles())
                return;

            float_64 localIonizationEnergy = 0.0;
            float_64 localExcitationEnergy = 0.0;

            forEachParticle(
                [&atomicStateBox,
                 &chargeStateBox,
                 &localIonizationEnergy,
                 &localExcitationEnergy](auto const&, auto& particle)
                {
                    uint32_t const stateIndex = particle[atomicStateCollectionIndex_];
                    auto const configNumber = atomicStateBox.configNumber(stateIndex);
                    uint8_t const chargeState = T_AtomicStateDataBox::ConfigNumber::getChargeState(configNumber);
                    float_64 const weighting = static_cast<float_64>(particle[weighting_]);

                    float_64 ionizationEnergy = 0.0;
                    for(uint8_t q = 0u; q < chargeState; ++q)
                        ionizationEnergy += static_cast<float_64>(chargeStateBox.ionizationEnergy(q));

                    localIonizationEnergy += weighting * ionizationEnergy;
                    localExcitationEnergy
                        += weighting * static_cast<float_64>(atomicStateBox.energy(stateIndex));
                });

            alpaka::atomicAdd(
                worker.getAcc(),
                &sharedIonizationEnergy,
                localIonizationEnergy,
                ::alpaka::hierarchy::Threads{});
            alpaka::atomicAdd(
                worker.getAcc(),
                &sharedExcitationEnergy,
                localExcitationEnergy,
                ::alpaka::hierarchy::Threads{});
            worker.sync();

            masterOnly(
                [&]()
                {
                    alpaka::atomicAdd(
                        worker.getAcc(),
                        &globalEnergy[0],
                        sharedIonizationEnergy,
                        ::alpaka::hierarchy::Blocks{});
                    alpaka::atomicAdd(
                        worker.getAcc(),
                        &globalEnergy[1],
                        sharedExcitationEnergy,
                        ::alpaka::hierarchy::Blocks{});
                });
        }
    };
} // namespace picongpu
