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

        HDINLINE float_64 getAtomicEnergy(uint32_t const process) const
        {
            return atomicEnergy[process];
        }

        HDINLINE float_64 getRadiatedEnergy(uint32_t const process) const
        {
            return radiatedEnergy[process];
        }

    private:
        float_64 atomicEnergy[numberProcesses] = {0.0};
        float_64 radiatedEnergy[numberProcesses] = {0.0};
    };
} // namespace picongpu::particles::atomicPhysics::localHelperFields
