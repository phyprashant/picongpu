/* Copyright 2023-2026 Brian Marre, Prashant Sharma
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/atomicPhysics/debug/param.hpp"

#include <cstdint>

namespace picongpu::particles::atomicPhysics::localHelperFields
{
    /** cache of electron weight captured from each electron histogram bin by recombination processes
     *
     * filled by the RecordChanges sub-stage for each accepted three-body recombination transition, consumed by the
     *  ApplyElectronCapture sub-stage which removes the captured weight from the electron macro particles.
     *
     * @attention shared by all atomicPhysics ion species, kept separate from the electron histogram's deltaWeight
     *  since deltaWeight also contains reservations of non-capturing collisional processes
     *
     * @tparam T_numberBins number of bin entries in cache
     */
    template<uint32_t T_numberBins>
    class CapturedWeightCache
    {
    public:
        static constexpr uint32_t numberBins = T_numberBins;

    private:
        // unitless (macro particle weighting)
        float_X capturedWeight[numberBins] = {0._X};

        //! @attention only active by debug setting
        HDINLINE static bool outOfRangeBinIndex(uint32_t const binIndex)
        {
            if constexpr(picongpu::atomicPhysics::debug::rejectionProbabilityCache::BIN_INDEX_RANGE_CHECK)
                if(binIndex >= numberBins)
                {
                    printf("atomicPhysics ERROR: out of range bin index in call to CapturedWeightCache\n");
                    return true;
                }
            return false;
        }

    public:
        /** add captured weight to cache entry, using atomics
         *
         * @param worker object containing the device and block information
         * @param binIndex index of electron histogram bin the weight is captured from
         * @param weight captured weight, unitless
         *
         * @attention no range checks outside a debug compile, invalid memory write on failure
         */
        template<typename T_Worker>
        HDINLINE void add(T_Worker const& worker, uint32_t const binIndex, float_X const weight)
        {
            if(outOfRangeBinIndex(binIndex))
                return;

            alpaka::atomicAdd(
                worker.getAcc(),
                &(this->capturedWeight[binIndex]),
                weight,
                ::alpaka::hierarchy::Threads{});
        }

        /** get captured weight of a bin
         *
         * @param binIndex index of electron histogram bin
         * @return captured weight, unitless
         *
         * @attention no range checks outside a debug compile, invalid memory access on failure
         */
        HDINLINE float_X getCapturedWeight(uint32_t const binIndex) const
        {
            if(outOfRangeBinIndex(binIndex))
                return 0._X;

            return capturedWeight[binIndex];
        }
    };
} // namespace picongpu::particles::atomicPhysics::localHelperFields
