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

        /* diagnostic accumulators, only written when
         *  picongpu::atomicPhysics::debug::electronCapture::CHECK_WEIGHT_BALANCE is set */
        //! weight actually removed from free electron macro particles, unitless
        float_X removedWeight[numberBins] = {0._X};
        //! capturedWeight in excess of the bin's weight0, lost to the captureFraction clamp, unitless
        float_X clampedWeight[numberBins] = {0._X};
        //! weight discarded by MIN_WEIGHTING macro particle deletion beyond the intended capture, unitless
        float_X deletionResidual[numberBins] = {0._X};

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

        /** reserve captured weight against the weight the histogram bin actually holds
         *
         * Recombination consumes free electron weight from a bin. The ApplyElectronCapture sub-stage can only ever
         *  remove the weight the bin holds, so a capture that would draw more than that is not realizable: the ion
         *  would be recombined while the corresponding free electron weight stays in the simulation, creating
         *  charge. Reserving here makes the invariant capturedWeight <= binWeight0 hold by construction, so the
         *  caller must skip the transition when this returns false.
         *
         * @param binIndex index of electron histogram bin the weight is captured from
         * @param weight captured weight to reserve, unitless
         * @param capacity weight the bin holds, i.e. its weight0, unitless
         *
         * @return true if the weight was reserved, false if the bin cannot supply it
         */
        template<typename T_Worker>
        HDINLINE bool tryReserve(
            T_Worker const& worker,
            uint32_t const binIndex,
            float_X const weight,
            float_X const capacity)
        {
            if(outOfRangeBinIndex(binIndex))
                return false;

            float_X const previous = alpaka::atomicAdd(
                worker.getAcc(),
                &(this->capturedWeight[binIndex]),
                weight,
                ::alpaka::hierarchy::Threads{});

            if((previous + weight) > capacity)
            {
                // roll back, this bin cannot supply the requested weight
                alpaka::atomicAdd(
                    worker.getAcc(),
                    &(this->capturedWeight[binIndex]),
                    -weight,
                    ::alpaka::hierarchy::Threads{});
                return false;
            }

            return true;
        }

        /** add diagnostic weight to a cache entry, using atomics
         *
         * @param binIndex index of electron histogram bin
         * @param weight weight to add, unitless
         *
         * @attention only called when CHECK_WEIGHT_BALANCE is set
         */
        template<typename T_Worker>
        HDINLINE void addRemovedWeight(T_Worker const& worker, uint32_t const binIndex, float_X const weight)
        {
            if(outOfRangeBinIndex(binIndex))
                return;
            alpaka::atomicAdd(
                worker.getAcc(),
                &(this->removedWeight[binIndex]),
                weight,
                ::alpaka::hierarchy::Threads{});
        }

        //! @copydoc addRemovedWeight
        template<typename T_Worker>
        HDINLINE void addClampedWeight(T_Worker const& worker, uint32_t const binIndex, float_X const weight)
        {
            if(outOfRangeBinIndex(binIndex))
                return;
            alpaka::atomicAdd(
                worker.getAcc(),
                &(this->clampedWeight[binIndex]),
                weight,
                ::alpaka::hierarchy::Threads{});
        }

        //! @copydoc addRemovedWeight
        template<typename T_Worker>
        HDINLINE void addDeletionResidual(T_Worker const& worker, uint32_t const binIndex, float_X const weight)
        {
            if(outOfRangeBinIndex(binIndex))
                return;
            alpaka::atomicAdd(
                worker.getAcc(),
                &(this->deletionResidual[binIndex]),
                weight,
                ::alpaka::hierarchy::Threads{});
        }

        //! @return weight actually removed from free electrons for a bin, unitless
        HDINLINE float_X getRemovedWeight(uint32_t const binIndex) const
        {
            if(outOfRangeBinIndex(binIndex))
                return 0._X;
            return removedWeight[binIndex];
        }

        //! @return weight lost to the captureFraction clamp for a bin, unitless
        HDINLINE float_X getClampedWeight(uint32_t const binIndex) const
        {
            if(outOfRangeBinIndex(binIndex))
                return 0._X;
            return clampedWeight[binIndex];
        }

        //! @return weight lost to MIN_WEIGHTING deletion for a bin, unitless
        HDINLINE float_X getDeletionResidual(uint32_t const binIndex) const
        {
            if(outOfRangeBinIndex(binIndex))
                return 0._X;
            return deletionResidual[binIndex];
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
