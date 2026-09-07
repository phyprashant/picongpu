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
#include "picongpu/particles/atomicPhysics/debug/ElectronCapture.hpp"
#include "picongpu/particles/atomicPhysics/debug/param.hpp"

#include <cstdint>
#include <limits>

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

#if PARAM_CHECK_ELECTRON_CAPTURE_BALANCE == 1
        //! weight actually removed from free electron macro particles, unitless
        float_X removedWeight[numberBins] = {0._X};
        //! capturedWeight in excess of the bin's weight0, lost to the captureFraction clamp, unitless
        float_X clampedWeight[numberBins] = {0._X};
        //! weight discarded by MIN_WEIGHTING macro particle deletion beyond the intended capture, unitless
        float_X deletionResidual[numberBins] = {0._X};
        //! failed reservation attempts, including requests too small to represent
        uint32_t rejectedReservations[numberBins] = {};
        //! invalid-index calls across all bins
        uint32_t invalidBinIndexCalls = 0u;
#endif

        /** @attention only active by debug setting
         *
         * Active under either BIN_INDEX_RANGE_CHECK or CHECK_WEIGHT_BALANCE: the latter's invalid-index counter
         *  (see getInvalidBinIndexCalls()) would otherwise never see an out-of-range call unless the unrelated
         *  BIN_INDEX_RANGE_CHECK switch is also enabled.
         */
        HDINLINE static bool outOfRangeBinIndex(uint32_t const binIndex)
        {
            if constexpr(
                picongpu::atomicPhysics::debug::rejectionProbabilityCache::BIN_INDEX_RANGE_CHECK
                || picongpu::atomicPhysics::debug::electronCapture::CHECK_WEIGHT_BALANCE)
                if(binIndex >= numberBins)
                {
                    printf("atomicPhysics ERROR: out of range bin index in call to CapturedWeightCache\n");
                    return true;
                }
            return false;
        }

    public:
        /** reserve captured weight against the weight the histogram bin actually holds
         *
         * The caller must skip the transition if reservation fails.
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
            {
#if PARAM_CHECK_ELECTRON_CAPTURE_BALANCE == 1
                alpaka::atomicAdd(
                    worker.getAcc(),
                    &invalidBinIndexCalls,
                    1u,
                    ::alpaka::hierarchy::Threads{});
#endif
                return false;
            }

            // Reject invalid requests without ever modifying the committed captured weight.
            auto reject = [&]()
            {
#if PARAM_CHECK_ELECTRON_CAPTURE_BALANCE == 1
                alpaka::atomicAdd(
                    worker.getAcc(),
                    &rejectedReservations[binIndex],
                    1u,
                    ::alpaka::hierarchy::Threads{});
#endif
                return false;
            };
            if(!(weight > 0._X) || !(weight <= capacity) || !(capacity <= std::numeric_limits<float_X>::max()))
                return reject();

            // Atomic read: other workers may already be reserving from this bin.
            float_X previous = alpaka::atomicCas(
                worker.getAcc(),
                &capturedWeight[binIndex],
                0._X,
                0._X,
                ::alpaka::hierarchy::Threads{});
            while(true)
            {
                float_X const next = previous + weight;
                // The subtraction check also rejects a request rounded down to capacity.
                // Do not commit an ion transition if its weight disappears in rounding.
                if(!(previous >= 0._X) || !(weight <= capacity - previous) || !(next > previous)
                   || !(next <= capacity))
                    return reject();

                float_X const observed = alpaka::atomicCas(
                    worker.getAcc(),
                    &capturedWeight[binIndex],
                    previous,
                    next,
                    ::alpaka::hierarchy::Threads{});
                if(observed == previous)
                    return true;
                previous = observed;
            }
        }

#if PARAM_CHECK_ELECTRON_CAPTURE_BALANCE == 1
        HDINLINE uint32_t getRejectedReservations(uint32_t const binIndex) const
        {
            return rejectedReservations[binIndex];
        }

        //! @return number of invalid-index calls
        HDINLINE uint32_t getInvalidBinIndexCalls() const
        {
            return invalidBinIndexCalls;
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
            {
                alpaka::atomicAdd(worker.getAcc(), &invalidBinIndexCalls, 1u, ::alpaka::hierarchy::Threads{});
                return;
            }
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
            {
                alpaka::atomicAdd(worker.getAcc(), &invalidBinIndexCalls, 1u, ::alpaka::hierarchy::Threads{});
                return;
            }
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
            {
                alpaka::atomicAdd(worker.getAcc(), &invalidBinIndexCalls, 1u, ::alpaka::hierarchy::Threads{});
                return;
            }
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

#endif

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
