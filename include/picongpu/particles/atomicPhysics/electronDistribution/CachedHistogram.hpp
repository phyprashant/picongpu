/* Copyright 2023-2024 Rene Widera
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

#include <pmacc/memory/Array.hpp>

#include <cstdint>

namespace picongpu::particles::atomicPhysics::kernel
{
    template<uint32_t T_size>
    struct CachedHistogram
    {
        pmacc::memory::Array<float_X, T_size> energy;
        pmacc::memory::Array<float_X, T_size> binWidth;
        pmacc::memory::Array<float_X, T_size> density;

        static constexpr uint32_t size = T_size;

        constexpr uint32_t numBins() const
        {
            return size;
        }

        /** Fill histogram
         *
         * @attention This method is synchronizing the worker before returning the handle.
         *
         * @tparam T_Worker
         * @tparam T_Histogram
         * @param worker
         * @param electronHistogram
         * @param volumeScalingFactor
         */
        template<typename T_Worker, typename T_Histogram>
        HDINLINE void fill(
            T_Worker const& worker,
            T_Histogram const& electronHistogram,
            float_X const volumeScalingFactor)
        {
            auto forEachElement = lockstep::makeForEach<T_size>(worker);
            forEachElement(
                [&](uint32_t const idx)
                {
                    energy[idx] = electronHistogram.getBinEnergy(idx);
                    // eV
                    float_X const binWithValue = electronHistogram.getBinWidth(idx);
                    binWidth[idx] = binWithValue;
                    // 1/(sim.unit.length()^3 * eV)
                    density[idx] = electronHistogram.getBinWeight0(idx) / volumeScalingFactor / binWithValue;
                });
            worker.sync();
        }

        /** finite-difference density slope towards the next higher-energy bin
         *
         * Used by threshold-bin integration to reconstruct the electron density
         * at a sample shifted above the original bin center. The last bin has no
         * upper neighbour and therefore falls back to a constant density.
         *
         * @param idx regular histogram-bin index
         * @return density slope, [1/(sim.unit.length()^3 * eV^2)]
         */
        HDINLINE float_X densitySlopeToNextBin(uint32_t const idx) const
        {
            if(idx + 1u >= T_size)
                return 0._X;

            float_X const energyDifference = energy[idx + 1u] - energy[idx];
            if(energyDifference <= 0._X)
                return 0._X;

            return (density[idx + 1u] - density[idx]) / energyDifference;
        }

        /** total electron number density of the histogram
         *
         * @attention does not include the overflow bin
         *
         * @return unit: 1/sim.unit.length()^3
         */
        HDINLINE float_X electronDensity() const
        {
            float_X result = 0._X;
            for(uint32_t idx = 0u; idx < T_size; ++idx)
                result += density[idx] * binWidth[idx];
            return result;
        }

        /** effective electron temperature of the histogram as k_B * T
         *
         * classical ideal gas estimator, T = 2/3 * <E_kin>, valid for non-relativistic electron spectra only
         *
         * @attention does not include the overflow bin
         *
         * @return unit: eV, 0 if histogram is empty
         */
        HDINLINE float_X temperatureEnergy() const
        {
            float_X sumWeight = 0._X;
            float_X sumEnergy = 0._X;
            for(uint32_t idx = 0u; idx < T_size; ++idx)
            {
                float_X const binDensity = density[idx] * binWidth[idx];
                sumWeight += binDensity;
                sumEnergy += binDensity * energy[idx];
            }

            if(sumWeight <= 0._X)
                return 0._X;
            return 2._X / 3._X * sumEnergy / sumWeight;
        }
    };


} // namespace picongpu::particles::atomicPhysics::kernel
