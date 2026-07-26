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

#include <pmacc/algorithms/math.hpp>
#include <pmacc/memory/Array.hpp>

#include <cstdint>

/** @file bin-center density reconstruction for collisional rate integration
 *
 * The histogram stores, per bin, the bin-MEAN differential electron density
 *   rhoBar_i = 1/h * integral over bin of rho(E) dE.
 * The midpoint rate integration however samples the cross section at the bin center
 * and wants the density AT that center, rho(c). For a flat spectrum the two agree, but
 * the log-spaced bins are ~16% wide, so for a steeply falling spectrum (h/T_e of order
 * one) rhoBar over-weights the cold lower bin edge and the rate is over-predicted.
 *
 * Modelling the density locally as an exponential, rho(E) = rho(c) * exp(k*(E-c)),
 * the stored bin mean is
 *   rhoBar = rho(c) * sinh(x)/x,   x = k*h/2,
 * so the center density follows from the stored mean by the scalar factor
 *   f = x/sinh(x) <= 1.
 * The decay constant k is estimated from the neighbouring bin mean,
 *   k = ln(rhoBar_{i+1}/rhoBar_i) / (c_{i+1} - c_i).
 *
 * f is a PER-BIN scalar evaluated once in fill(), so the correction costs nothing in the
 * (bin x atomicState x transition) rate loops. It is a local two-bin fit, not a global
 * Maxwellian assumption, and is positivity preserving by construction.
 *
 * @attention only the rate integration uses the reconstructed density. electronDensity(),
 *  temperatureEnergy() and the capture-bin selection in ChooseTransition_* need true
 *  particle numbers and keep using the stored bin means.
 *
 * Switch, overridable at configure time:
 *   0 =^= previous behavior, bin-mean density used as the bin-center density (default)
 *   1 =^= exponential bin-center density reconstruction
 *
 *   pic-configure -c "-DPARAM_OVERWRITES:LIST=\"-DPARAM_ATOMIC_PHYSICS_RATE_CENTER_DENSITY=1\"" ...
 *
 * @attention the unit-test reference values in debug/TestRateCalculation.hpp assume the
 *  default (0); run RUN_UNIT_TESTS builds with the switch off.
 */

#ifndef PARAM_ATOMIC_PHYSICS_RATE_CENTER_DENSITY
#    define PARAM_ATOMIC_PHYSICS_RATE_CENTER_DENSITY 0
#endif

namespace picongpu::particles::atomicPhysics::kernel
{
    static_assert(
        PARAM_ATOMIC_PHYSICS_RATE_CENTER_DENSITY == 0 || PARAM_ATOMIC_PHYSICS_RATE_CENTER_DENSITY == 1,
        "PARAM_ATOMIC_PHYSICS_RATE_CENTER_DENSITY must be 0 or 1");

    //! compile-time switch, see file description
    constexpr bool useCenterDensityReconstruction = (PARAM_ATOMIC_PHYSICS_RATE_CENTER_DENSITY != 0);

    template<uint32_t T_size>
    struct CachedHistogram
    {
        pmacc::memory::Array<float_X, T_size> energy;
        pmacc::memory::Array<float_X, T_size> binWidth;
        //! bin-mean differential density, the physical electron content of the bin
        pmacc::memory::Array<float_X, T_size> density;
        //! reconstructed bin-center differential density, used by the rate integration only
        pmacc::memory::Array<float_X, T_size> rateDensity;
        /** electron number density above the histogram range, [1/sim.unit.length()^3]
         *
         * the overflow bin stores the weight of all electrons with energy >= maxEnergy but not their energy,
         *  it can therefore contribute to a number density but not to an energy average
         */
        float_X overflowDensity;
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

                    // single writer, the overflow bin is a scalar and not part of the regular bin range
                    if(idx == 0u)
                        // 1/sim.unit.length()^3
                        overflowDensity = electronHistogram.getOverflowWeight() / volumeScalingFactor;
                });
            // the bin-center reconstruction reads the neighbouring bin, needs all means written
            worker.sync();

            forEachElement([&](uint32_t const idx) { rateDensity[idx] = density[idx] * centerDensityFactor(idx); });
            worker.sync();
        }

        /** exponential bin-center density reconstruction factor
         *
         * ratio of the bin-center density to the stored bin-mean density, see file
         * description. Falls back to 1 (no correction) for the last bin, for empty bins
         * and for a non-positive energy spacing.
         *
         * @param idx regular histogram-bin index
         * @return unitless, in (0, 1]
         */
        HDINLINE float_X centerDensityFactor(uint32_t const idx) const
        {
            if constexpr(!useCenterDensityReconstruction)
                return 1._X;
            else
            {
                if(idx + 1u >= T_size)
                    return 1._X;

                float_X const densityThisBin = density[idx];
                float_X const densityNextBin = density[idx + 1u];
                // an empty bin gives no usable decay constant
                if((densityThisBin <= 0._X) || (densityNextBin <= 0._X))
                    return 1._X;

                // eV
                float_X const energyDifference = energy[idx + 1u] - energy[idx];
                if(energyDifference <= 0._X)
                    return 1._X;

                /* limits the correction for sparsely populated bins, where the log-ratio of two
                 * macro-particle estimates is dominated by sampling noise, to f >= 3/sinh(3) ~ 0.3 */
                constexpr float_X maxHalfBinDecay = 3._X;

                // 1/eV, local exponential decay constant of the electron spectrum
                float_X const decayConstant = math::log(densityNextBin / densityThisBin) / energyDifference;
                // unitless, half a bin measured in decay lengths
                float_X const x = pmacc::math::max(
                    -maxHalfBinDecay,
                    pmacc::math::min(maxHalfBinDecay, decayConstant * binWidth[idx] / 2._X));

                // x/sinh(x), series expansion near zero avoids the removable 0/0
                if(x * x < 1.e-6_X)
                    return 1._X - x * x / 6._X;

                return 2._X * x / (math::exp(x) - math::exp(-x));
            }
        }

        /** finite-difference density slope towards the next higher-energy bin
         *
         * Used by threshold-bin integration to reconstruct the electron density
         * at a sample shifted above the original bin center. The last bin has no
         * upper neighbour and therefore falls back to a constant density.
         *
         * @attention uses the reconstructed bin-center densities, so that a clip-shifted
         *  sample composes correctly with the bin-center reconstruction. Identical to the
         *  bin-mean slope when the reconstruction is switched off.
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

            return (rateDensity[idx + 1u] - rateDensity[idx]) / energyDifference;
        }

        /** total electron number density of the histogram
         *
         * includes the overflow bin, so that the three-body recombination detailed balance factor sees the
         *  complete electron population acting as the third body
         *
         * @return unit: 1/sim.unit.length()^3
         */
        HDINLINE float_X electronDensity() const
        {
            float_X result = overflowDensity;
            for(uint32_t idx = 0u; idx < T_size; ++idx)
                result += density[idx] * binWidth[idx];
            return result;
        }

        /** effective electron temperature of the histogram as k_B * T
         *
         * classical ideal gas estimator, T = 2/3 * <E_kin>, valid for non-relativistic electron spectra only
         *
         * @attention does not include the overflow bin, which stores no energy information. A significant
         *  overflow population therefore biases the estimate low, unlike electronDensity(), which does account
         *  for it. Both are only meaningful for a spectrum that the histogram range actually resolves.
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
