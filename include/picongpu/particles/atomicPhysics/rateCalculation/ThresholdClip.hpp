/* Copyright 2026 Prashant Sharma
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

/** @file threshold-clipped bin integration for collisional rate calculation
 *
 * The electron histogram bins are wide compared to sharp transition thresholds
 * (log-spaced bins are ~60 eV wide near a 500 eV threshold). The default rate
 * integration samples the cross section once at the bin center over the full bin
 * width. For the single bin straddling a transition threshold deltaE this is wrong
 * in one of two ways:
 *  - bin center below deltaE: the entire bin contributes zero, although the
 *    sub-interval [deltaE, binMax) is above threshold, or
 *  - bin center above deltaE: the entire bin width contributes, although the
 *    sub-interval [binMin, deltaE) is below threshold.
 *
 * The threshold clip repositions the sample of the straddling bin to the center of
 * the valid sub-interval [deltaE, binMax) and uses its width (binMax - deltaE),
 * still one cross-section evaluation. Bins entirely below threshold return zero
 * without a cross-section evaluation.
 *
 * The electron histogram itself (the physical, possibly non-Maxwellian electron
 * distribution) is unchanged; only the in-bin placement of the analytic
 * cross-section sample changes.
 *
 * Switch, overridable at configure time:
 *   0 =^= previous behavior, bin-center sample over full bin width (default)
 *   1 =^= threshold-clipped bin integration with the full-bin mean density
 *   2 =^= threshold clip plus linear density reconstruction at the clipped midpoint
 *
 *   pic-configure -c "-DPARAM_OVERWRITES:LIST=\"-DPARAM_ATOMIC_PHYSICS_RATE_THRESHOLD_CLIP=2\"" ...
 *
 * @attention the unit-test reference values in debug/TestRateCalculation.hpp and the
 *  python reference module assume the default (0); run RUN_UNIT_TESTS builds with the
 *  switch off.
 * @attention applied inside the shared rate functions, therefore consistently used by
 *  both the FillRateCache_* and ChooseTransition_* kernels and by the 3BR EII bin sum.
 * @attention with the clip enabled, the threshold-straddling bin can be selected for a
 *  transition although its bin-center energy lies below deltaE. The bin-mean energy
 *  bookkeeping in DecelerateElectrons may then drive the per-electron energy slightly
 *  negative (bounded by half a bin width per event); this is handled by the existing
 *  newEnergyElectron < 0 clamp in DecelerateElectrons.kernel.
 */

#pragma once

#include "picongpu/defines.hpp"

#ifndef PARAM_ATOMIC_PHYSICS_RATE_THRESHOLD_CLIP
#    define PARAM_ATOMIC_PHYSICS_RATE_THRESHOLD_CLIP 0
#endif

namespace picongpu::particles::atomicPhysics::rateCalculation
{
    static_assert(
        PARAM_ATOMIC_PHYSICS_RATE_THRESHOLD_CLIP >= 0 && PARAM_ATOMIC_PHYSICS_RATE_THRESHOLD_CLIP <= 2,
        "PARAM_ATOMIC_PHYSICS_RATE_THRESHOLD_CLIP must be 0, 1, or 2");

    //! compile-time switches, see file description
    constexpr bool useThresholdClippedBinIntegration = (PARAM_ATOMIC_PHYSICS_RATE_THRESHOLD_CLIP != 0);
    constexpr bool useThresholdClipDensityReconstruction = (PARAM_ATOMIC_PHYSICS_RATE_THRESHOLD_CLIP == 2);

    /** clip an electron histogram bin against a transition threshold energy
     *
     * For the bin straddling the threshold, repositions the bin sample to the center
     * of the above-threshold sub-interval and shrinks the width accordingly. Bins
     * entirely above the threshold are left unchanged.
     *
     * @param energyElectron[in,out] center energy of the electron bin, [eV]
     * @param energyElectronBinWidth[in,out] energy width of the electron bin, [eV]
     * @param energyThreshold threshold energy (deltaE) of the transition, [eV]
     *
     * @return false if the bin lies entirely below the threshold (rate contribution
     *  is exactly zero, cross-section evaluation may be skipped), true otherwise
     */
    HDINLINE bool thresholdClipBin(
        // eV
        float_X& energyElectron,
        // eV
        float_X& energyElectronBinWidth,
        // eV
        float_X const energyThreshold)
    {
        // eV
        float_X const binMax = energyElectron + energyElectronBinWidth / 2._X;

        // entire bin below threshold, no contribution
        if(binMax <= energyThreshold)
            return false;

        // eV
        float_X const binMin = energyElectron - energyElectronBinWidth / 2._X;

        // bin straddles the threshold, restrict to the above-threshold sub-interval
        if(binMin < energyThreshold)
        {
            energyElectron = (energyThreshold + binMax) / 2._X;
            energyElectronBinWidth = binMax - energyThreshold;
        }

        return true;
    }

    /** clip a threshold bin and reconstruct its density at the shifted sample
     *
     * Mode 2 uses a forward finite-difference slope from the next histogram bin.
     * For a linear in-bin density, its value at the clipped midpoint is also the
     * exact mean density over the clipped interval. Modes 0 and 1 leave the
     * density unchanged.
     *
     * @param densityElectron[in,out] electron density represented by the sample
     * @param densitySlopeToNextBin forward density slope from the histogram cache
     */
    HDINLINE bool thresholdClipBin(
        float_X& energyElectron,
        float_X& energyElectronBinWidth,
        float_X const energyThreshold,
        float_X& densityElectron,
        float_X const densitySlopeToNextBin)
    {
        float_X const originalEnergy = energyElectron;
        float_X const originalBinWidth = energyElectronBinWidth;

        bool const contributes = thresholdClipBin(energyElectron, energyElectronBinWidth, energyThreshold);
        if(!contributes)
            return false;

        if constexpr(useThresholdClipDensityReconstruction)
        {
            bool const binWasClipped = energyElectronBinWidth < originalBinWidth;
            if(binWasClipped)
            {
                densityElectron += densitySlopeToNextBin * (energyElectron - originalEnergy);
                if(densityElectron < 0._X)
                    densityElectron = 0._X;
            }
        }

        return true;
    }
} // namespace picongpu::particles::atomicPhysics::rateCalculation
