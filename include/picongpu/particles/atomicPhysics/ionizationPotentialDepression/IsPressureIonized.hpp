/* Copyright 2025 Brian Marre
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

#include "picongpu/particles/atomicPhysics/DeltaEnergyTransition.hpp"
#include "picongpu/particles/atomicPhysics/enums/ProcessClassGroup.hpp"

namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression
{
    namespace s_enums = picongpu::particles::atomicPhysics::enums;

    /** is the atomic state unbound at the given ionization potential depression?
     *
     * A state whose IPD-ionization state is energetically at or below it no longer exists as a bound state at
     *  the local plasma conditions; ApplyIPDIonization will remove any ion found in it at the end of the
     *  atomicPhysics step.
     *
     * @attention must stay in sync with the `stateIsUnbound` criterion in
     *  ionizationPotentialDepression/kernel/ApplyIPDIonization.kernel, otherwise transitions may be suppressed
     *  into states that are not actually removed, or vice versa.
     *
     * @note the barrier suppression contribution of a local E-field is deliberately not included here, since it
     *  is a per-cell rather than a per-superCell quantity. Callers therefore under- rather than over-suppress.
     *
     * @param stateCollectionIndex collection index of the atomic state to test
     * @param ionizationPotentialDepression IPD of that state's charge state, in eV
     */
    template<typename T_ChargeStateDataBox, typename T_AtomicStateDataBox, typename T_IPDIonizationStateDataBox>
    HDINLINE bool isPressureIonized(
        uint32_t const stateCollectionIndex,
        float_X const ionizationPotentialDepression,
        T_ChargeStateDataBox const chargeStateBox,
        T_AtomicStateDataBox const atomicStateBox,
        T_IPDIonizationStateDataBox const ipdIonizationStateBox)
    {
        auto const ipdIonizationStateCollectionIndex = ipdIonizationStateBox.ipdIonizationState(stateCollectionIndex);

        // fully ionized states have no IPD-ionization path and are always bound
        if(ipdIonizationStateCollectionIndex == stateCollectionIndex)
            return false;

        uint8_t const stateChargeState
            = T_AtomicStateDataBox::ConfigNumber::getChargeState(atomicStateBox.configNumber(stateCollectionIndex));
        uint8_t const ipdIonizationStateChargeState = T_AtomicStateDataBox::ConfigNumber::getChargeState(
            atomicStateBox.configNumber(ipdIonizationStateCollectionIndex));

        // eV
        float_X const ionizationEnergyToIPDIonizationState
            = DeltaEnergyTransition::template ionizationEnergy<s_enums::ProcessClassGroup::boundFreeBased, float_X>(
                  stateChargeState,
                  ipdIonizationStateChargeState,
                  ionizationPotentialDepression,
                  chargeStateBox)
            + atomicStateBox.energy(ipdIonizationStateCollectionIndex)
            - atomicStateBox.energy(stateCollectionIndex);

        return (ionizationEnergyToIPDIonizationState <= 0._X);
    }

    /** should this upward bound-bound transition be suppressed as already counted by collisional ionization?
     *
     * Collisional ionization is evaluated with the IPD-shifted threshold, i.e. it already integrates over every
     *  energy transfer able to free the electron at the local plasma conditions. An excitation into a state that
     *  is itself pressure-ionized frees the same electron by the same collision at an energy transfer inside
     *  that same window, and ApplyIPDIonization then completes it. Keeping both counts the collision twice and
     *  inflates the return flux out of the recombined state.
     *
     * @attention only meaningful for upward transitions; the upper state of a bound-bound transition shares the
     *  charge state of the lower state, so the caller's IPD is valid for both.
     */
    template<
        typename T_ChargeStateDataBox,
        typename T_AtomicStateDataBox,
        typename T_IPDIonizationStateDataBox,
        typename T_BoundBoundTransitionDataBox>
    HDINLINE bool upperStateIsPressureIonized(
        uint32_t const transitionCollectionIndex,
        float_X const ionizationPotentialDepression,
        T_ChargeStateDataBox const chargeStateBox,
        T_AtomicStateDataBox const atomicStateBox,
        T_IPDIonizationStateDataBox const ipdIonizationStateBox,
        T_BoundBoundTransitionDataBox const transitionBox)
    {
        return isPressureIonized(
            transitionBox.upperStateCollectionIndex(transitionCollectionIndex),
            ionizationPotentialDepression,
            chargeStateBox,
            atomicStateBox,
            ipdIonizationStateBox);
    }
} // namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression
