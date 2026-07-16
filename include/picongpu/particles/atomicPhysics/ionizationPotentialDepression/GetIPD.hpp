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

// need simulation.param for normalisation and units, memory.param for SuperCellSize and dim.param for simDim
#include "picongpu/defines.hpp"

namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression
{
    // get IPD from IPD model
    template<typename T_IPDModel, typename T_AtomicStateDataDataBox>
    HDINLINE float_X getIPD(
        T_AtomicStateDataDataBox atomicStateBox,
        uint32_t const stateCollectionIndex,
        typename T_IPDModel::SuperCellConstantInput const superCellConstantIPDInput)
    {
        auto const stateConfigNumber = atomicStateBox.configNumber(stateCollectionIndex);
        uint8_t const stateChargeState = T_AtomicStateDataDataBox::ConfigNumber::getChargeState(stateConfigNumber);

        // eV
        return T_IPDModel::ipd(superCellConstantIPDInput, stateChargeState);
    }

    /** get IPD of the lower(recombined) state of a bound-free transition
     *
     * The IPD depends on the state's charge state. Upward bound-free processes compute their IPD-shifted
     *  threshold from the lower state they start from, therefore downward bound-free processes(recombination)
     *  must use the IPD of the transition's lower state, not of their current(upper) state, for the same
     *  threshold convention on the same transition.
     *
     * @attention must be evaluated per transition, different downward transitions of the same upper state may
     *  have different lower states
     */
    template<typename T_IPDModel, typename T_AtomicStateDataDataBox, typename T_BoundFreeTransitionDataBox>
    HDINLINE float_X getBoundFreeLowerStateIPD(
        T_AtomicStateDataDataBox const atomicStateBox,
        T_BoundFreeTransitionDataBox const transitionBox,
        uint32_t const transitionCollectionIndex,
        typename T_IPDModel::SuperCellConstantInput const superCellConstantIPDInput)
    {
        return getIPD<T_IPDModel>(
            atomicStateBox,
            transitionBox.lowerStateCollectionIndex(transitionCollectionIndex),
            superCellConstantIPDInput);
    }
} // namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression
