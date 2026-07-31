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

/** @file cap collisional bound-free rates sub-stage of atomicPhysics
 *
 * bounds the collisional bound-free contribution to each atomic state's total loss rate by the instant
 *  transition rate limit while leaving the equilibrium of the sub-network unchanged,
 *  see CollisionalBoundFreePairCap
 *
 * @attention must run after fillRateCache and before calculateStepLength
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/atomicPhysics/CollisionalBoundFreePairCap.hpp"
#include "picongpu/particles/atomicPhysics/electronDistribution/LocalHistogramField.hpp"
#include "picongpu/particles/atomicPhysics/enums/TransitionOrdering.hpp"
#include "picongpu/particles/atomicPhysics/kernel/CapCollisionalBoundFreeRates.kernel"
#include "picongpu/particles/atomicPhysics/localHelperFields/RateCacheField.hpp"
#include "picongpu/particles/atomicPhysics/localHelperFields/TimeRemainingField.hpp"
#include "picongpu/particles/param.hpp"
#include "picongpu/particles/traits/GetAtomicDataType.hpp"
#include "picongpu/particles/traits/GetNumberAtomicStates.hpp"

#include <pmacc/Environment.hpp>
#include <pmacc/particles/meta/FindByNameOrType.hpp>

#include <cstdint>
#include <string>

namespace picongpu::particles::atomicPhysics::stage
{
    namespace enums = picongpu::particles::atomicPhysics::enums;

    /** @class atomicPhysics sub-stage capping the collisional bound-free rates of one ion species
     *
     * @tparam T_IonSpecies ion species type
     */
    template<typename T_IonSpecies>
    struct CapCollisionalBoundFreeRates
    {
        // might be alias, from here on out no more
        //! resolved type of alias T_IonSpecies
        using IonSpecies = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_IonSpecies>;

        // ionization potential depression model to use
        using IPDModel = picongpu::atomicPhysics::IPDModel;

        //! call of kernel for every superCell
        HINLINE void operator()(picongpu::MappingDesc const mappingDesc) const
        {
            using AtomicDataType = typename picongpu::traits::GetAtomicDataType<IonSpecies>::type;

            //! nothing to cap unless at least one of the two collisional bound-free directions is active
            if constexpr(!picongpu::particles::atomicPhysics::CollisionalBoundFreePairCap::enabled
                         || !(AtomicDataType::switchElectronicIonization
                              || AtomicDataType::switchThreeBodyRecombination))
                return;
            else
            {
                // full local domain, no guards
                pmacc::AreaMapping<CORE + BORDER, MappingDesc> mapper(mappingDesc);
                pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();

                auto timeRemainingField = dc.get<
                    picongpu::particles::atomicPhysics::localHelperFields::TimeRemainingField<picongpu::MappingDesc>>(
                    "TimeRemainingField");

                auto rateCacheField = dc.get<picongpu::particles::atomicPhysics::localHelperFields::
                                                 RateCacheField<picongpu::MappingDesc, IonSpecies>>(
                    IonSpecies::FrameType::getName() + "_rateCacheField");

                auto electronHistogramField = dc.get<
                    picongpu::particles::atomicPhysics::electronDistribution::
                        LocalHistogramField<picongpu::atomicPhysics::ElectronHistogram, picongpu::MappingDesc>>(
                    "Electron_HistogramField");

                auto atomicData = dc.get<AtomicDataType>(IonSpecies::FrameType::getName() + "_atomicData");

                constexpr uint8_t n_max = AtomicDataType::ConfigNumber::numberLevels;
                constexpr uint32_t numberAtomicStatesOfSpecies
                    = picongpu::traits::GetNumberAtomicStates<IonSpecies>::value;
                constexpr uint32_t numberBins = picongpu::atomicPhysics::ElectronHistogram::numberBins;

                //    derive the per state scaling factors from the un-capped rates
                using ComputeScaling
                    = kernel::ComputeCollisionalBoundFreeScalingKernel<numberAtomicStatesOfSpecies>;

                PMACC_LOCKSTEP_KERNEL(ComputeScaling())
                    .template config<IonSpecies::FrameType::frameSize>(mapper.getGridDim())(
                        mapper,
                        timeRemainingField->getDeviceDataBox(),
                        rateCacheField->getDeviceDataBox());

                //    recompute the collisional bound-free cache entries with the scaling applied
                using ApplyCap = kernel::ApplyCollisionalBoundFreeCapKernel<
                    IPDModel,
                    n_max,
                    numberAtomicStatesOfSpecies,
                    numberBins,
                    AtomicDataType::switchElectronicIonization,
                    AtomicDataType::switchThreeBodyRecombination>;

                IPDModel::template callKernelWithIPDInput<ApplyCap, IonSpecies::FrameType::frameSize>(
                    dc,
                    mapper,
                    timeRemainingField->getDeviceDataBox(),
                    rateCacheField->getDeviceDataBox(),
                    electronHistogramField->getDeviceDataBox(),
                    atomicData->template getChargeStateDataDataBox<false>(),
                    atomicData->template getAtomicStateDataDataBox<false>(),
                    atomicData->template getBoundFreeStartIndexBlockDataBox<false>(),
                    atomicData->template getBoundFreeNumberTransitionsDataBox<false>(),
                    atomicData->template getBoundFreeTransitionDataBox<false, enums::TransitionOrdering::byLowerState>(),
                    atomicData
                        ->template getBoundFreeTransitionDataBox<false, enums::TransitionOrdering::byUpperState>());
            }
        }
    };

} // namespace picongpu::particles::atomicPhysics::stage
