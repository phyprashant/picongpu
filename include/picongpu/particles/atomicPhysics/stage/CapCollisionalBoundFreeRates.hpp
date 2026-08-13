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
#include <iostream>
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

        /** tell the user once that the requested pair cap is not being applied
         *
         * Silently dropping a requested numerical scheme is worse than the scheme itself; without this a
         *  single-direction setup would keep the un-capped sub-step cost with no indication why.
         */
        HINLINE static void reportPairCapInactive()
        {
            static bool alreadyReported = false;
            if(alreadyReported)
                return;
            alreadyReported = true;

            std::cout << "atomicPhysics: capCollisionalBoundFreePairRates is enabled but inactive for species "
                      << IonSpecies::FrameType::getName()
                      << ", it requires both electronic ionization and three-body recombination. Rates are"
                         " un-capped and the sub-step count is unbounded."
                      << std::endl;
        }

        //! call of kernel for every superCell
        HINLINE void operator()(picongpu::MappingDesc const mappingDesc) const
        {
            using AtomicDataType = typename picongpu::traits::GetAtomicDataType<IonSpecies>::type;

            /** the pair cap requires *both* collisional bound-free directions to be active
             *
             * Its correctness rests on scaling a transition's ionization and its detailed balance inverse by
             *  the same factor, which leaves their ratio, and therefore the equilibrium the pair relaxes to,
             *  exact. With only one direction compiled in there is no inverse to carry the matching factor,
             *  so scaling the surviving direction is not a stretched transient but an uncompensated change of
             *  how much ionization, or recombination, happens per PIC time step.
             *
             * This is not a hypothetical configuration. A state's collisional bound-free loss rate is
             *  routinely dominated by its electronic ionization alone - measured at up to 7x the instant
             *  transition rate limit for N2+ at 300 eV and 1e22 cm^-3, with three-body recombination
             *  contributing under 1e-5 of it - so an ionization-only setup would be capped just as hard as
             *  the full one, with nothing balancing it.
             *
             * @attention deliberately a silent no-op rather than a hard error: the cap defaults to enabled,
             *  so a static_assert here would break every otherwise valid single-direction setup at compile
             *  time. reportPairCapInactive() reports the decision once at runtime instead.
             */
            if constexpr(!picongpu::particles::atomicPhysics::CollisionalBoundFreePairCap::enabled
                         || !(AtomicDataType::switchElectronicIonization
                              && AtomicDataType::switchThreeBodyRecombination))
            {
                if constexpr(picongpu::particles::atomicPhysics::CollisionalBoundFreePairCap::enabled)
                    reportPairCapInactive();
                return;
            }
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
