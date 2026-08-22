/* Copyright 2023-2024 Brian Marre
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

/** @file fixAtomicState sub-stage of atomicPhysics
 *
 * searches for inconsistencies between atomic state and charge state of macro-ions and fixes them by setting the
 *  atomic state to the ground state of the charge state if mismatched.
 *
 * The charge state of macro-ions is specified both directly by the boundElectrons and indirectly by the
 *  atomicStateCollectionIndex attributes, therefore requiring them to always be update together.
 * Unfortunately this requires access to the atomicState DataBase unavailable in code disregarding atomic states.
 *
 * Therefore it is possible that the boundElectrons attribute is updated without updating the
 * atomicStateCollectionIndex attribute outside the atomic physics step.
 *
 * To prevent this inconsistency from propagating, we check all macro ions with atomic physics for such inconsistencies
 * at the start of every atomicPhysics step and set the atomic state to ground state of the indicated charge state
 * if they do not match.
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/atomicPhysics/kernel/FixAtomicState.kernel"
#include "picongpu/particles/atomicPhysics/localHelperFields/AtomicEnergyExchangeField.hpp"
#include "picongpu/particles/param.hpp"
#include "picongpu/particles/traits/GetAtomicDataType.hpp"

#include <pmacc/Environment.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/particles/meta/FindByNameOrType.hpp>
#include <pmacc/type/Area.hpp>

#include <cstdint>

namespace picongpu::particles::atomicPhysics::stage
{
    /** Fixes mismatches between atomic state and boundElectrons charge state
     *
     * If the charge state of the atomicStateCollectionIndex particle attribute is inconsistent with the charge state
     * indicated by the boundElectrons particle attributes, sets the atomic state to ground state of charge state
     * specified by boundElectrons particle attribute.
     *
     * @tparam T_IonSpecies ion species type
     */
    template<typename T_IonSpecies, bool T_countExternalIonization>
    struct FixAtomicStateImpl
    {
        // might be alias, from here on out no more
        //! resolved type of alias T_IonSpecies
        using IonSpecies = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_IonSpecies>;

        //! call of kernel for every superCell
        HINLINE void operator()(picongpu::MappingDesc const mappingDesc) const
        {
            // full local domain, no guards
            pmacc::AreaMapping<CORE + BORDER, MappingDesc> mapper(mappingDesc);
            pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();

            using AtomicDataType = typename picongpu::traits::GetAtomicDataType<IonSpecies>::type;

            auto& ions = *dc.get<IonSpecies>(IonSpecies::FrameType::getName());

            auto& atomicData = *dc.get<AtomicDataType>(IonSpecies::FrameType::getName() + "_atomicData");

            auto atomicEnergyExchangeField
                = dc.get<particles::atomicPhysics::localHelperFields::AtomicEnergyExchangeField<
                    picongpu::MappingDesc,
                    IonSpecies>>(IonSpecies::FrameType::getName() + "_atomicEnergyExchangeField");

            PMACC_LOCKSTEP_KERNEL(picongpu::particles::atomicPhysics::kernel::
                                      FixAtomicStateKernel<IonSpecies, T_countExternalIonization>())
                .config(mapper.getGridDim(), ions)(
                    mapper,
                    ions.getDeviceParticlesBox(),
                    atomicData.template getChargeStateOrgaDataBox<false>(),
                    atomicData.template getAtomicStateDataDataBox<false>(),
                    atomicData.template getChargeStateDataDataBox<false>(),
                    atomicEnergyExchangeField->getDeviceDataBox());
        }
    };

    /* Both spellings below take exactly one template parameter on purpose: pmacc::meta::ForEach substitutes a
     * boost::mpl placeholder, and a class template with a second parameter is not a valid mpl lambda expression,
     * defaulted or not.
     */

    //! per-step call, counts macro ions arriving from an ionization module outside atomicPhysics
    template<typename T_IonSpecies>
    struct FixAtomicState : FixAtomicStateImpl<T_IonSpecies, true>
    {
    };

    /** initialisation call, external-ionization counting switched off
     *
     * Every ion still has an invalid collection index at initialisation and none of it is energy moving, so counting
     * it as arriving from an external ionizer would be wrong.
     */
    template<typename T_IonSpecies>
    struct FixAtomicStateInit : FixAtomicStateImpl<T_IonSpecies, false>
    {
    };
} // namespace picongpu::particles::atomicPhysics::stage
