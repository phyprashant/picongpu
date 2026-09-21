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

/** @file merge electrons stage of atomicPhysics
 *
 * bounds the macro electron count grown by ionization <-> recombination cycling, see particleMerging.param
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/fields/FieldJ.hpp"
#include "picongpu/particles/atomicPhysics/electronDistribution/LocalHistogramField.hpp"
#include "picongpu/particles/atomicPhysics/kernel/MergeElectrons.kernel"
#include "picongpu/particles/atomicPhysics/param.hpp"
#include "picongpu/particles/param.hpp"
#include "picongpu/particles/traits/GetShape.hpp"

#include <pmacc/Environment.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/particles/meta/FindByNameOrType.hpp>
#include <pmacc/type/Area.hpp>

#include <cstdint>

namespace picongpu::particles::atomicPhysics::stage
{
    /** @class atomicPhysics stage merging the macro electrons of every superCell above the trigger
     *
     * called once per PIC step after the atomicPhysics sub-stepping for every isElectron species
     *
     * @tparam T_ElectronSpecies species for which to call the functor
     */
    template<typename T_ElectronSpecies>
    struct MergeElectrons
    {
        // might be alias, from here on out no more
        //! resolved type of alias T_ElectronSpecies
        using ElectronSpecies = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_ElectronSpecies>;

        //! call of kernel for every superCell
        HINLINE void operator()(picongpu::MappingDesc const mappingDesc, uint32_t const currentStep) const
        {
            // full local domain, no guards
            pmacc::AreaMapping<CORE + BORDER, MappingDesc> mapper(mappingDesc);
            pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();

            auto& electrons = *dc.get<ElectronSpecies>(ElectronSpecies::FrameType::getName());
            auto& electronHistogramField
                = *dc.get<picongpu::particles::atomicPhysics::electronDistribution::
                              LocalHistogramField<picongpu::atomicPhysics::ElectronHistogram, picongpu::MappingDesc>>(
                    "Electron_HistogramField");
            // reset before this stage, consumed by the field solver after the particle push
            auto& fieldJ = *dc.get<FieldJ>(FieldJ::getName());

            using Shape = typename ::picongpu::traits::GetShape<ElectronSpecies>::type;
            using MergeElectronsKernel = picongpu::particles::atomicPhysics::kernel::
                MergeElectronsKernel<picongpu::atomicPhysics::ElectronHistogram, Shape>;

            PMACC_LOCKSTEP_KERNEL(MergeElectronsKernel())
                .config(mapper.getGridDim(), electrons)(
                    mapper,
                    electrons.getDeviceParticlesBox(),
                    electronHistogramField.getDeviceDataBox(),
                    fieldJ.getDeviceDataBox(),
                    currentStep);

            // remove the gaps left by merged away macro electrons
            electrons.fillAllGaps();
        }
    };
} // namespace picongpu::particles::atomicPhysics::stage
