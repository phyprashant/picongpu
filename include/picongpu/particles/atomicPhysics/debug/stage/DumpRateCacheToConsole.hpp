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

//! @file dump rateCache to console, debug stage of atomicPhysics


#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/atomicPhysics/localHelperFields/RateCacheField.hpp"
#include "picongpu/particles/param.hpp"

#include <pmacc/Environment.hpp>
#include <pmacc/particles/meta/FindByNameOrType.hpp>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>

namespace picongpu::particles::atomicPhysics::stage
{
    /** @class atomicPhysics sub-stage dumping rateCache for one ion species to console,
     * calls the corresponding kernel per superCell
     *
     * is called once per time step for the entire local simulation volume by the atomicPhysicsStage
     */
    template<typename T_IonSpecies>
    struct DumpRateCacheToConsole
    {
        // might be alias, from here on out no more
        //! resolved type of alias T_Species
        using IonSpecies = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_IonSpecies>;

        //! call of kernel for every superCell
        HINLINE void operator()(picongpu::MappingDesc const mappingDesc) const
        {
            pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();

            auto& rateCacheField = *dc.get<picongpu::particles::atomicPhysics::localHelperFields::
                                               RateCacheField<picongpu::MappingDesc, IonSpecies>>(
                IonSpecies::FrameType::getName() + "_rateCacheField");

            // copy rate cache from device to host so we can write it from host side
            rateCacheField.synchronize();

            std::string const filename = "rateCache_" + IonSpecies::FrameType::getName() + ".txt";
            std::ofstream out(filename, std::ios::out);
            if(!out.is_open())
            {
                std::cerr << "atomicPhysics ERROR: could not open " << filename << " for writing" << std::endl;
                return;
            }

            auto hostDataBox = rateCacheField.superCellField->getHostBuffer().getDataBox();
            auto const gridSuperCells = mappingDesc.getGridSuperCellsWithoutGuards();

            // dimension agnostic iteration, works for both 2D and 3D simulations
            int const numberSuperCells = gridSuperCells.productOfComponents();
            for(int linearIdx = 0; linearIdx < numberSuperCells; ++linearIdx)
            {
                pmacc::DataSpace<picongpu::simDim> const superCellFieldIdx
                    = pmacc::math::mapToND(gridSuperCells, linearIdx);
                hostDataBox(superCellFieldIdx).printToFile(out, superCellFieldIdx);
            }
        }
    };
} // namespace picongpu::particles::atomicPhysics::stage
