/* Copyright 2023-2026 Brian Marre, Prashant Sharma
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

/** @file implements a super cell local cache of the electron weight captured from each electron histogram bin by
 *   recombination processes
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/atomicPhysics/SuperCellField.hpp"
#include "picongpu/particles/atomicPhysics/localHelperFields/CapturedWeightCache.hpp"
#include "picongpu/particles/atomicPhysics/param.hpp"

#include <cstdint>

namespace picongpu::particles::atomicPhysics::localHelperFields
{
    /** superCell field of the CapturedWeightCache for all electron histogram bins
     *
     * @tparam T_MappingDescription description of local mapping from device to grid
     */
    template<typename T_MappingDescription>
    struct CapturedWeightCacheField
        : public SuperCellField<
              CapturedWeightCache<picongpu::atomicPhysics::ElectronHistogram::numberBins>,
              T_MappingDescription,
              false /*no guards*/>
    {
        using ElementType = CapturedWeightCache<picongpu::atomicPhysics::ElectronHistogram::numberBins>;

        CapturedWeightCacheField(T_MappingDescription const& mappingDesc)
            : SuperCellField<
                  CapturedWeightCache<picongpu::atomicPhysics::ElectronHistogram::numberBins>,
                  T_MappingDescription,
                  false /*no guards*/>(mappingDesc)
        {
        }

        // required by ISimulationData
        std::string getUniqueId() override
        {
            return "CapturedWeightCacheField";
        }
    };
} // namespace picongpu::particles::atomicPhysics::localHelperFields
