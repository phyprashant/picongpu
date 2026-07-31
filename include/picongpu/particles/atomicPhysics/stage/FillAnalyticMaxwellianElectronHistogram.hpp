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

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/atomicPhysics/electronDistribution/LocalHistogramField.hpp"
#include "picongpu/particles/atomicPhysics/kernel/FillAnalyticMaxwellianElectronHistogram.kernel"
#include "picongpu/particles/atomicPhysics/localHelperFields/TimeRemainingField.hpp"
#include "picongpu/particles/atomicPhysics/param.hpp"

#include <pmacc/Environment.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/type/Area.hpp>

namespace picongpu::particles::atomicPhysics::stage
{
    struct FillAnalyticMaxwellianElectronHistogram
    {
        HINLINE void operator()(picongpu::MappingDesc const mappingDesc) const
        {
            pmacc::AreaMapping<CORE + BORDER, MappingDesc> mapper(mappingDesc);
            pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();

            auto& timeRemainingField
                = *dc.get<localHelperFields::TimeRemainingField<picongpu::MappingDesc>>("TimeRemainingField");

            using picongpu::atomicPhysics::ElectronHistogram;
            auto& electronHistogramField = *dc.get<
                electronDistribution::LocalHistogramField<ElectronHistogram, picongpu::MappingDesc>>(
                "Electron_HistogramField");

            PMACC_LOCKSTEP_KERNEL(kernel::FillAnalyticMaxwellianElectronHistogramKernel<ElectronHistogram>())
                .template config<ElectronHistogram::numberBins>(mapper.getGridDim())(
                    mapper,
                    timeRemainingField.getDeviceDataBox(),
                    electronHistogramField.getDeviceDataBox());
        }
    };
} // namespace picongpu::particles::atomicPhysics::stage
