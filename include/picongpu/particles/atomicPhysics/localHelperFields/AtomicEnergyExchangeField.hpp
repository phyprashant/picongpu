/* Copyright 2026 Prashant Sharma
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#pragma once

#include "picongpu/particles/atomicPhysics/SuperCellField.hpp"
#include "picongpu/particles/atomicPhysics/localHelperFields/AtomicEnergyExchange.hpp"

#include <string>

namespace picongpu::particles::atomicPhysics::localHelperFields
{
    /** Persistent per-supercell energy ledger for one atomic ion species. */
    template<typename T_MappingDescription, typename T_IonSpecies>
    struct AtomicEnergyExchangeField
        : public SuperCellField<AtomicEnergyExchange, T_MappingDescription, false /* no guards */>
    {
        using Base = SuperCellField<AtomicEnergyExchange, T_MappingDescription, false /* no guards */>;

        AtomicEnergyExchangeField(T_MappingDescription const& mappingDesc)
            : Base(mappingDesc)
        {
        }

        std::string getUniqueId() override
        {
            return T_IonSpecies::FrameType::getName() + "_atomicEnergyExchangeField";
        }
    };
} // namespace picongpu::particles::atomicPhysics::localHelperFields
