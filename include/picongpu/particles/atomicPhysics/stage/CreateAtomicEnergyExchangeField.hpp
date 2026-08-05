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

#include "picongpu/particles/atomicPhysics/localHelperFields/AtomicEnergyExchangeField.hpp"
#include "picongpu/particles/param.hpp"

#include <pmacc/particles/meta/FindByNameOrType.hpp>

#include <memory>

namespace picongpu::particles::atomicPhysics::stage
{
    template<typename T_IonSpecies>
    struct CreateAtomicEnergyExchangeField
    {
        using IonSpecies = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_IonSpecies>;

        template<typename T_MappingDescription>
        HINLINE void operator()(DataConnector& dataConnector, T_MappingDescription const& mappingDesc) const
        {
            using Field = localHelperFields::AtomicEnergyExchangeField<picongpu::MappingDesc, IonSpecies>;
            auto field = std::make_unique<Field>(mappingDesc);
            field->reset(0u);
            dataConnector.consume(std::move(field));
        }
    };
} // namespace picongpu::particles::atomicPhysics::stage
