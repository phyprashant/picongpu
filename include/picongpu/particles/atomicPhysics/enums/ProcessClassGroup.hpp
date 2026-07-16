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

//! @file implements enum of different groups of physical processes

#pragma once

#include "picongpu/particles/atomicPhysics/ConvertEnum.hpp"

#include <cstdint>
#include <string>

namespace picongpu::particles::atomicPhysics
{
    namespace enums
    {
        //! enum of different physics based groups of transitions
        enum struct ProcessClassGroup : uint8_t
        {
            boundBoundBased = 0u,
            boundFreeBased = 1u,
            autonomousBased = 2u,
            ionizing = 3u,
            electronicCollisional = 4u,
            electricFieldBased = 5u,
            upward = 6u,
            downward = 7u,
            /** processes creating one or more free (ionization) macro electrons
             *
             * @attention distinct from boundFreeBased/ionizing: three-body recombination is bound-free based but
             *  must never spawn electrons, use this group in electron spawn logic */
            spawnsFreeElectron = 8u,
            /** processes capturing/removing a free electron, e.g. three-body recombination */
            capturesFreeElectron = 9u
        };
    } // namespace enums

    template<enums::ProcessClassGroup T_ProcessClassGroup>
    std::string enumToString()
    {
        if constexpr(u8(T_ProcessClassGroup) == u8(enums::ProcessClassGroup::boundBoundBased))
            return "boundBound";
        if constexpr(u8(T_ProcessClassGroup) == u8(enums::ProcessClassGroup::boundFreeBased))
            return "boundFree";
        if constexpr(u8(T_ProcessClassGroup) == u8(enums::ProcessClassGroup::autonomousBased))
            return "autonomous";
        if constexpr(u8(T_ProcessClassGroup) == u8(enums::ProcessClassGroup::ionizing))
            return "ionizing";
        if constexpr(u8(T_ProcessClassGroup) == u8(enums::ProcessClassGroup::electronicCollisional))
            return "electronicCollisional";
        if constexpr(u8(T_ProcessClassGroup) == u8(enums::ProcessClassGroup::electricFieldBased))
            return "electricFieldBased";
        if constexpr(u8(T_ProcessClassGroup) == u8(enums::ProcessClassGroup::upward))
            return "upward";
        if constexpr(u8(T_ProcessClassGroup) == u8(enums::ProcessClassGroup::downward))
            return "downard";
        if constexpr(u8(T_ProcessClassGroup) == u8(enums::ProcessClassGroup::spawnsFreeElectron))
            return "spawnsFreeElectron";
        if constexpr(u8(T_ProcessClassGroup) == u8(enums::ProcessClassGroup::capturesFreeElectron))
            return "capturesFreeElectron";
        return "unknown";
    }
} // namespace picongpu::particles::atomicPhysics
