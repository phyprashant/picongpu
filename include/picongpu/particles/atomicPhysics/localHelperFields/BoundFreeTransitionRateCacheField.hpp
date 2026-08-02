/* Copyright 2026 Brian Marre, Prashant Sharma
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
#include "picongpu/logging.hpp"

#include <pmacc/dataManagement/ISimulationData.hpp>
#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/memory/buffers/DeviceBuffer.hpp>
#include <pmacc/verify.hpp>

#include <cstdint>
#include <limits>
#include <memory>
#include <string>

namespace picongpu::particles::atomicPhysics::localHelperFields
{
    /** Device view of a compact, per-supercell transition-rate cache.
     *
     * Rates for all transitions are stored contiguously within each supercell. Bound-free data is loaded at run time,
     * so a flat allocation avoids a compile-time upper bound and the waste of a dense state-pair matrix.
     *
     * @attention an entry is only valid for a transition whose upper state was marked present in the rate cache of
     *  the *current* atomicPhysics sub-step. FillRateCache skips absent atomic states entirely, so the entries of
     *  their transitions still hold the value of an earlier sub-step, or 0 if never written.
     *
     * That invariant is what makes filling the cache once per sub-step and reading it once per selected ion sound:
     *  ions only ever occupy present states, CheckPresence runs before FillRateCache, and no ion changes state
     *  between FillRateCache and the last ChooseTransition call of the sub-step, see the sub-stepping loop in
     *  simulation/stage/AtomicPhysics.x.cpp. Never read an entry without first establishing that its upper state is
     *  present.
     */
    template<typename T_DataBox>
    class BoundFreeTransitionRateCacheDataBox
    {
    private:
        T_DataBox m_dataBox;
        pmacc::DataSpace<picongpu::simDim> m_superCellExtent;
        uint32_t m_numberTransitions;

    public:
        HDINLINE BoundFreeTransitionRateCacheDataBox(
            T_DataBox const dataBox,
            pmacc::DataSpace<picongpu::simDim> const superCellExtent,
            uint32_t const numberTransitions)
            : m_dataBox(dataBox)
            , m_superCellExtent(superCellExtent)
            , m_numberTransitions(numberTransitions)
        {
        }

        HDINLINE uint32_t linearIndex(
            pmacc::DataSpace<picongpu::simDim> const superCellFieldIdx,
            uint32_t const transitionCollectionIndex) const
        {
            uint32_t const superCellLinearIdx
                = static_cast<uint32_t>(pmacc::math::linearize(m_superCellExtent, superCellFieldIdx));
            return superCellLinearIdx * m_numberTransitions + transitionCollectionIndex;
        }

        HDINLINE float_X& rate(
            pmacc::DataSpace<picongpu::simDim> const superCellFieldIdx,
            uint32_t const transitionCollectionIndex)
        {
            return m_dataBox(static_cast<int>(linearIndex(superCellFieldIdx, transitionCollectionIndex)));
        }

        HDINLINE float_X rate(
            pmacc::DataSpace<picongpu::simDim> const superCellFieldIdx,
            uint32_t const transitionCollectionIndex) const
        {
            return m_dataBox(static_cast<int>(linearIndex(superCellFieldIdx, transitionCollectionIndex)));
        }
    };

    /** Flat cache of one downward collisional bound-free rate per transition and local supercell.
     *
     * @attention see BoundFreeTransitionRateCacheDataBox for the validity invariant of an entry.
     *
     * Device only, the host never reads the rates. The allocation is
     *  sizeof(float_X) * numberSuperCells * numberBoundFreeTransitions per ion species and can rival the particle
     *  memory for a large atomic input data set, therefore its size is logged on creation.
     */
    template<typename T_MappingDescription, typename T_IonSpecies>
    class BoundFreeTransitionRateCacheField : public pmacc::ISimulationData
    {
    private:
        using Buffer = pmacc::DeviceBuffer<float_X, DIM1>;

        std::unique_ptr<Buffer> m_buffer;
        pmacc::DataSpace<picongpu::simDim> m_superCellExtent;
        uint32_t m_numberTransitions;

    public:
        BoundFreeTransitionRateCacheField(T_MappingDescription const& mappingDesc, uint32_t const numberTransitions)
            : m_superCellExtent(mappingDesc.getGridSuperCellsWithoutGuards())
            , m_numberTransitions(numberTransitions)
        {
            /* multiply the extents as uint64_t, DataSpace stores signed int components and
             *  productOfComponents() would already have overflowed before the check below could catch it */
            uint64_t numberSuperCells = 1u;
            for(uint32_t d = 0u; d < picongpu::simDim; ++d)
                numberSuperCells *= static_cast<uint64_t>(m_superCellExtent[d]);
            uint64_t const numberEntries = numberSuperCells * static_cast<uint64_t>(m_numberTransitions);

            /* the dataBox indexes with a signed int and linearIndex() accumulates in uint32_t, a cache larger than
             *  that can not be addressed at all and must fail loudly instead of wrapping silently */
            PMACC_VERIFY_MSG(
                numberEntries <= static_cast<uint64_t>(std::numeric_limits<int>::max()),
                "atomicPhysics ERROR: bound-free transition rate cache too large to be addressed, reduce the number "
                "of superCells per device or the number of bound-free transitions");

            /* DeviceBuffer's owning constructor zeroes the allocation, no explicit initialization needed. The
             *  content of an entry of an absent atomic state is irrelevant either way, it is never read, see the
             *  validity invariant above. */
            m_buffer = std::make_unique<Buffer>(pmacc::MemSpace<DIM1>(static_cast<size_t>(numberEntries)));

            log<picLog::MEMORY>(
                "atomicPhysics: %1% bound-free transition rate cache: %2% MiB (%3% superCells * %4% transitions)")
                % T_IonSpecies::FrameType::getName()
                % (static_cast<float_64>(numberEntries * sizeof(float_X)) / (1024. * 1024.)) % numberSuperCells
                % m_numberTransitions;
        }

        std::string getUniqueId() override
        {
            return T_IonSpecies::FrameType::getName() + "_boundFreeTransitionRateCacheField";
        }

        /** required by ISimulationData, intentionally empty
         *
         * The cache is device only and only valid within the current atomicPhysics sub-step, there is nothing
         *  meaningful to synchronize to the host.
         */
        HINLINE void synchronize() override
        {
        }

        HINLINE auto getDeviceDataBox()
        {
            return BoundFreeTransitionRateCacheDataBox{m_buffer->getDataBox(), m_superCellExtent, m_numberTransitions};
        }
    };
} // namespace picongpu::particles::atomicPhysics::localHelperFields
