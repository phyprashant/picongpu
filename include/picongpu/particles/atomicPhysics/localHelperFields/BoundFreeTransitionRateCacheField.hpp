/* Copyright 2026 Brian Marre, Prashant Sharma
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#pragma once

#include "picongpu/defines.hpp"

#include <pmacc/dataManagement/ISimulationData.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/memory/buffers/GridBuffer.hpp>

#include <cstdint>
#include <memory>
#include <string>

namespace picongpu::particles::atomicPhysics::localHelperFields
{
    /** Device view of a compact, per-supercell transition-rate cache.
     *
     * Rates for all transitions are stored contiguously within each supercell. Bound-free data is loaded at run time,
     * so a flat allocation avoids a compile-time upper bound and the waste of a dense state-pair matrix.
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
            return m_dataBox(linearIndex(superCellFieldIdx, transitionCollectionIndex));
        }

        HDINLINE float_X rate(
            pmacc::DataSpace<picongpu::simDim> const superCellFieldIdx,
            uint32_t const transitionCollectionIndex) const
        {
            return m_dataBox(linearIndex(superCellFieldIdx, transitionCollectionIndex));
        }
    };

    /** Flat cache of one downward collisional bound-free rate per transition and local supercell. */
    template<typename T_MappingDescription, typename T_IonSpecies>
    class BoundFreeTransitionRateCacheField : public pmacc::ISimulationData
    {
    private:
        using Buffer = pmacc::GridBuffer<float_X, 1u>;

        std::unique_ptr<Buffer> m_buffer;
        pmacc::DataSpace<picongpu::simDim> m_superCellExtent;
        uint32_t m_numberTransitions;

    public:
        BoundFreeTransitionRateCacheField(
            T_MappingDescription const& mappingDesc,
            uint32_t const numberTransitions)
            : m_superCellExtent(mappingDesc.getGridSuperCellsWithoutGuards())
            , m_numberTransitions(numberTransitions)
        {
            uint32_t const numberSuperCells
                = static_cast<uint32_t>(m_superCellExtent.productOfComponents());
            m_buffer = std::make_unique<Buffer>(pmacc::DataSpace<1u>(numberSuperCells * m_numberTransitions));
        }

        std::string getUniqueId() override
        {
            return T_IonSpecies::FrameType::getName() + "_boundFreeTransitionRateCacheField";
        }

        HINLINE void synchronize() override
        {
            m_buffer->deviceToHost();
        }

        HINLINE auto getDeviceDataBox()
        {
            return BoundFreeTransitionRateCacheDataBox{
                m_buffer->getDeviceBuffer().getDataBox(),
                m_superCellExtent,
                m_numberTransitions};
        }
    };
} // namespace picongpu::particles::atomicPhysics::localHelperFields
