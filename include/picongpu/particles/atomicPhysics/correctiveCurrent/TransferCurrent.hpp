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

/** @file exact current for a charge transported between two arbitrary points
 *
 * Recombination moves negative charge from donor electrons to the recombining ion without
 * depositing any current, so eps0*div(E) - rho picks up a permanent error. This primitive
 * deposits the current such a transfer would have produced.
 *
 * @attention the ordinary current solver cannot be reused: CurrentDeposition.x.cpp:59-71
 *  compile-asserts at most one cell crossing per step, whereas a transfer leg spans up to half a
 *  superCell. JIonizationAssignment is a point deposition, not transport.
 */

#pragma once

#include "picongpu/defines.hpp"

#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/math/Vector.hpp>

#include <cstdint>

namespace picongpu::particles::atomicPhysics::correctiveCurrent
{
    /** axis traversal order for one transfer leg
     *
     * @attention THE TWO LEGS OF A HUB TRANSFER MUST USE OPPOSITE ORDERS.
     *  With Forward on both, a coincident donor and ion trace
     *      a -> (h_x, a_y) -> h -> (a_x, h_y) -> a
     *  a CLOSED LOOP: div(J) is exactly zero but the current is NOT. Measured on a 32x32 grid,
     *  PCS, coincident equal-weight pair:
     *      same order:      max|div J|dt = 3.2e-35   max|J| = 2.9e-02   max|curl J| = 2.6e-02
     *      opposite order:  max|div J|dt = 0         max|J| = 0         max|curl J| = 0
     *  A spurious circulating current drives B and does work while satisfying Gauss perfectly, so
     *  a divergence-only test CANNOT catch it. With opposite orders both legs retrace the shared
     *  corner and cancel identically.
     *
     *  Axis order fixes part of the divergence-free freedom. Like the hub position it is a
     *  modelling choice that must be documented, not treated as arbitrary.
     */
    enum struct AxisOrder : uint8_t
    {
        //! x, then y, then z - use for the DONOR leg (particle -> hub)
        Forward,
        //! z, then y, then x - use for the ION leg (hub -> particle)
        Reverse
    };

    namespace detail
    {
        //! half-width of cells to visit around a position, generous by one cell against rounding
        template<typename T_Shape>
        static constexpr int shapeReach = static_cast<int>(T_Shape::ChargeAssignment::support) / 2 + 1;
    } // namespace detail

    /** deposit the exact current for charge Q moving from `from` to `to`
     *
     * Splits the displacement into per-axis moves and uses the cumulative-sum identity. For a
     * pure-x move at fixed transverse position, with assignment function S:
     *
     *     drho(i)       =  Q * [S_i(b) - S_i(a)] * S_perp
     *     Jx(i+1/2)*dt  = -Q * [sum_{k<=i} (S_k(b) - S_k(a))] * S_perp
     *
     * so (Jx(i+1/2) - Jx(i-1/2))*dt = -drho(i) telescopes EXACTLY, for any shape and any
     * displacement. This is the Esirkepov identity applied one axis at a time.
     *
     * Storage convention matches ChargeConservation.tpp's backward-difference Div: the value at
     * cell index i is the flux through the face between cell i and cell i+1.
     *
     * Units follow Esirkepov2D.hpp:178 - a current DENSITY, i.e. divided by the cell volume and
     * dt and multiplied by the moved axis' cell edge length, because the in-cell position
     * attribute is normalised to [0,1).
     *
     * @param from    start position in cell units relative to the box origin
     * @param to      end position in cell units relative to the box origin
     * @param charge  the charge TRANSPORTED. For a donor losing weight this is -e*dw, the
     *                negative charge that leaves - NOT the +e*dw change in rho. Passing the
     *                charge change flips div(J) and leaves a residual of exactly 2*drho.
     * @param dt      the time over which FieldJ is consumed, i.e. the PIC step - NOT the atomic
     *                sub-step. FieldJ is reset once per PIC step and consumed once, so depositing
     *                Q/dt_sub would overcount by the sub-step count (200+ here).
     * @param cellSize cell edge lengths, ALWAYS 3 components. Passed in rather than read from
     *                `sim.pic` so the primitive is pure and can be unit-tested without the full
     *                param stack, and so the unequal-cell-size gate can vary it.
     *
     * @attention the volume is the full THREE-dimensional cell volume, dx*dy*dz, even in 2D.
     *  `sim.pic.getCellSize()` returns 3 components in 2D3V as well: dimension.param:34 states
     *  that in 2D `getCellSize().z()` is the system size integrated over. rho is a 3D charge
     *  density, so dividing by dx*dy alone would scale every deposited current by 1/dz. The same
     *  3-component product appears in Esirkepov2D.hpp:178.
     *
     * @attention BOTH LEGS OF A HUB TRANSFER MUST USE THE SAME T_Shape. The hub cancels only
     *  because the donor leg's arrival term and the ion leg's departure term are the same
     *  function of the hub position. With different shapes the hub becomes a real source of
     *  charge - see the "mismatched shapes break the hub" test. Call sites must therefore pass a
     *  single shared shape alias, not each species' own `UsedParticleShape`.
     *
     * @attention guard margin. A leg stays inside its superCell (hub = superCell centre) but the
     *  deposition reaches `floor(pos) +- reach` with reach = support/2 + 1 = 3 for PCS. FieldJ's
     *  guard is derived from the current solver: Esirkepov2D.hpp:47-48 gives
     *  lower = supp/2 + 1 - (supp+1)%2 = 3 and upper = (supp+1)/2 + 1 = 4 for supp = 5. So the
     *  existing margin covers this primitive exactly, with no extra exchange needed. Re-check if
     *  the shape or the hub choice changes.
     */
    /** @tparam T_Hierarchy atomic scope of the deposition. hierarchy::Threads maps to CUDA's
     *   atomicAdd_block and is correct ONLY when the target is private to the block, e.g. the
     *   per-superCell pending pool. Writing straight into the shared FieldJ needs
     *   hierarchy::Grids, since a shape of support s reaches s/2+1 cells past the particle and
     *   therefore into cells that a neighbouring superCell's block also writes. PIConGPU's own
     *   ionization current deposits into FieldJ with alpaka's default, which is Grids.
     */
    template<
        typename T_Shape,
        typename T_Worker,
        typename T_JBox,
        typename T_Hierarchy = ::alpaka::hierarchy::Threads>
    HDINLINE void depositTransferCurrent(
        T_Worker const& worker,
        T_JBox jBox,
        pmacc::math::Vector<float_X, simDim> const& from,
        pmacc::math::Vector<float_X, simDim> const& to,
        float_X const charge,
        float_X const dt,
        pmacc::math::Vector<float_X, 3u> const& cellSize,
        AxisOrder const order)
    {
        using Assignment = typename T_Shape::ChargeAssignment;
        constexpr int reach = detail::shapeReach<T_Shape>;
        constexpr int width = 2 * reach + 1;
        //! transverse axes: 1 in 2D, 2 in 3D
        constexpr uint32_t nTransverse = simDim - 1u;

        Assignment const assign{};
        float_X const cellVolume = cellSize.productOfComponents();

        // the running end of the leg: one axis is completed per outer iteration
        auto current = from;

        for(uint32_t stepIdx = 0u; stepIdx < simDim; ++stepIdx)
        {
            uint32_t const axis = (order == AxisOrder::Forward) ? stepIdx : (simDim - 1u - stepIdx);

            float_X const a = current[axis];
            float_X const b = to[axis];
            if(a == b)
                continue;

            /* current DENSITY normalisation, Esirkepov2D.hpp:178. cellSize[axis] enters because
             * the cumulative sum is dimensionless while J must be per unit area. */
            float_X const norm = charge * cellSize[axis] / (cellVolume * dt);



            /* the transverse shape factors do NOT depend on the cumulative-sum index, so hoist
             * them out of that loop entirely - they were previously re-evaluated for every cell
             * along the moved axis */
            uint32_t transverseAxis[nTransverse];
            int transverseBase[nTransverse];
            float_X transverseWeight[nTransverse][width];
            {
                uint32_t t = 0u;
                for(uint32_t d = 0u; d < simDim; ++d)
                {
                    if(d == axis)
                        continue;
                    transverseAxis[t] = d;
                    transverseBase[t] = static_cast<int>(pmacc::math::floor(current[d])) - reach;
                    for(int o = 0; o < width; ++o)
                        transverseWeight[t][o]
                            = assign(static_cast<float_X>(transverseBase[t] + o) - current[d]);
                    ++t;
                }
            }

            int transverseSpan = 1;
            for(uint32_t t = 0u; t < nTransverse; ++t)
                transverseSpan *= width;

            int const lo = static_cast<int>(pmacc::math::floor(pmacc::math::min(a, b))) - reach;
            int const hi = static_cast<int>(pmacc::math::floor(pmacc::math::max(a, b))) + reach;

            float_X cumulative = 0._X;
            for(int i = lo; i <= hi; ++i)
            {
                cumulative += assign(static_cast<float_X>(i) - b) - assign(static_cast<float_X>(i) - a);
                if(cumulative == 0._X)
                    continue;

                /* iterate the transverse support box with a flat index, so the same code serves
                 * 2D and 3D. A hand-branched `if constexpr(simDim == 2)` would be correct inside
                 * this template, but the equivalent in a non-template helper silently fails to
                 * compile in the inactive dimension - a trap already hit once in the test. */
                for(int f = 0; f < transverseSpan; ++f)
                {
                    pmacc::DataSpace<simDim> idx;
                    idx[axis] = i;
                    float_X weight = -norm * cumulative;
                    int r = f;
                    for(uint32_t t = 0u; t < nTransverse; ++t)
                    {
                        int const o = r % width;
                        r /= width;
                        idx[transverseAxis[t]] = transverseBase[t] + o;
                        weight *= transverseWeight[t][o];
                    }
                    if(weight == 0._X)
                        continue;
                    alpaka::atomicAdd(worker.getAcc(), &(jBox(idx)[axis]), weight, T_Hierarchy{});
                    if constexpr(requires { jBox.record(worker, idx, axis, weight); })
                        jBox.record(worker, idx, axis, weight);
                }
            }

            current[axis] = b;
        }
    }
} // namespace picongpu::particles::atomicPhysics::correctiveCurrent
