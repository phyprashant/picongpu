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

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/atomicPhysics/InstantTransitionRateLimit.hpp"
#include "picongpu/particles/atomicPhysics/param.hpp"

#include <pmacc/algorithms/math.hpp>

namespace picongpu::particles::atomicPhysics
{
    /** proportional capping of unresolvably fast collisional bound-free rates
     *
     * Collisional ionization of a bound-free transition and its detailed balance inverse, three-body
     *  recombination, form a two-way pair between the transition's lower and upper state. At high electron
     *  density both rates, and in particular their sum over all transitions of a state, can far exceed the
     *  instant transition rate limit, i.e. the collisional bound-free sub-network reaches its Saha equilibrium
     *  much faster than the smallest sub-step the solver is willing to take. Resolving that relaxation costs an
     *  unbounded number of sub-steps and changes nothing, the sub-network sits at its equilibrium at the end of
     *  the PIC time step either way.
     *
     * The cap assigns every atomic state i a scaling factor
     *      scaling(i) = min(1, rateLimit / L(i)),
     *  with L(i) the un-capped total collisional bound-free loss rate of the state, i.e. the sum of its upward
     *  collisional ionization and its downward three-body recombination rates, and scales the rates of a
     *  transition t = (lower, upper) in *both* directions by the common factor
     *      s(t) = min(scaling(lower), scaling(upper)).
     *
     * Two properties make this safe:
     *  - Both directions of a transition carry the same factor, so R_ionization / R_recombination, and therefore
     *    the equilibrium population ratio of every pair, is exactly unchanged. Only the (never resolved)
     *    transient is stretched.
     *  - Every state's capped collisional bound-free loss rate is bounded by rateLimit, since
     *    sum_t s(t) * R_t(i) <= sum_t scaling(i) * R_t(i) = scaling(i) * L(i) <= rateLimit.
     *    This is what actually bounds the sub-step count; capping single transitions does not, a state's loss
     *    rate is dominated by the sum over its many transitions, not by any one of them.
     *
     * For a state whose own scaling is the binding one, the branching ratios between all of its collisional
     *  bound-free channels are preserved exactly as well.
     *
     * @attention the scaling must be applied consistently everywhere the rates enter, i.e. both when filling the
     *  rate cache and when rolling a specific transition, otherwise the cumulative sums of the choose-transition
     *  kernels no longer normalize to their cached total.
     */
    struct CollisionalBoundFreePairCap
    {
        //! is proportional capping active at all?
        static constexpr bool enabled = picongpu::atomicPhysics::RateSolverParam::capCollisionalBoundFreePairRates;

        /** per state scaling factor from a state's un-capped collisional bound-free loss rate
         *
         * @param stateLossRate un-capped sum of the state's collisional ionization and three-body recombination
         *  rates, [1/sim.unit.time()]
         *
         * @return unitless, in (0, 1], 1 if the state is resolvable or capping is disabled
         */
        HDINLINE static float_X stateScalingFactor(
            // 1/sim.unit.time()
            float_X const stateLossRate)
        {
            if constexpr(!enabled)
                return 1._X;
            else
            {
                //! maximum state loss rate still resolved by the sub-stepping, 1/sim.unit.time()
                constexpr float_X rateLimit = InstantTransitionRateLimit::get<float_X>();

                // NaN-safe, an unordered comparison leaves the rates untouched
                if(!(stateLossRate > rateLimit))
                    return 1._X;

                return rateLimit / stateLossRate;
            }
        }

        /** scaling factor of one transition, applied to both of its directions
         *
         * @param rateCache rate cache of the superCell, must already hold the per state scaling factors
         * @param lowerStateCollectionIndex collection index of the transition's lower(recombined) state
         * @param upperStateCollectionIndex collection index of the transition's upper(ionized) state
         *
         * @return unitless, in (0, 1]
         */
        template<typename T_RateCache>
        HDINLINE static float_X transitionScalingFactor(
            T_RateCache const& rateCache,
            uint32_t const lowerStateCollectionIndex,
            uint32_t const upperStateCollectionIndex)
        {
            if constexpr(!enabled)
                return 1._X;
            else
                return pmacc::math::min(
                    rateCache.collisionalBoundFreeScaling(lowerStateCollectionIndex),
                    rateCache.collisionalBoundFreeScaling(upperStateCollectionIndex));
        }
    };
} // namespace picongpu::particles::atomicPhysics
