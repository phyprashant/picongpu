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

/** @file analytic Maxwellian closure for three-body recombination
 *
 * Three-body recombination is the inverse of collisional ionization and is obtained from
 * it by detailed balance, R_3BR = D * R_EII with the Saha-like factor
 *
 *   D = g_lower/(2 * g_upper) * n_e * lambda_deBroglie^3(T_e) * exp(DeltaE/T_e).
 *
 * The historic implementation formed D and R_EII separately and multiplied them, with
 * R_EII the sum over the sampled electron histogram. That fails in two independent ways
 * once DeltaE >> T_e:
 *
 *  1. The Maxwellian tail above the ionization threshold is not resolved by a finite
 *     number of macro particles, so the sampled R_EII is *exactly* zero and the product
 *     is exactly zero. Three-body recombination however proceeds through cold electrons
 *     and must stay finite. At DeltaE/T_e ~ 20 the true tail fraction is already ~1e-8,
 *     far below one macro particle per cell.
 *  2. exp(DeltaE/T_e) overflows float_64 beyond DeltaE/T_e ~ 709, which the historic
 *     code papered over by capping the exponent at 500. Any transition past the cap
 *     silently violates detailed balance.
 *
 * Both are artifacts of evaluating a well behaved product through two badly behaved
 * factors. Combining them analytically before evaluation removes both at once. For a
 * Maxwellian at T_e,
 *
 *   R_EII^M = n_e * 2/sqrt(pi) * T^(-3/2) * int_{DeltaE}^inf sigma(E) v(E) sqrt(E) e^(-E/T) dE,
 *
 * so, substituting the excess energy eps = E - DeltaE and then x = eps/T, the dangerous
 * exp(+DeltaE/T) of D cancels the exp(-DeltaE/T) of R_EII^M analytically and
 *
 *   R_3BR = 1/2 * g_lower/g_upper * n_e^2 * lambda_deBroglie^3(T) * 2/sqrt(pi) * T^(-1/2)
 *           * int_0^inf sigma(DeltaE + T*x) v(DeltaE + T*x) sqrt(DeltaE + T*x) e^(-x) dx.
 *
 * Neither large exponential is ever constructed and no cap is needed. The remaining
 * weight is e^(-x) on [0, inf), which is exactly the Gauss-Laguerre weight, so the
 * integral is evaluated with the fixed 16-node rule tabulated below. With the
 * Burgess-Chidichimo cross section used here the integrand reduces further, see
 * BoundFreeCollisionalTransitionRates::rateCollisionalThreeBodyRecombinationTransition().
 *
 * Accuracy of the 16-node rule against adaptive quadrature, over DeltaE/T_e in
 * [0.1, 2193]: better than 2e-4 relative for every screened charge >= 0, which is the
 * whole physical range (betaFactor() is monotone on [0, inf) with beta in [0, 0.127]).
 * Against the production code path, analyticMaxwellianThreeBodyRate() reproduces
 * D * sum_bins(rateCollisionalIonizationTransition()) over a finely resolved Maxwellian
 * to 5e-5, see TestRateCalculation::testThreeBodyRecombinationRate(). Both are far
 * inside the model error of the cross section itself.
 *
 * @attention the integrand behaves as x^(1+beta) at the origin, so a hypothetical
 *  screenedCharge below zero (beta < 0, weaker endpoint power) would degrade the rule to
 *  ~2e-3 at beta = -0.5. betaFactor() also has a pole at screenedCharge = -0.75. Neither
 *  is reachable from physical screened charges, and collisionalIonizationCrossSection()
 *  shares the same betaFactor(), so this is not specific to the closure.
 *
 * @attention this closure is analytic in T_e and therefore assumes the local electron
 *  spectrum is Maxwellian. The historic implementation was not free of that assumption
 *  either, it took a Maxwellian Saha factor at the histogram temperature and multiplied
 *  it by a kinetic ionization rate, which is not a consistent pairing. The change makes
 *  the 3BR closure internally consistent; it does not make it valid for a strongly
 *  non-Maxwellian spectrum. Collisional ionization itself remains fully kinetic and
 *  histogram driven, and the captured electron is still taken from an occupied bin.
 *
 * @attention T_e comes from the histogram and excludes overflow electrons, so an
 *  unresolved hot tail biases it low.
 *
 * @attention the underlying Saha relation is classical Maxwell-Boltzmann. At
 *  n_e = 2.4e30 1/m^3 and T_e = 5 eV the Fermi energy is ~65 eV and the electrons are
 *  strongly degenerate, so the closure stays physically approximate there even though
 *  it is now numerically sound. SCFLY makes the same classical assumption, so a
 *  code-to-code comparison remains meaningful.
 *
 * Switch, overridable at configure time:
 *   1 =^= analytic Maxwellian closure (default)
 *   0 =^= historic behavior, capped Saha factor times the sampled histogram EII sum
 *
 *   pic-configure -c "-DPARAM_OVERWRITES:LIST=\"-DPARAM_ATOMIC_PHYSICS_3BR_ANALYTIC_CLOSURE=0\"" ...
 *
 * @attention the switch only exists to reproduce pre-fix results for comparison. With
 *  it off, three-body recombination is zero wherever the sampled ionization tail is
 *  empty; do not use it for production.
 */

#pragma once

#include "picongpu/defines.hpp"

#include <cstdint>

#ifndef PARAM_ATOMIC_PHYSICS_3BR_ANALYTIC_CLOSURE
#    define PARAM_ATOMIC_PHYSICS_3BR_ANALYTIC_CLOSURE 1
#endif

namespace picongpu::particles::atomicPhysics::rateCalculation
{
    static_assert(
        PARAM_ATOMIC_PHYSICS_3BR_ANALYTIC_CLOSURE == 0 || PARAM_ATOMIC_PHYSICS_3BR_ANALYTIC_CLOSURE == 1,
        "PARAM_ATOMIC_PHYSICS_3BR_ANALYTIC_CLOSURE must be 0 or 1");

    //! compile-time switch, see file description
    constexpr bool useAnalyticThreeBodyClosure = (PARAM_ATOMIC_PHYSICS_3BR_ANALYTIC_CLOSURE != 0);

    /** fixed Gauss-Laguerre quadrature for int_0^inf f(x) e^(-x) dx ~ sum_k weight_k * f(node_k)
     *
     * 16 nodes, exact for polynomial f up to degree 31. The 3BR integrand is not polynomial,
     * it behaves as x^(1+beta) near the origin with beta = betaFactor(screenedCharge) in
     * [0, 0.13], which costs some of that order; the measured accuracy is quoted above.
     */
    struct GaussLaguerre16
    {
        static constexpr uint32_t numberNodes = 16u;

        /** @attention nodes and weights are handed out by value, not as arrays. A constexpr static array
         *  member has no address in device code, nvcc rejects any use of one from an HDINLINE function. A
         *  function-local constexpr array does work, so the table lives inside these accessors. */

        //! abscissa x_k, unitless
        HDINLINE static constexpr float_64 node(uint32_t const k)
        {
            constexpr float_64 values[numberNodes]
                = {8.76494104789277556e-02,
                   4.62696328915080390e-01,
                   1.14105777483122650e+00,
                   2.12928364509838053e+00,
                   3.43708663389320668e+00,
                   5.07801861454976766e+00,
                   7.07033853504823373e+00,
                   9.43831433639193840e+00,
                   1.22142233688661594e+01,
                   1.54415273687816175e+01,
                   1.91801568567531362e+01,
                   2.35159056939919076e+01,
                   2.85787297428821390e+01,
                   3.45833987022866225e+01,
                   4.19404526476883319e+01,
                   5.17011603395433212e+01};
            return values[k];
        }

        //! weight w_k, unitless, sum over all nodes is 1
        HDINLINE static constexpr float_64 weight(uint32_t const k)
        {
            constexpr float_64 values[numberNodes]
                = {2.06151714957804905e-01,
                   3.31057854950878305e-01,
                   2.65795777644214415e-01,
                   1.36296934296378736e-01,
                   4.73289286941256312e-02,
                   1.12999000803395977e-02,
                   1.84907094352632713e-03,
                   2.04271915308280893e-04,
                   1.48445868739815022e-05,
                   6.82831933087133066e-07,
                   1.88102484107972224e-08,
                   2.86235024297389688e-10,
                   2.12707903322412137e-12,
                   6.29796700251787974e-15,
                   5.05047370003560824e-18,
                   4.16146237037285100e-22};
            return values[k];
        }
    };
} // namespace picongpu::particles::atomicPhysics::rateCalculation
