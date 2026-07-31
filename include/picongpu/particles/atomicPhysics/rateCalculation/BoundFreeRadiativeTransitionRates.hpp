/* Copyright 2023-2026 Prashant Sharma
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

#include "picongpu/defines.hpp" // need atomicPhysics_Debug.param
#include "picongpu/particles/atomicPhysics/ConvertEnum.hpp"
#include "picongpu/particles/atomicPhysics/DeltaEnergyTransition.hpp"
#include "picongpu/particles/atomicPhysics/debug/param.hpp"
#include "picongpu/particles/atomicPhysics/rateCalculation/CollisionalRate.hpp"

#include <pmacc/algorithms/math.hpp>

#include <cstdint>
#include <limits>

/** @file implements calculation of rates for radiative bound-free atomic physics transitions
 *
 * this includes radiative-recombination:
 *  upperState + free electron -> lowerState + photon
 * @todo photoionization from a radiation field, requires a radiation field model, Prashant Sharma, 2026
 * @todo stimulated recombination, requires a radiation field model, Prashant Sharma, 2026
 *
 * The radiative-recombination cross section is obtained from the photoionization cross section of the same
 *  bound-free transition via the Milne(detailed balance) relation and integrated over the local electron
 *  histogram, following the SCFLY/FLYCHK conventions.
 *
 * The photoionization cross section is the Scofield-type fit stored in the bound-free transition input data
 *  (cxin1..cxin8), with a hydrogenic Kramers fallback for transitions without a valid fit, chosen such that the
 *  Maxwellian average reproduces SCFLY's hydrogenic recombination rate coefficient "alfxx" exactly.
 *
 * References:
 * - SCFLY source, phifun/phrec/alfxx, https://github.com/ComputationalRadiationPhysics/scfly
 * - H.-K. Chung et al. "FLYCHK: Generalized population kinetics and spectral model for rapid spectroscopic
 *   analysis for all elements", High Energy Density Physics 1, 3 (2005)
 * - Milne relation, e.g. G. B. Rybicki, A. P. Lightman, "Radiative Processes in Astrophysics"
 */

namespace picongpu::particles::atomicPhysics::rateCalculation
{
    /** compilation of static rate- and crossSection-calculation methods for radiative processes working on
     *  bound-free transition data
     *
     * @tparam T_numberLevels maximum principal quantum number of atomic states of species
     * @tparam T_debug activate debug print to console output
     *
     * @attention atomic data box input data is assumed to be in eV
     */
    template<uint8_t T_numberLevels, bool T_debug = false>
    struct BoundFreeRadiativeTransitionRates
    {
        /** Milne relation prefactor, [1/eV]
         *
         * sigma_RR(E_e) = milneConstant * g_lower/g_upper * E_gamma^2 / E_e * sigma_PI(E_gamma)
         *
         * derived from SCFLY's recombination integral such that the Maxwellian average of sigma_RR
         *  reproduces it exactly:
         *  milneConstant = 1.656415e-22[cm^3] * 2.4179e14[Hz/eV] * 1.636e9 / (2/sqrt(pi) * sqrt(2 * e/m_e)[cm/s])
         * numerically equal to the textbook Milne factor 1/(2 * m_e c^2) within 0.06%, the electron spin factor 2
         *  is included
         */
        static constexpr float_64 milneConstant = 9.790616361479e-07;

        /** Kramers fallback cross section constant, [1e6*b * eV^0.5]
         *
         * derived from SCFLY's hydrogenic recombination rate coefficient
         *  alfxx = 1.917e-15 * pn * g_l/g_u * sqrt(x) * x*e^x*E1(x) * DeltaE/Z_eff [cm^3/s], x = DeltaE/T,
         * which equals the Maxwellian average of Milne(kramersPhotoIonizationCrossSection) exactly for
         *  kramersConstant = 1.917e-15 / (2/sqrt(pi) * sqrt(2 * e/m_e) * milneConstant) = 2.925710e-17 cm^2*eV^0.5
         * converted to 1e6*b = 1e-18 cm^2. Close to SCFLY's internal Kramers opacity constant 2.9140e-17 cm^2.
         */
        static constexpr float_64 kramersConstant = 29.25710319866;

        /** photoionization cross section fit of a bound-free transition, SCFLY/Scofield-type
         *
         * sigma_PI(E) = cxin5 * exp(cxin1 + L*(cxin2 + L*(cxin3 + L*cxin4))) * (13.606/cxin6),
         *  L = max(ln(E/cxin6), 0), in 1e-18 cm^2 = 1e6*b, with cxin6 the fit edge energy in eV and cxin8 the
         *  upper validity limit in eV, above which the cross section is taken as zero.
         *
         * @attention does not check fit validity, see photoIonizationCrossSection() for the dispatch
         *
         * @param energyPhoton photon energy, [eV]
         * @param transitionCollectionIndex index of transition in boundFreeTransitionDataBox
         * @param boundFreeTransitionDataBox access to bound-free transition data
         *
         * @return unit: 10^6*b = 10^(-22)m^2; (b == barn = 10^(-28) m^2)
         */
        template<typename T_BoundFreeTransitionDataBox>
        HDINLINE static float_X scofieldPhotoIonizationCrossSection(
            // eV
            float_X const energyPhoton,
            uint32_t const transitionCollectionIndex,
            T_BoundFreeTransitionDataBox const boundFreeTransitionDataBox)
        {
            float_X const edgeEnergy = boundFreeTransitionDataBox.cxin6(transitionCollectionIndex);

            if((energyPhoton <= 0._X) || (edgeEnergy <= 0._X))
                return 0._X;
            // fit not valid above the upper validity limit
            if(energyPhoton > boundFreeTransitionDataBox.cxin8(transitionCollectionIndex))
                return 0._X;

            float_64 const L = pmacc::math::max(
                math::log(static_cast<float_64>(energyPhoton / edgeEnergy)),
                0.);

            float_64 const polynomial = static_cast<float_64>(boundFreeTransitionDataBox.cxin1(
                                            transitionCollectionIndex))
                + L
                    * (static_cast<float_64>(boundFreeTransitionDataBox.cxin2(transitionCollectionIndex))
                       + L
                           * (static_cast<float_64>(boundFreeTransitionDataBox.cxin3(transitionCollectionIndex))
                              + L
                                  * static_cast<float_64>(
                                      boundFreeTransitionDataBox.cxin4(transitionCollectionIndex))));

            /* 1e6*b * unitless * eV/eV, unit: 1e6*b
             * @note 1e-18 cm^2, the fit's natural unit, is exactly 1e6*b */
            return static_cast<float_X>(
                static_cast<float_64>(boundFreeTransitionDataBox.cxin5(transitionCollectionIndex))
                * math::exp(polynomial) * (13.606 / static_cast<float_64>(edgeEnergy)));
        }

        /** hydrogenic Kramers photoionization cross section, fallback for transitions without a valid fit
         *
         * sigma_PI(E) = kramersConstant * multiplicity * DeltaE^2.5 / (Z_eff * E^3)
         *
         * @param energyPhoton photon energy, [eV]
         * @param deltaEnergyTransition IPD-shifted energy difference of the transition, [eV]
         * @param screenedCharge screened charge of the lower(recombined) state, [e]
         * @param transitionMultiplicity combinatorial multiplicity of the bound-free transition
         *
         * @return unit: 10^6*b
         */
        HDINLINE static float_X kramersPhotoIonizationCrossSection(
            // eV
            float_X const energyPhoton,
            // eV
            float_X const deltaEnergyTransition,
            // e
            float_X const screenedCharge,
            float_64 const transitionMultiplicity)
        {
            if((energyPhoton <= 0._X) || (deltaEnergyTransition <= 0._X) || (screenedCharge <= 0._X))
                return 0._X;

            /* 1e6*b*eV^0.5 * unitless * eV^2.5 / (e * eV^3), unit: 1e6*b */
            return static_cast<float_X>(
                kramersConstant * transitionMultiplicity
                * math::pow(static_cast<float_64>(deltaEnergyTransition), 2.5)
                / (static_cast<float_64>(screenedCharge)
                   * pmacc::math::cPow(static_cast<float_64>(energyPhoton), 3u)));
        }

        /** photoionization cross section of a bound-free transition, fit with hydrogenic fallback
         *
         * dispatches between the Scofield-type fit and the Kramers fallback by fit validity:
         *  fit is valid if cxin5 > 1e-30 and cxin6 > 0 and cxin8 > DeltaE, following SCFLY
         *
         * @param energyPhoton photon energy, [eV]
         * @param deltaEnergyTransition IPD-shifted energy difference of the transition, [eV]
         * @param transitionCollectionIndex index of transition in boundFreeTransitionDataBox
         * @param atomicStateDataBox access to atomic state property data
         * @param boundFreeTransitionDataBox access to bound-free transition data
         *
         * @return unit: 10^6*b
         */
        template<typename T_AtomicStateDataBox, typename T_BoundFreeTransitionDataBox>
        HDINLINE static float_X photoIonizationCrossSection(
            // eV
            float_X const energyPhoton,
            // eV
            float_X const deltaEnergyTransition,
            uint32_t const transitionCollectionIndex,
            T_AtomicStateDataBox const atomicStateDataBox,
            T_BoundFreeTransitionDataBox const boundFreeTransitionDataBox)
        {
            bool const fitValid = (boundFreeTransitionDataBox.cxin5(transitionCollectionIndex) > 1.e-30_X)
                && (boundFreeTransitionDataBox.cxin6(transitionCollectionIndex) > 0._X)
                && (boundFreeTransitionDataBox.cxin8(transitionCollectionIndex) > deltaEnergyTransition);

            if(fitValid)
                return scofieldPhotoIonizationCrossSection(
                    energyPhoton,
                    transitionCollectionIndex,
                    boundFreeTransitionDataBox);

            uint32_t const lowerStateClctIdx
                = boundFreeTransitionDataBox.lowerStateCollectionIndex(transitionCollectionIndex);
            return kramersPhotoIonizationCrossSection(
                energyPhoton,
                deltaEnergyTransition,
                atomicStateDataBox.screenedCharge(lowerStateClctIdx),
                static_cast<float_64>(boundFreeTransitionDataBox.multiplicity(transitionCollectionIndex)));
        }

        /** Radiative-recombination cross section of a bound-free transition
         *
         * Milne(detailed balance) relation applied to the photoionization cross section of the same transition,
         *  sigma_RR(E_e) = milneConstant * g_lower/g_upper * E_gamma^2 / E_e * sigma_PI(E_gamma),
         *  E_gamma = E_e + DeltaE_IPD
         *
         * @param energyElectron kinetic energy of the captured free electron(/electron bin), [eV]
         * @param ionizationPotentialDepression, [eV]
         * @param transitionCollectionIndex index of transition in boundFreeTransitionDataBox
         * @param chargeStateDataBox access to charge state property data
         * @param atomicStateDataBox access to atomic state property data
         * @param boundFreeTransitionDataBox access to bound-free transition data
         *
         * @return unit: 10^6*b
         */
        template<typename T_ChargeStateDataBox, typename T_AtomicStateDataBox, typename T_BoundFreeTransitionDataBox>
        HDINLINE static float_X radiativeRecombinationCrossSection(
            // eV
            float_X const energyElectron,
            // eV
            float_X const ionizationPotentialDepression,
            uint32_t const transitionCollectionIndex,
            T_ChargeStateDataBox const chargeStateDataBox,
            T_AtomicStateDataBox const atomicStateDataBox,
            T_BoundFreeTransitionDataBox const boundFreeTransitionDataBox)
        {
            // eV, IPD-shifted, same threshold convention as collisional ionization
            float_X const deltaEnergyTransition = picongpu::particles::atomicPhysics::DeltaEnergyTransition::get(
                transitionCollectionIndex,
                atomicStateDataBox,
                boundFreeTransitionDataBox,
                ionizationPotentialDepression,
                chargeStateDataBox);

            if((energyElectron <= 0._X) || (deltaEnergyTransition <= 0._X))
                return 0._X;

            // eV
            float_X const energyPhoton = energyElectron + deltaEnergyTransition;

            float_X const sigmaPhotoIonization = photoIonizationCrossSection(
                energyPhoton,
                deltaEnergyTransition,
                transitionCollectionIndex,
                atomicStateDataBox,
                boundFreeTransitionDataBox);

            uint32_t const lowerStateClctIdx
                = boundFreeTransitionDataBox.lowerStateCollectionIndex(transitionCollectionIndex);
            uint32_t const upperStateClctIdx
                = boundFreeTransitionDataBox.upperStateCollectionIndex(transitionCollectionIndex);

            /* 1/eV * unitless * eV^2/eV * 1e6*b, unit: 1e6*b */
            float_64 const sigmaRadiativeRecombination = milneConstant
                * static_cast<float_64>(atomicStateDataBox.multiplicity(lowerStateClctIdx))
                / static_cast<float_64>(atomicStateDataBox.multiplicity(upperStateClctIdx))
                * pmacc::math::cPow(static_cast<float_64>(energyPhoton), 2u)
                / static_cast<float_64>(energyElectron) * static_cast<float_64>(sigmaPhotoIonization);

            /* the cross section diverges as 1/E_e towards zero electron energy, protect the float_X cast for a
             *  very low lying first histogram bin; overlarge rates only force smaller atomicPhysics sub-steps */
            return static_cast<float_X>(pmacc::math::min(
                sigmaRadiativeRecombination,
                static_cast<float_64>(std::numeric_limits<float_X>::max())));
        }

        /** Rate of radiative-recombination for a bound-free transition and a free electron bin
         *
         * uses second order integration(bin middle)
         *
         * @param energyElectron kinetic energy of the captured free electron(/electron bin), [eV]
         * @param energyElectronBinWidth energy width of electron bin, [eV]
         * @param densityElectrons [1/(sim.unit.length()^3*eV)], local superCell electron density in this bin
         * @param ionizationPotentialDepression, [eV]
         * @param transitionCollectionIndex index of transition in boundFreeTransitionDataBox
         * @param chargeStateDataBox access to charge state property data
         * @param atomicStateDataBox access to atomic state property data
         * @param boundFreeTransitionDataBox access to bound-free transition data
         *
         * @return unit: 1/sim.unit.time()
         */
        template<typename T_ChargeStateDataBox, typename T_AtomicStateDataBox, typename T_BoundFreeTransitionDataBox>
        HDINLINE static float_X rateRadiativeRecombinationTransition(
            // eV
            float_X const energyElectron,
            // eV
            float_X const energyElectronBinWidth,
            // 1/(sim.unit.length()^3*eV)
            float_X const densityElectrons,
            // eV
            float_X const ionizationPotentialDepression,
            uint32_t const transitionCollectionIndex,
            T_ChargeStateDataBox const chargeStateDataBox,
            T_AtomicStateDataBox const atomicStateDataBox,
            T_BoundFreeTransitionDataBox const boundFreeTransitionDataBox)
        {
#if defined(PARAM_SUPPRESS_RECOMB_ABOVE_NMAX) && (PARAM_SUPPRESS_RECOMB_ABOVE_NMAX > 0)
            // momentary diagnostic: suppress radiative-recombination into high-n Rydberg
            // states (highest occupied shell >= PARAM_SUPPRESS_RECOMB_ABOVE_NMAX) whose rate
            // coefficients are over-predicted vs native SCFLY. Photoionization is left untouched.
            {
                using GuardConfigNumber = typename T_AtomicStateDataBox::ConfigNumber;
                auto const guardLevelVector = GuardConfigNumber::getLevelVector(atomicStateDataBox.configNumber(
                    boundFreeTransitionDataBox.lowerStateCollectionIndex(transitionCollectionIndex)));
                uint8_t guardNMax = 0u;
                for(uint8_t guardN = 0u; guardN < GuardConfigNumber::numberLevels; ++guardN)
                    if(guardLevelVector[guardN] > static_cast<uint8_t>(0u))
                        guardNMax = static_cast<uint8_t>(guardN + 1u);
                if(guardNMax >= static_cast<uint8_t>(PARAM_SUPPRESS_RECOMB_ABOVE_NMAX))
                    return 0._X;
            }
#endif
            if constexpr(picongpu::atomicPhysics::debug::fixedRateMatrix::USE_FIXED_RATE_INSTEAD_OF_RATE_CALCULATION)
                return 0._X;

            float_X const sigma = radiativeRecombinationCrossSection(
                energyElectron,
                ionizationPotentialDepression,
                transitionCollectionIndex,
                chargeStateDataBox,
                atomicStateDataBox,
                boundFreeTransitionDataBox); // [1e6*b]

            return picongpu::particles2::atomicPhysics::rateCalculation::collisionalRate(
                energyElectron,
                energyElectronBinWidth,
                densityElectrons,
                sigma);
        }
    };
} // namespace picongpu::particles::atomicPhysics::rateCalculation
