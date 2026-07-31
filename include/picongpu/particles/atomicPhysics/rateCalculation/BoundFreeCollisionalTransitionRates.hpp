/* Copyright 2023-2026 Brian Marre, Axel Huebl, Prashant Sharma
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
#include "picongpu/particles/atomicPhysics/rateCalculation/ThresholdClip.hpp"
#include "picongpu/particles/atomicPhysics/rateCalculation/Multiplicities.hpp"
#include "picongpu/particles/atomicPhysics/stateRepresentation/ConfigNumber.hpp"

#include <pmacc/algorithms/math.hpp>

#include <cstdint>
#include <limits>

/** @file implements calculation of rates for bound-free collisional atomic physics transitions
 *
 * this includes ionization due to:
 *  - free electron interaction
 *  @todo photon ionization based processes, Brian Marre, 2023
 *  @todo recombination processes Brian Marre, 2023
 *
 * based on the rate calculation of FLYCHK, as extracted by Axel Huebl in the original
 *  flylite prototype.
 *
 * References:
 * - Axel Huebl
 *  first flylite prototype, not published
 *
 * - A. Burgess, M.C. Chidichimo.
 * "Electron impact ionization of complex ions."
 * Mon. Not. R. astr. Soc. 203, 1269-1280 (1983)
 *   based on the works of
 *     W. Lotz
 *     Zeitschrift fuer Physik 216, 241-247 (1968)
 */

namespace picongpu::particles::atomicPhysics::rateCalculation
{
    /** compilation of static rate- and crossSection-calc. methods for all processes
     * working on bound-free transition data
     *
     * @tparam T_numberLevels maximum principal quantum number of atomic states of species
     * @tparam T_debug activate debug print to console output
     *
     * @attention atomic data box input data is assumed to be in eV
     */
    template<uint8_t T_numberLevels, bool T_debug = false>
    struct BoundFreeCollisionalTransitionRates
    {
    private:
        /** beta factor from Burgess and Chidichimo(1983)
         *
         * @param Z screened charge of the ion, [e]
         *
         * @return unitless
         */
        HDINLINE static float_X betaFactor(float_X const Z)
        {
            float_X const x = (100._X * Z + 91._X) / (4._X * Z + 3._X);
            return 0.25_X * (math::sqrt(x) - 5._X);
        }

        /** w factor from Burgess and Chidichimo(1983)
         *
         * @param U (kinetic energy of interacting electron)/(ionization potential of initial level), unitless
         * @param beta beta factor from
         *
         * @attention U must be > 0
         *
         * @return unitless
         */
        HDINLINE static float_64 wFactor(float_X const U, float_X const beta)
        {
            return math::pow(static_cast<float_64>(math::log(U)), static_cast<float_64>(beta / U));
        }

    public:
        /** ionization cross section for a given bound-free transition and electron energy
         *
         * @tparam T_ChargeStateDataBox instantiated type of dataBox
         * @tparam T_AtomicStateDataBox instantiated type of dataBox
         * @tparam T_BoundFreeTransitionDataBox instantiated type of dataBox
         *
         * @param energyElectron kinetic energy of interacting free electron(/electron bin), in eV
         * @param ionizationPotentialDepression, in eV
         * @param transitionCollectionIndex index of transition in boundBoundTransitionDataBox
         * @param atomicStateDataBox access to atomic state property data
         * @param boundBoundTransitionDataBox access to bound-bound transition data
         *
         * @return unit: 10^6*b = 10^(-22)m^2; (b == barn = 10^(-28) m^2)
         *
         * @attention assumes that chargeStateDataBox, atomicStateDataBox and boundFreeTransitionDataBox
         *      belong to the same AtomicData instance
         */
        template<typename T_ChargeStateDataBox, typename T_AtomicStateDataBox, typename T_BoundFreeTransitionDataBox>
        HDINLINE static float_X collisionalIonizationCrossSection(
            // eV
            float_X const energyElectron,
            // eV
            float_X const ionizationPotentialDepression,
            uint32_t const transitionCollectionIndex,
            T_ChargeStateDataBox const chargeStateDataBox,
            T_AtomicStateDataBox const atomicStateDataBox,
            T_BoundFreeTransitionDataBox const boundFreeTransitionDataBox)
        {
            /// @todo provide as constexpr with one of the dataBoxes?, Brian Marre, 2023
            using S_ConfigNumber = typename T_AtomicStateDataBox::ConfigNumber;
            using LevelVector = pmacc::math::Vector<uint8_t, T_numberLevels>;

            uint32_t const lowerStateClctIdx
                = boundFreeTransitionDataBox.lowerStateCollectionIndex(transitionCollectionIndex);

            float_64 const combinatorialFactor
                = static_cast<float_64>(boundFreeTransitionDataBox.multiplicity(transitionCollectionIndex));

            // eV
            float_X const energyDifference = picongpu::particles::atomicPhysics::DeltaEnergyTransition::get(
                transitionCollectionIndex,
                atomicStateDataBox,
                boundFreeTransitionDataBox,
                ionizationPotentialDepression,
                chargeStateDataBox);

            if constexpr(picongpu::atomicPhysics::debug::rateCalculation::DEBUG_CHECKS)
            {
                uint32_t const upperStateClctIdx
                    = boundFreeTransitionDataBox.upperStateCollectionIndex(transitionCollectionIndex);

                auto const upperStateConfigNumber = atomicStateDataBox.configNumber(upperStateClctIdx);
                auto const lowerStateConfigNumber = atomicStateDataBox.configNumber(lowerStateClctIdx);

                if(S_ConfigNumber::getChargeState(upperStateConfigNumber)
                   < S_ConfigNumber::getChargeState(lowerStateConfigNumber))
                {
                    printf(
                        "atomicPhysics ERROR: upper and lower state inverted in "
                        "collisionalIonizationCrossSection() call\n");
                    return 0._X;
                }
            }

            float_X result = 0._X;

            /// @details ">", since energyDifference == energyElectron causes w == 0 -> cross section == 0, therefore
            /// skip calculation
            bool const electronEnergySufficientForTransitions = (energyDifference < energyElectron);

            /// @details energyDifference > = since otherwise U is a NaN
            /// @details also ensures energyElectron > 0, otherwise U = 0 for which the wFactor(U, beta) is not defined
            if((energyDifference > 0) && electronEnergySufficientForTransitions)
            {
                // a fitting factor well suited for Z >= 2, unitless
                constexpr float_64 C = 2.3;
                // m
                constexpr float_64 a0 = sim.si.getBohrRadius();
                // eV
                constexpr float_64 E_R = sim.si.conv().joule2eV(sim.si.getRydbergEnergy());
                /* m^2 / (m^2/10^6*b) * (eV)^2= m^2/m^2 * 10^6*b * eV^2
                 * unit:  10^6*b * eV^2*/
                constexpr float_64 scalingConstant
                    = C * picongpu::PI * pmacc::math::cPow(a0, 2u) / 1e-22 * pmacc::math::cPow(E_R, 2u);

                // e
                float_X const screenedCharge = atomicStateDataBox.screenedCharge(lowerStateClctIdx) - 1._X;

                // unitless
                float_X const U = energyElectron / energyDifference;
                // unitless
                float_X const beta = betaFactor(screenedCharge);
                // unitless
                float_64 const w = wFactor(U, beta);

                /* 1e6*b * (eV)^2 * unitless / (eV)^2 / unitless * log(unitless) * unitless
                 * unit: 1e6*b */
                float_X const crossSection = static_cast<float_X>(
                    scalingConstant * combinatorialFactor
                    / static_cast<float_64>(pmacc::math::cPow(energyDifference, 2u)) / static_cast<float_64>(U)
                    * math::log(static_cast<float_64>(U)) * w);

                if(crossSection > 0._X)
                    result = crossSection;
            }

            return result;
        }

        /** rate for collisional bound-free (ionization) transition of ion with free electron bin
         *
         * uses second order integration(bin middle); with useThresholdClippedBinIntegration
         * the bin straddling the (IPD-shifted) ionization threshold is clipped to its
         * above-threshold sub-interval, see ThresholdClip.hpp. The 3BR detailed-balance
         * rate inherits the clip through its per-bin EII sum.
         *
         * @todo implement higher order integrations, Brian Marre, 2023
         *
         * @tparam T_ChargeStateDataBox instantiated type of dataBox
         * @tparam T_AtomicStateDataBox instantiated type of dataBox
         * @tparam T_BoundFreeTransitionDataBox instantiated type of dataBox
         *
         * @param energyElectron kinetic energy of interacting electron(/electron bin), [eV]
         * @param energyElectronBinWidth energy width of electron bin, [eV]
         * @param densityElectrons [1/(m^3 * eV)], local superCell number density of electrons in this bin
         * @param densitySlopeToNextBin forward electron-density slope, [1/(m^3 * eV^2)]
         * @param ionizationPotentialDepression eV
         * @param transitionCollectionIndex index of transition in boundBoundTransitionDataBox
         * @param atomicStateDataBox access to atomic state property data
         * @param boundBoundTransitionDataBox access to bound-bound transition data
         *
         * @return unit: 1/sim.unit.time()
         */
        template<typename T_ChargeStateDataBox, typename T_AtomicStateDataBox, typename T_BoundFreeTransitionDataBox>
        HDINLINE static float_X rateCollisionalIonizationTransition(
            // eV
            float_X const energyElectron,
            // eV
            float_X const energyElectronBinWidth,
            // 1/(sim.unit.length()^3*eV)
            float_X const densityElectrons,
            // 1/(sim.unit.length()^3*eV^2)
            float_X const densitySlopeToNextBin,
            // eV
            float_X const ionizationPotentialDepression,
            uint32_t const transitionCollectionIndex,
            T_ChargeStateDataBox const chargeStateDataBox,
            T_AtomicStateDataBox const atomicStateDataBox,
            T_BoundFreeTransitionDataBox const boundFreeTransitionDataBox)
        {
            if constexpr(picongpu::atomicPhysics::debug::fixedRateMatrix::USE_FIXED_RATE_INSTEAD_OF_RATE_CALCULATION)
                return 0._X;

            // eV, may be repositioned by the threshold clip below
            float_X energy = energyElectron;
            // eV, may be shrunk by the threshold clip below
            float_X binWidth = energyElectronBinWidth;
            // 1/(sim.unit.length()^3*eV), may be reconstructed at the clipped midpoint
            float_X density = densityElectrons;

            if constexpr(useThresholdClippedBinIntegration)
            {
                // eV, IPD-shifted ionization threshold
                float_X const energyDifference = picongpu::particles::atomicPhysics::DeltaEnergyTransition::get(
                    transitionCollectionIndex,
                    atomicStateDataBox,
                    boundFreeTransitionDataBox,
                    ionizationPotentialDepression,
                    chargeStateDataBox);

                // entire bin below threshold, no contribution
                if(!thresholdClipBin(energy, binWidth, energyDifference, density, densitySlopeToNextBin))
                    return 0._X;
            }

            float_X sigma = collisionalIonizationCrossSection(
                // eV
                energy,
                ionizationPotentialDepression,
                transitionCollectionIndex,
                chargeStateDataBox,
                atomicStateDataBox,
                boundFreeTransitionDataBox); // [1e6*b]

            return picongpu::particles2::atomicPhysics::rateCalculation::collisionalRate(
                energy,
                binWidth,
                density,
                sigma);
        }

        /// @todo radiativeIonizationCrossSection, Scofield+Kramer
        /// @todo rateRadiativeIonization

        /// @todo spontaneousRadiativeRecombinationCrossSection
        /// @todo rateSpontaneousRadiaitveRecombination

        /** Saha-like detailed balance factor relating the three-body recombination rate of a bound-free transition
         *  to the collisional ionization rate of the same transition, for Maxwellian electrons
         *
         * R_3BR = D * R_EII, with
         *  D = g_lower/(2 * g_upper) * n_e * lambda_deBroglie^3(T_e) * exp(DeltaE/T_e)
         *  lambda_deBroglie^3(T_e) = (2 * pi * hbar^2 / (m_e * T_e))^(3/2)
         *
         * equivalent to the SCFLY/FLYCHK convention
         *  beta_3BR = 1.656415e-22 cm^3 / T_e[eV]^(3/2) * exp(DeltaE/T_e) * g_lower/g_upper * alpha_EII,
         * with 1.656415e-22 cm^3 = lambda_deBroglie^3(1 eV) / 2.
         *
         * @attention Maxwellian approximation! The factor assumes the local electron spectrum is Maxwellian with
         *  temperature T_e, in contrast to the histogram based collisional ionization rate.
         *
         * @param temperatureElectrons local electron temperature as k_B * T, [eV]
         * @param densityElectrons local total electron number density, [1/sim.unit.length()^3]
         * @param deltaEnergyTransition IPD-shifted energy difference of the transition, [eV]
         * @param multiplicityLowerState statistical weight of the lower(recombined) atomic state
         * @param multiplicityUpperState statistical weight of the upper(ionized) atomic state
         *
         * @return unitless, 0 for invalid input(T_e <= 0, n_e <= 0 or DeltaE <= 0)
         */
        HDINLINE static float_64 threeBodyDetailedBalanceFactor(
            // eV
            float_X const temperatureElectrons,
            // 1/sim.unit.length()^3
            float_X const densityElectrons,
            // eV
            float_X const deltaEnergyTransition,
            float_64 const multiplicityLowerState,
            float_64 const multiplicityUpperState)
        {
            /* barrier-free transitions (DeltaE <= 0, possible with strong IPD) have no defined detailed balance
             *  factor, skip them like the collisional ionization cross section does */
            if((temperatureElectrons <= 0._X) || (densityElectrons <= 0._X) || (deltaEnergyTransition <= 0._X))
                return 0.;

            // m^2 * eV, = 2 * pi * hbar^2 / (m_e * 1eV)
            constexpr float_64 deBroglieFactor = 2. * picongpu::PI * pmacc::math::cPow(sim.si.getHbar(), 2u)
                / (sim.si.getElectronMass() * sim.si.get_eV());
            // sim.unit.length()^3 / m^3
            constexpr float_64 volumeConversionFactor = 1. / pmacc::math::cPow(float_64(sim.unit.length()), 3u);

            // m^3
            float_64 const lambdaDeBroglieCubed
                = math::pow(deBroglieFactor / static_cast<float_64>(temperatureElectrons), 1.5);

            // cap exponent like SCFLY, prevents overflow at low temperature
            float_64 const exponent = pmacc::math::min(
                static_cast<float_64>(deltaEnergyTransition / temperatureElectrons),
                500.);

            /* unitless * 1/sim.unit.length()^3 * m^3 * sim.unit.length()^3/m^3 * unitless
             * unit: unitless */
            return 0.5 * multiplicityLowerState / multiplicityUpperState * static_cast<float_64>(densityElectrons)
                * lambdaDeBroglieCubed * volumeConversionFactor * math::exp(exponent);
        }

        /** relative probability of capturing a free electron of given energy in three-body recombination
         *
         * Three-body recombination captures a free electron by transferring its kinetic energy plus the released
         *  binding energy to a second, spectator, free electron. The cross section for such an energy transfer
         *  falls off as 1/(E_e + DeltaE)^2, giving the capture preference
         *  w(E_e) = (DeltaE / (E_e + DeltaE))^2, normalised to w(0) = 1.
         *
         * This is the classical(Thomson) energy transfer scaling of three-body capture and equals, up to
         *  normalisation, the shape of the ejected electron spectrum of the inverse collisional ionization, as
         *  required by detailed balance. Capture is therefore strongly biased towards the slow electrons of the
         *  distribution, in contrast to a selection weighted by electron number alone, which would draw the
         *  captured electron from the bulk of the distribution, <E_e> = 3/2 * T_e.
         *
         * @attention shape only! This distributes the capture over the electron histogram, the total three-body
         *  recombination rate is set by rateCollisionalThreeBodyRecombinationTransition() and is unaffected.
         *
         * @param energyElectron kinetic energy of the candidate captured electron(/electron bin), [eV]
         * @param deltaEnergyTransition IPD-shifted energy difference of the transition, [eV]
         *
         * @return unitless, in (0, 1]; 1 for undefined input(DeltaE <= 0), falling back to weighting the bins by
         *  electron number alone
         */
        HDINLINE static float_X threeBodyCaptureWeight(
            // eV
            float_X const energyElectron,
            // eV
            float_X const deltaEnergyTransition)
        {
            /* barrier-free transitions (DeltaE <= 0, possible with strong IPD) have no defined energy transfer
             *  suppression, weight by electron number alone */
            if((deltaEnergyTransition <= 0._X) || (energyElectron <= 0._X))
                return 1._X;

            // unitless, in (0, 1)
            float_X const suppression = deltaEnergyTransition / (energyElectron + deltaEnergyTransition);
            return suppression * suppression;
        }

        /** rate of collisional three-body recombination for a given bound-free transition
         *
         * Maxwellian detailed balance inverse of the collisional ionization rate of the same transition,
         *  see threeBodyDetailedBalanceFactor() for the model and its limitations.
         *
         * @tparam T_ChargeStateDataBox instantiated type of dataBox
         * @tparam T_AtomicStateDataBox instantiated type of dataBox
         * @tparam T_BoundFreeTransitionDataBox instantiated type of dataBox
         *
         * @param temperatureElectrons local electron temperature as k_B * T, [eV]
         * @param densityElectrons local total electron number density, [1/sim.unit.length()^3]
         * @param sumRateCollisionalIonization collisional ionization rate of the same transition, summed over all
         *  electron histogram bins, [1/sim.unit.time()]
         * @param ionizationPotentialDepression eV
         * @param transitionCollectionIndex index of transition in boundFreeTransitionDataBox
         * @param chargeStateDataBox access to charge state property data
         * @param atomicStateDataBox access to atomic state property data
         * @param boundFreeTransitionDataBox access to bound-free transition data
         *
         * @return unit: 1/sim.unit.time()
         */
        template<typename T_ChargeStateDataBox, typename T_AtomicStateDataBox, typename T_BoundFreeTransitionDataBox>
        HDINLINE static float_X rateCollisionalThreeBodyRecombinationTransition(
            // eV
            float_X const temperatureElectrons,
            // 1/sim.unit.length()^3
            float_X const densityElectrons,
            // 1/sim.unit.time()
            float_X const sumRateCollisionalIonization,
            // eV
            float_X const ionizationPotentialDepression,
            uint32_t const transitionCollectionIndex,
            T_ChargeStateDataBox const chargeStateDataBox,
            T_AtomicStateDataBox const atomicStateDataBox,
            T_BoundFreeTransitionDataBox const boundFreeTransitionDataBox)
        {
#if defined(PARAM_SUPPRESS_RECOMB_ABOVE_NMAX) && (PARAM_SUPPRESS_RECOMB_ABOVE_NMAX > 0)
            // momentary diagnostic: suppress three-body recombination into high-n Rydberg
            // states (highest occupied shell >= PARAM_SUPPRESS_RECOMB_ABOVE_NMAX) whose rate
            // coefficients are over-predicted vs native SCFLY. Ionization is left untouched.
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
            if(sumRateCollisionalIonization <= 0._X)
                return 0._X;

            uint32_t const lowerStateClctIdx
                = boundFreeTransitionDataBox.lowerStateCollectionIndex(transitionCollectionIndex);
            uint32_t const upperStateClctIdx
                = boundFreeTransitionDataBox.upperStateCollectionIndex(transitionCollectionIndex);

            // eV, IPD-shifted, same threshold as in the collisional ionization cross section
            float_X const energyDifference = picongpu::particles::atomicPhysics::DeltaEnergyTransition::get(
                transitionCollectionIndex,
                atomicStateDataBox,
                boundFreeTransitionDataBox,
                ionizationPotentialDepression,
                chargeStateDataBox);

            float_64 const rate = threeBodyDetailedBalanceFactor(
                                      temperatureElectrons,
                                      densityElectrons,
                                      energyDifference,
                                      static_cast<float_64>(atomicStateDataBox.multiplicity(lowerStateClctIdx)),
                                      static_cast<float_64>(atomicStateDataBox.multiplicity(upperStateClctIdx)))
                * static_cast<float_64>(sumRateCollisionalIonization);

            // protect float_X cast, overlarge rates only force smaller atomicPhysics sub-steps
            return static_cast<float_X>(
                pmacc::math::min(rate, static_cast<float_64>(std::numeric_limits<float_X>::max())));
        }

        /// @todo stimulatedRecombinationCrossSection
        /// @todo rateStimulatedRecombination
    };
} // namespace picongpu::particles::atomicPhysics::rateCalculation
