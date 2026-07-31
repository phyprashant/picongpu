/* Copyright 2023-2026 Brian Marre, Prashant Sharma
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

/** @file unit tests for the rate cross section calculation
 *
 * test are activated by the global debug switch debug::rateCalculation::RUN_UNIT_TESTS
 *  in atomicPhysics_Debug.param
 *
 * for updating the tests see the python [rate calculator tool](
 *  https://github.com/BrianMarre/picongpuAtomicPhysicsTools/tree/dev/RateCalculationReference)
 */

#pragma once

#include "picongpu/defines.hpp"
// need unit.param

#include "picongpu/particles/atomicPhysics/ConvertEnum.hpp"
#include "picongpu/particles/atomicPhysics/atomicData/AtomicTuples.def"
#include "picongpu/particles/atomicPhysics/debug/TestRelativeError.hpp"
#include "picongpu/particles/atomicPhysics/enums/ADKLaserPolarization.hpp"
#include "picongpu/particles/atomicPhysics/enums/TransitionOrdering.hpp"
#include "picongpu/particles/atomicPhysics/rateCalculation/BoundBoundTransitionRates.hpp"
#include "picongpu/particles/atomicPhysics/rateCalculation/BoundFreeCollisionalTransitionRates.hpp"
#include "picongpu/particles/atomicPhysics/rateCalculation/BoundFreeFieldTransitionRates.hpp"
#include "picongpu/particles/atomicPhysics/rateCalculation/BoundFreeRadiativeTransitionRates.hpp"
#include "picongpu/particles/atomicPhysics/stateRepresentation/ConfigNumber.hpp"

#include <pmacc/algorithms/math.hpp>

#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <tuple>

namespace picongpu::particles::atomicPhysics::debug
{
    using tranOrd = picongpu::particles::atomicPhysics::enums::TransitionOrdering;

    /** collection of rate calculation tests
     *
     * see picongpuAtomicPhysicsTools repo, RateCalculationReference/calculatorMain.py
     *  for calculating reference rates.
     *
     * @tparam T_n_max maximum principal occupation number used in atomic state configNumber
     *      description
     * @tparam T_ConsoleOutput true =^= write result also to console
     */
    template<uint8_t T_n_max, bool T_consoleOutput = true>
    struct TestRateCalculation
    {
        static constexpr uint8_t numberLevels = 10u;
        static constexpr uint8_t atomicNumber = 4u;
        using ConfigNumberDataType = uint64_t;

        using S_ConfigNumber = picongpu::particles::atomicPhysics::stateRepresentation ::
            ConfigNumber<ConfigNumberDataType, numberLevels, atomicNumber>;

        static constexpr float_X energyElectron = 1000._X; // eV
        static constexpr float_X energyElectronBinWidth = 10._X; // eV
        static constexpr float_64 densityElectrons = 1.e28_X; // 1/(eV * m^3)

        using S_BoundBoundBuffer
            = atomicData::BoundBoundTransitionDataBuffer<uint32_t, float_X, uint32_t, uint64_t, tranOrd::byLowerState>;
        using S_BoundFreeBuffer = atomicData::BoundFreeTransitionDataBuffer<
            uint32_t,
            float_X,
            uint32_t,
            S_ConfigNumber,
            float_64,
            tranOrd::byLowerState>;
        using S_BoundBoundBox
            = atomicData::BoundBoundTransitionDataBox<uint32_t, float_X, uint32_t, uint64_t, tranOrd::byLowerState>;
        using S_BoundFreeBox = atomicData::
            BoundFreeTransitionDataBox<uint32_t, float_X, uint32_t, S_ConfigNumber, float_64, tranOrd::byLowerState>;

        using S_ChargeStateBuffer = atomicData::ChargeStateDataBuffer<uint32_t, float_X, S_ConfigNumber::atomicNumber>;
        using S_ChargeStateBox = atomicData::ChargeStateDataBox<uint32_t, float_X, S_ConfigNumber::atomicNumber>;

        using S_AtomicStateBuffer = atomicData::AtomicStateDataBuffer<uint32_t, float_X, S_ConfigNumber, float_64>;
        using S_AtomicStateBox = atomicData::AtomicStateDataBox<uint32_t, float_X, S_ConfigNumber, float_64>;

        std::unique_ptr<S_ChargeStateBuffer> chargeStateBuffer;
        std::unique_ptr<S_AtomicStateBuffer> atomicStateBuffer;
        std::unique_ptr<S_BoundBoundBuffer> boundBoundBuffer;
        std::unique_ptr<S_BoundFreeBuffer> boundFreeBuffer;

        TestRateCalculation()
        {
            // charge state already specifies number of entries
            chargeStateBuffer.reset(new S_ChargeStateBuffer());

            /// @attention number of states/transitions set for buffers needs to be == as actually added states in
            ///     setup()
            atomicStateBuffer.reset(new S_AtomicStateBuffer(5u));
            boundBoundBuffer.reset(new S_BoundBoundBuffer(1u));
            boundFreeBuffer.reset(new S_BoundFreeBuffer(2u));

            setup();
        }

    private:
        /** fill dataBuffers with small not physical example data
         *
         * dataBuffers are filled by hand to bypass checks of atomicData
         */
        void setup()
        {
            // charge states
            S_ChargeStateBox chargeStateHostBox = chargeStateBuffer->getHostDataBox();
            //      ionizationEnergy = 100 eV
            auto tupleChargeState_1 = std::make_tuple(u8(0u), 100._X);
            //      ionizationEnergy = 5 eV
            auto tupleChargeState_2 = std::make_tuple(u8(1u), 5._X);
            //      ionizationEnergy = 100 eV
            auto tupleChargeState_3 = std::make_tuple(u8(2u), 100._X);

            chargeStateHostBox.store(u8(0u), tupleChargeState_1);
            chargeStateHostBox.store(u8(1u), tupleChargeState_2);
            chargeStateHostBox.store(u8(2u), tupleChargeState_3);

            chargeStateBuffer->hostToDevice();

            /// atomic states, @attention caution all atomic state must differ in configNumber
            S_AtomicStateBox atomicStateHostBox = atomicStateBuffer->getHostDataBox();

            // 1:(1,1,0,0,0,0,1,0,1,0) lowerStateBoundFree, exictationEnergy = 0 eV(ground state), screenedCharge 5
            auto tupleAtomicState_bf_1 = std::make_tuple(static_cast<uint64_t>(243754u), 0._X, 5._X);
            // 2:(1,1,0,0,0,0,1,0,0,0) upperStateBoundFree, excitationEnergy = 5 eV, screenedCharge 5
            auto tupleAtomicState_bf_2 = std::make_tuple(static_cast<uint64_t>(9379u), 5._X, 5._X);
            // 3:(1,1,0,0,0,0,0,0,0,0) upperStateBoundFree, excitationEnergyDifference = 5 eV, screenedCharge 5
            auto tupleAtomicState_bf_3 = std::make_tuple(static_cast<uint64_t>(4u), 5._X, 5._X);

            /// @note states must be sorted primarily ascending by charge state, secondarily ascending by configNumber
            atomicStateHostBox.store(u8(1u), tupleAtomicState_bf_1);
            atomicStateHostBox.store(u8(3u), tupleAtomicState_bf_2);
            atomicStateHostBox.store(u8(4u), tupleAtomicState_bf_3);

            // 1:(1,0,2,0,0,0,1,0,0,0) lowerStateBoundBound
            auto tupleAtomicState_bb_1 = std::make_tuple(static_cast<uint64_t>(9406u), 0._X, 5._X);
            // 1:(1,0,1,0,0,0,1,0,1,0) upperStateBoundBound, energyDiffLowerUpper = 5 eV
            auto tupleAtomicState_bb_2 = std::make_tuple(static_cast<uint64_t>(243766u), 5._X, 5._X);

            /// @note states must be sorted primarily ascending by charge state, secondarily ascending by configNumber
            atomicStateHostBox.store(u8(0u), tupleAtomicState_bb_1);
            atomicStateHostBox.store(u8(2u), tupleAtomicState_bb_2);

            atomicStateBuffer->hostToDevice();

            // bound-bound transitions
            S_BoundBoundBox boundBoundHostBox = boundBoundBuffer->getHostDataBox();
            /* collisionalOscillatorStrength = 1, absorptionOscillatorStrength = 1e-1,
             *  cxin1 = 1., cxin2 = 2., cxin3 = 3., cxin4 = 4.cxin5 = 5.
             */
            auto tupleBoundBound = std::make_tuple(
                1._X,
                0.1_X,
                1._X,
                2._X,
                3._X,
                4._X,
                5._X,
                static_cast<uint64_t>(std::get<0>(tupleAtomicState_bb_1)),
                static_cast<uint64_t>(std::get<0>(tupleAtomicState_bb_2)));
            boundBoundHostBox.store(0u, tupleBoundBound, atomicStateHostBox);
            boundBoundBuffer->hostToDevice();

            // bound-free transition
            auto tupleBoundFree_1 = std::make_tuple(
                1._X,
                2._X,
                3._X,
                4._X,
                5._X,
                6._X,
                7._X,
                8._X,
                static_cast<uint64_t>(std::get<0>(tupleAtomicState_bf_1)),
                static_cast<uint64_t>(std::get<0>(tupleAtomicState_bf_2)));
            auto tupleBoundFree_2 = std::make_tuple(
                1._X,
                2._X,
                3._X,
                4._X,
                5._X,
                6._X,
                7._X,
                8._X,
                static_cast<uint64_t>(std::get<0>(tupleAtomicState_bf_2)),
                static_cast<uint64_t>(std::get<0>(tupleAtomicState_bf_3)));

            S_BoundFreeBox boundFreeHostBox = boundFreeBuffer->getHostDataBox();
            boundFreeHostBox.store(0u, tupleBoundFree_1, atomicStateHostBox);
            boundFreeHostBox.store(1u, tupleBoundFree_2, atomicStateHostBox);
            boundFreeBuffer->hostToDevice();
        }

    public:
        //! @return true =^= test passed
        bool testCollisionalExcitationCrossSection() const
        {
            float_X const correctCrossSection = 3.456217425189e+02; // 1e6b
            float_X const crossSection = rateCalculation::BoundBoundTransitionRates<T_n_max>::
                template collisionalBoundBoundCrossSection<S_AtomicStateBox, S_BoundBoundBox, true>(
                    energyElectron,
                    u32(0u),
                    atomicStateBuffer->getHostDataBox(),
                    boundBoundBuffer->getHostDataBox()); // 1e6b

            return testRelativeError<T_consoleOutput>(
                correctCrossSection,
                crossSection,
                "collisional excitation cross section",
                static_cast<float_X>(1e-5));
        }

        //! @return true =^= test passed
        bool testCollisionalDeexcitationCrossSection() const
        {
            float_X const energyElectron = 1000._X;
            float_X const correctCrossSection = 1.814666351842e+01; // 1e6b
            float_X const crossSection = rateCalculation::BoundBoundTransitionRates<T_n_max>::
                template collisionalBoundBoundCrossSection<S_AtomicStateBox, S_BoundBoundBox, false>(
                    energyElectron,
                    0u,
                    atomicStateBuffer->getHostDataBox(),
                    boundBoundBuffer->getHostDataBox()); // 1e6

            return testRelativeError<T_consoleOutput>(
                correctCrossSection,
                crossSection,
                "collisional deexcitation cross section",
                static_cast<float_X>(1.e-5));
        }

        //! @return true =^= test passed
        bool testCollisionalIonizationCrossSection() const
        {
            float_X const correctCrossSection = 8.051678880120e-01; // 1e6b
            float_X const crossSection = rateCalculation::BoundFreeCollisionalTransitionRates<T_n_max, true>::
                collisionalIonizationCrossSection(
                    // eV
                    energyElectron,
                    // ionization potential depression, eV
                    0._X,
                    0u,
                    chargeStateBuffer->getHostDataBox(),
                    atomicStateBuffer->getHostDataBox(),
                    boundFreeBuffer->getHostDataBox()); // 1e6b

            return testRelativeError<T_consoleOutput>(
                correctCrossSection,
                crossSection,
                "collisional ionization cross section",
                static_cast<float_X>(
                    1e-3)); /// @todo find out why error is larger than for de-/excitation, Brian Marre, 2023
        }

        //! @return true =^= test passed
        bool testCollisionalExcitationRate() const
        {
            float_64 const correctRate = 6.472768268762e+16; // 1/s
            float_64 const rate
                = static_cast<float_64>(
                      rateCalculation::BoundBoundTransitionRates<T_n_max>::
                          template rateCollisionalBoundBoundTransition<S_AtomicStateBox, S_BoundBoundBox, true>(
                              energyElectron,
                              energyElectronBinWidth,
                              // 1/(eV*m^3) * (m/sim.unit.length())^3 = = 1/(eV * sim.unit.length()^3)
                              static_cast<float_X>(
                                  densityElectrons * pmacc::math::cPow(picongpu::sim.unit.length(), 3u)),
                              0._X,
                              0u,
                              atomicStateBuffer->getHostDataBox(),
                              boundBoundBuffer->getHostDataBox()))
                  * 1. / sim.unit.time(); // 1/s

            return testRelativeError<T_consoleOutput>(correctRate, rate, "collisional excitation rate", 1e-5);
        }

        //! @return true =^= test passed
        bool testCollisionalDeexcitationRate() const
        {
            float_64 const correctRate = 3.398488386461e+15; // 1/s
            float_64 const rate
                = static_cast<float_64>(
                      rateCalculation::BoundBoundTransitionRates<T_n_max>::
                          template rateCollisionalBoundBoundTransition<S_AtomicStateBox, S_BoundBoundBox, false>(
                              energyElectron,
                              energyElectronBinWidth,
                              static_cast<float_X>(
                                  densityElectrons * pmacc::math::cPow(picongpu::sim.unit.length(), 3u)),
                              0._X,
                              0u,
                              atomicStateBuffer->getHostDataBox(),
                              boundBoundBuffer->getHostDataBox()))
                  * 1. / sim.unit.time(); // 1/s

            return testRelativeError<T_consoleOutput>(correctRate, rate, "collisional deexcitation rate", 1e-5);
        }

        //! @return true =^= test passed
        bool testSpontaneousRadiativeDeexcitationRate() const
        {
            float_64 const correctRate = 5.691850311676e+06; // 1/s
            float_64 const rate
                = static_cast<float_64>(
                      rateCalculation::BoundBoundTransitionRates<T_n_max>::rateSpontaneousRadiativeDeexcitation(
                          0u,
                          atomicStateBuffer->getHostDataBox(),
                          boundBoundBuffer->getHostDataBox()))
                  * 1. / sim.unit.time(); // 1/s

            return testRelativeError<T_consoleOutput>(
                correctRate,
                rate,
                "spontaneous radiative deexcitation rate",
                1e-5);
        }

        //! @return true =^= test passed, pass silently if correct
        bool testCollisionalIonizationRate() const
        {
            float_64 const correctRate = 1.507910098065e+14; // 1/s
            float_64 const rate
                = static_cast<float_64>(
                      rateCalculation::BoundFreeCollisionalTransitionRates<T_n_max, true>::
                          rateCollisionalIonizationTransition(
                              energyElectron,
                              energyElectronBinWidth,
                              static_cast<float_X>(
                                  densityElectrons * pmacc::math::cPow(picongpu::sim.unit.length(), 3u)),
                              0._X,
                              // ionization potential depression
                              0._X,
                              0u,
                              chargeStateBuffer->getHostDataBox(),
                              atomicStateBuffer->getHostDataBox(),
                              boundFreeBuffer->getHostDataBox()))
                  * 1. / sim.unit.time(); // 1/s

            //! @note larger error limit required due to numerics of rate formula
            return testRelativeError<T_consoleOutput>(correctRate, rate, "collisional ionization rate", 1e-3);
        }

        /** @return true =^= test passed
         *
         * reference value D_ref computed by hand from the Saha detailed balance factor
         *  D = g_lower/(2 * g_upper) * n_e * (2 * pi * hbar^2/(m_e * T_e))^(3/2) * exp(DeltaE/T_e)
         * with T_e = 100 eV, n_e = 1e28 1/m^3, DeltaE = 105 eV, g_lower/g_upper = 162
         * and 2022 CODATA values hbar = 1.054571817e-34 J*s, m_e = 9.1093837139e-31 kg
         */
        bool testThreeBodyDetailedBalanceFactor() const
        {
            // eV
            float_X const temperatureElectrons = 100._X;
            // 1/sim.unit.length()^3, = 1e28 1/m^3
            float_X const densityTotalElectrons
                = static_cast<float_X>(1.e28 * pmacc::math::cPow(picongpu::sim.unit.length(), 3u));
            // eV, = 5 eV excitation energy difference + 100 eV ionization energy of charge state 0
            float_X const deltaEnergyTransition = 105._X;

            // unitless
            float_64 const correctFactor = 7.668199883550e-01;
            float_64 const factor = rateCalculation::BoundFreeCollisionalTransitionRates<T_n_max, true>::
                threeBodyDetailedBalanceFactor(
                    temperatureElectrons,
                    densityTotalElectrons,
                    deltaEnergyTransition,
                    // g_lower/g_upper of the first bound-free test transition
                    162.,
                    1.);

            return testRelativeError<T_consoleOutput>(
                correctFactor,
                factor,
                "three-body recombination detailed balance factor",
                1e-4);
        }

        /** @return true =^= test passed
         *
         * checks the detailed balance identity rate_3BR == D * rate_EII against the hand computed D_ref, with the
         *  state multiplicities taken from the test data boxes, g_lower/g_upper = 162
         */
        bool testThreeBodyRecombinationRate() const
        {
            // eV
            float_X const temperatureElectrons = 100._X;
            // 1/sim.unit.length()^3, = 1e28 1/m^3
            float_X const densityTotalElectrons
                = static_cast<float_X>(1.e28 * pmacc::math::cPow(picongpu::sim.unit.length(), 3u));

            // 1/sim.unit.time()
            float_X const sumRateCollisionalIonization
                = rateCalculation::BoundFreeCollisionalTransitionRates<T_n_max, true>::
                    rateCollisionalIonizationTransition(
                        energyElectron,
                        energyElectronBinWidth,
                        static_cast<float_X>(densityElectrons * pmacc::math::cPow(picongpu::sim.unit.length(), 3u)),
                        0._X,
                        // ionization potential depression
                        0._X,
                        0u,
                        chargeStateBuffer->getHostDataBox(),
                        atomicStateBuffer->getHostDataBox(),
                        boundFreeBuffer->getHostDataBox());

            // 1/sim.unit.time()
            float_X const rate = rateCalculation::BoundFreeCollisionalTransitionRates<T_n_max, true>::
                rateCollisionalThreeBodyRecombinationTransition(
                    temperatureElectrons,
                    densityTotalElectrons,
                    sumRateCollisionalIonization,
                    // ionization potential depression
                    0._X,
                    0u,
                    chargeStateBuffer->getHostDataBox(),
                    atomicStateBuffer->getHostDataBox(),
                    boundFreeBuffer->getHostDataBox());

            // unitless, same D_ref as in testThreeBodyDetailedBalanceFactor
            float_64 const correctFactor = 7.668199883550e-01;

            return testRelativeError<T_consoleOutput>(
                correctFactor * static_cast<float_64>(sumRateCollisionalIonization),
                static_cast<float_64>(rate),
                "three-body recombination rate",
                1e-4);
        }

        //! @return true =^= test passed, checks input guards and the n_e scaling of the detailed balance factor
        bool testThreeBodyRecombinationGuards() const
        {
            using RateCalculator = rateCalculation::BoundFreeCollisionalTransitionRates<T_n_max, true>;

            // 1/sim.unit.length()^3, = 1e28 1/m^3
            float_X const densityTotalElectrons
                = static_cast<float_X>(1.e28 * pmacc::math::cPow(picongpu::sim.unit.length(), 3u));

            // barrier-free transition(strong IPD), zero temperature and zero density all have no defined factor
            bool const passBarrierFree
                = (RateCalculator::threeBodyDetailedBalanceFactor(100._X, densityTotalElectrons, -5._X, 162., 1.)
                   == 0.);
            bool const passZeroTemperature
                = (RateCalculator::threeBodyDetailedBalanceFactor(0._X, densityTotalElectrons, 105._X, 162., 1.)
                   == 0.);
            bool const passZeroDensity
                = (RateCalculator::threeBodyDetailedBalanceFactor(100._X, 0._X, 105._X, 162., 1.) == 0.);

            // exponent cap prevents overflow at low temperature
            bool const passLowTemperature = std::isfinite(
                RateCalculator::threeBodyDetailedBalanceFactor(1.e-6_X, densityTotalElectrons, 105._X, 162., 1.));

            // factor is linear in n_e, together with the density linearity of the summed collisional ionization
            //  rate this gives the physical n_e^2 scaling of the three-body recombination rate
            float_64 const factorSingleDensity = RateCalculator::threeBodyDetailedBalanceFactor(
                100._X,
                densityTotalElectrons,
                105._X,
                162.,
                1.);
            float_64 const factorDoubleDensity = RateCalculator::threeBodyDetailedBalanceFactor(
                100._X,
                2._X * densityTotalElectrons,
                105._X,
                162.,
                1.);
            bool const passDensityScaling = testRelativeError<false>(
                2. * factorSingleDensity,
                factorDoubleDensity,
                "three-body recombination density scaling",
                1e-6);

            bool const pass = passBarrierFree && passZeroTemperature && passZeroDensity && passLowTemperature
                && passDensityScaling;

            if constexpr(T_consoleOutput)
            {
                if(pass)
                    std::cout << "three-body recombination guards: * " << std::endl;
                else
                    std::cout << "three-body recombination guards: x  barrierFree: " << passBarrierFree
                              << " zeroTemperature: " << passZeroTemperature << " zeroDensity: " << passZeroDensity
                              << " lowTemperature: " << passLowTemperature
                              << " densityScaling: " << passDensityScaling << std::endl;
            }
            return pass;
        }

        /** @return true =^= test passed
         *
         * reference values from lib/python/picongpu/extra/utils/FLYonPICRateCalculationReference/, mock bound-free
         *  transitions have cxin1..8 = 1..8, i.e. fit amplitude cxin5 = 5, edge cxin6 = 6 eV, upper validity limit
         *  cxin8 = 8 eV
         */
        bool testScofieldPhotoIonizationCrossSection() const
        {
            // 1e6b, at E_gamma = 7 eV, inside [cxin6, cxin8]
            float_X const correctCrossSection = 4.571515907410e+01;
            float_X const crossSection = rateCalculation::BoundFreeRadiativeTransitionRates<T_n_max, true>::
                scofieldPhotoIonizationCrossSection(
                    // eV
                    7._X,
                    1u,
                    boundFreeBuffer->getHostDataBox());

            bool const passValue = testRelativeError<T_consoleOutput>(
                correctCrossSection,
                crossSection,
                "scofield photoionization cross section",
                static_cast<float_X>(1e-5));

            // fit is invalid above the upper validity limit cxin8
            bool const passUpperLimit
                = (rateCalculation::BoundFreeRadiativeTransitionRates<T_n_max, true>::
                       scofieldPhotoIonizationCrossSection(9._X, 1u, boundFreeBuffer->getHostDataBox())
                   == 0._X);

            if constexpr(T_consoleOutput)
                if(!passUpperLimit)
                    std::cout << "scofield cross section upper validity limit: x" << std::endl;

            return passValue && passUpperLimit;
        }

        //! @return true =^= test passed
        bool testKramersPhotoIonizationCrossSection() const
        {
            // 1e6b, at E_gamma = 205 eV, DeltaE = 105 eV, screenedCharge = 5, transitionMultiplicity = 162
            float_X const correctCrossSection = 1.243048283842e+01;
            float_X const crossSection = rateCalculation::BoundFreeRadiativeTransitionRates<T_n_max, true>::
                kramersPhotoIonizationCrossSection(
                    // eV
                    205._X,
                    // eV
                    105._X,
                    5._X,
                    162.);

            return testRelativeError<T_consoleOutput>(
                correctCrossSection,
                crossSection,
                "kramers photoionization cross section",
                static_cast<float_X>(1e-5));
        }

        /** @return true =^= test passed
         *
         * Milne relation cross section, fit path: mock transition 1 with DeltaE = 5 eV(ionization energy of charge
         *  state 1, no excitation energy difference), E_e = 2 eV -> E_gamma = 7 eV,
         *  g_lower/g_upper = 1568/16 = 98
         */
        bool testRadiativeRecombinationCrossSection() const
        {
            // 1e6b
            float_X const correctCrossSection = 1.074638581409e-01;
            float_X const crossSection = rateCalculation::BoundFreeRadiativeTransitionRates<T_n_max, true>::
                radiativeRecombinationCrossSection(
                    // eV
                    2._X,
                    // ionization potential depression, eV
                    0._X,
                    1u,
                    chargeStateBuffer->getHostDataBox(),
                    atomicStateBuffer->getHostDataBox(),
                    boundFreeBuffer->getHostDataBox());

            bool const passFit = testRelativeError<T_consoleOutput>(
                correctCrossSection,
                crossSection,
                "radiative recombination cross section(fit)",
                static_cast<float_X>(1e-4));

            /* fallback path: mock transition 0 has DeltaE = 105 eV > cxin8 = 8 eV -> Kramers fallback,
             *  E_e = 100 eV -> E_gamma = 205 eV, g_lower/g_upper = 254016/1568 = 162,
             *  transitionMultiplicity = 1, screenedCharge = 5 */
            // 1e6b
            float_X const correctCrossSectionFallback = 5.114530272410e-03;
            float_X const crossSectionFallback = rateCalculation::BoundFreeRadiativeTransitionRates<T_n_max, true>::
                radiativeRecombinationCrossSection(
                    // eV
                    100._X,
                    // ionization potential depression, eV
                    0._X,
                    0u,
                    chargeStateBuffer->getHostDataBox(),
                    atomicStateBuffer->getHostDataBox(),
                    boundFreeBuffer->getHostDataBox());

            bool const passFallback = testRelativeError<T_consoleOutput>(
                correctCrossSectionFallback,
                crossSectionFallback,
                "radiative recombination cross section(fallback)",
                static_cast<float_X>(1e-4));

            // zero electron energy and barrier free transitions have no cross section
            bool const passGuards
                = (rateCalculation::BoundFreeRadiativeTransitionRates<T_n_max, true>::
                       radiativeRecombinationCrossSection(
                           0._X,
                           0._X,
                           1u,
                           chargeStateBuffer->getHostDataBox(),
                           atomicStateBuffer->getHostDataBox(),
                           boundFreeBuffer->getHostDataBox())
                   == 0._X)
                && (rateCalculation::BoundFreeRadiativeTransitionRates<T_n_max, true>::
                        radiativeRecombinationCrossSection(
                            2._X,
                            // ionization potential depression larger than the threshold, eV
                            1000._X,
                            1u,
                            chargeStateBuffer->getHostDataBox(),
                            atomicStateBuffer->getHostDataBox(),
                            boundFreeBuffer->getHostDataBox())
                    == 0._X);

            if constexpr(T_consoleOutput)
                if(!passGuards)
                    std::cout << "radiative recombination cross section guards: x" << std::endl;

            return passFit && passFallback && passGuards;
        }

        //! @return true =^= test passed
        bool testRadiativeRecombinationRate() const
        {
            // 1/s, fit path bin rate at E_e = 2 eV, binWidth 10 eV, density 1e28 1/(eV m^3)
            float_64 const correctRate = 9.013674032225e+11;
            float_64 const rate
                = static_cast<float_64>(
                      rateCalculation::BoundFreeRadiativeTransitionRates<T_n_max, true>::
                          rateRadiativeRecombinationTransition(
                              // eV
                              2._X,
                              energyElectronBinWidth,
                              static_cast<float_X>(
                                  densityElectrons * pmacc::math::cPow(picongpu::sim.unit.length(), 3u)),
                              // ionization potential depression
                              0._X,
                              1u,
                              chargeStateBuffer->getHostDataBox(),
                              atomicStateBuffer->getHostDataBox(),
                              boundFreeBuffer->getHostDataBox()))
                  * 1. / sim.unit.time(); // 1/s

            return testRelativeError<T_consoleOutput>(correctRate, rate, "radiative recombination rate", 1e-3);
        }

        //! @return true =^= test passed
        bool testADKIonizationRate() const
        {
            // unit: 1/s
            float_64 const correctRate = 1.823335012e+10 * 1. / 3.3e-17;

            // unit: unit_eField
            float_X const eFieldNorm = 0.03 * sim.atomicUnit.eField() / sim.unit.eField();

            // unit: eV
            float_X const ipd = 0._X;

            constexpr auto laserPolarization
                = picongpu::particles::atomicPhysics::enums::ADKLaserPolarization::linearPolarization;
            // unit: 1/s
            float_64 const rate
                = static_cast<float_64>(
                      rateCalculation::BoundFreeFieldTransitionRates<laserPolarization>::rateADKFieldIonization<
                          float_X>(
                          eFieldNorm,
                          ipd,
                          u32(1u),
                          chargeStateBuffer->getHostDataBox(),
                          atomicStateBuffer->getHostDataBox(),
                          boundFreeBuffer->getHostDataBox()))
                  * 1. / sim.unit.time();

            /// @note larger error limit required due to numerics of ADK rate formula
            return testRelativeError<T_consoleOutput>(correctRate, rate, "ADK field ionization", 1e-5);
        }

        //! @return true =^= all tests passed
        bool testAll()
        {
            constexpr uint8_t numberTests = 15;
            bool pass[numberTests];
            pass[0] = testCollisionalExcitationCrossSection();
            pass[1] = testCollisionalDeexcitationCrossSection();
            pass[2] = testCollisionalIonizationCrossSection();
            pass[3] = testCollisionalExcitationRate();
            pass[4] = testCollisionalDeexcitationRate();
            pass[5] = testSpontaneousRadiativeDeexcitationRate();
            pass[6] = testCollisionalIonizationRate();
            pass[7] = testADKIonizationRate();
            pass[8] = testThreeBodyDetailedBalanceFactor();
            pass[9] = testThreeBodyRecombinationRate();
            pass[10] = testThreeBodyRecombinationGuards();
            pass[11] = testScofieldPhotoIonizationCrossSection();
            pass[12] = testKramersPhotoIonizationCrossSection();
            pass[13] = testRadiativeRecombinationCrossSection();
            pass[14] = testRadiativeRecombinationRate();

            bool passTotal = true;
            for(uint8_t i = 0u; i < numberTests; ++i)
            {
                passTotal = passTotal && pass[i];
            }

            if constexpr(T_consoleOutput)
            {
                std::cout << "Result:";
                if(passTotal)
                    std::cout << " * Success" << std::endl;
                else
                    std::cout << " x Fail" << std::endl;
            }
            std::cout << std::endl;
            return passTotal;
        }
    };
} // namespace picongpu::particles::atomicPhysics::debug
