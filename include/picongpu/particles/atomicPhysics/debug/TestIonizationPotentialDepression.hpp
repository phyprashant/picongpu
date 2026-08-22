/* Copyright 2025 Brian Marre
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

/** @file unit tests for the Ionization Potential Depression(IPD) calculation
 *
 * test are activated by the global debug switch debug::ionizationPotentialDepression::RUN_UNIT_TESTS
 *  in atomicPhysics_Debug.param
 *
 * for updating the tests see the python [rate calculator tool](
 *  https://github.com/BrianMarre/picongpuAtomicPhysicsTools/tree/dev/RateCalculationReference)
 */

#pragma once

#include "picongpu/defines.hpp"
// need unit.param

#include "picongpu/particles/atomicPhysics/ConvertEnum.hpp"
#include "picongpu/particles/atomicPhysics/debug/TestRelativeError.hpp"

#include <cmath>
#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/RelativisticTemperatureFunctor.hpp"
#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/StewartPyattIPD.hpp"

namespace picongpu::particles::atomicPhysics::debug
{
    template<bool T_consoleOutput = true>
    struct TestIonizationPotentialDepression
    {
        //! @return true =^= test passed
        bool testStewartPyattIPD() const
        {
            // 1/m^3
            float_64 const electronDensity = 4.e28;

            // eV
            float_64 const temperatureTimesk_Boltzman = 1.e3;

            uint8_t const chargeState = 2u;
            float_64 const zStar = static_cast<float_64>(chargeState);
            float_64 const unit_length = static_cast<float_64>(sim.unit.length());

            // sqrt(UNIT_CHARGE^2 * UNIT_TIME^2 * / UNIT_LENGTH^3 / UNIT_MASS * UNIT_ENERGY
            //  * 1/( UNIT_CHARGE^2 * 1/m^3 * (m/UNIT_LENGTH)^3))
            //  = sqrt(1/UNIT_ENERGY * 1/UNIT_LENGTH * UNIT_ENERGY * UNIT_LENGTH^3) = UNIT_LENGTH
            float_64 const debyeLength = std::sqrt(
                sim.pic.getEps0<float_64>() * sim.pic.conv().eV2Joule<float_64>(temperatureTimesk_Boltzman)
                / (sim.pic.getElectronCharge<float_64>() * sim.pic.getElectronCharge<float_64>() * electronDensity
                   * (unit_length * unit_length * unit_length) * static_cast<float_64>(chargeState + 1)));

            // eV
            float_64 const correctIPDValue = 6.306390823271927;

            using StewartPyattIPD = particles::atomicPhysics::ionizationPotentialDepression::template StewartPyattIPD<
                particles::atomicPhysics::ionizationPotentialDepression::RelativisticTemperatureFunctor,
                false>;

            auto const superCellConstantInput = StewartPyattIPD::SuperCellConstantInput{
                static_cast<float_X>(temperatureTimesk_Boltzman),
                static_cast<float_X>(debyeLength),
                static_cast<float_X>(zStar),
                static_cast<float_X>(electronDensity * (sim.unit.length() * sim.unit.length() * sim.unit.length()))};

            // eV
            float_64 const ipd = static_cast<float_X>(StewartPyattIPD::ipd(superCellConstantInput, chargeState));

            return testRelativeError<T_consoleOutput>(
                correctIPDValue,
                ipd,
                "Stewart-Pyatt ionization potential depression",
                1e-5);
        }

        /** SCFLY-extended Stewart-Pyatt at a given temperature
         *
         * @param electronDensity 1/m^3
         * @param temperatureTimesk_Boltzman eV, may be exactly 0
         * @param chargeState charge state before ionization
         * @return eV
         */
        static float_64 scflyExtendedIPD(
            float_64 const electronDensity,
            float_64 const temperatureTimesk_Boltzman,
            uint8_t const chargeState)
        {
            float_64 const unit_length = static_cast<float_64>(sim.unit.length());
            float_64 const densityPIC = electronDensity * (unit_length * unit_length * unit_length);

            // UNIT_LENGTH; exactly 0 for T == 0, which is the case under test
            float_64 const debyeLength = std::sqrt(
                sim.pic.getEps0<float_64>() * sim.pic.conv().eV2Joule<float_64>(temperatureTimesk_Boltzman)
                / (sim.pic.getElectronCharge<float_64>() * sim.pic.getElectronCharge<float_64>() * electronDensity
                   * (unit_length * unit_length * unit_length) * static_cast<float_64>(chargeState + 1)));

            using SCFLYStewartPyattIPD =
                particles::atomicPhysics::ionizationPotentialDepression::template StewartPyattIPD<
                    particles::atomicPhysics::ionizationPotentialDepression::RelativisticTemperatureFunctor,
                    true>;

            auto const input = SCFLYStewartPyattIPD::SuperCellConstantInput{
                static_cast<float_X>(temperatureTimesk_Boltzman),
                static_cast<float_X>(debyeLength),
                static_cast<float_X>(chargeState),
                static_cast<float_X>(densityPIC)};

            return static_cast<float_64>(SCFLYStewartPyattIPD::ipd(input, chargeState));
        }

        /** cold limit: T == 0 must give the finite ion-sphere value, not 0
         *
         * SCFLY scdrv.f:3537,  dE(T=0) = 2.16e-7[eV cm] * z / r_ion,  z = chargeState + 1,
         *  r_ion = (0.75 * z / (pi * n_e))^(1/3).
         * The K-based Stewart-Pyatt branch returns exactly 0 here, which suppressed IPD entirely
         *  in cold dense plasma and left high-n Rydberg states available to recombination.
         */
        bool testStewartPyattIPDColdLimit() const
        {
            // 1/m^3, solid-density Si3+ conditions
            float_64 const electronDensity = 2.4e30;
            uint8_t const chargeState = 3u;
            float_64 const z = static_cast<float_64>(chargeState + 1u);

            // m
            float_64 const ionSphereRadius = std::pow(0.75 * z / (3.14159265358979323846 * electronDensity), 1. / 3.);
            // eV; 2.16e-7 eV*cm = 2.16e-9 eV*m
            float_64 const correctIPDValue = 2.16e-9 * z / ionSphereRadius;

            float_64 const ipd = scflyExtendedIPD(electronDensity, 0., chargeState);

            return testRelativeError<T_consoleOutput>(
                correctIPDValue,
                ipd,
                "Stewart-Pyatt IPD, cold limit (T = 0)",
                1e-2);
        }

        //! low temperature must converge to the cold limit, not collapse to 0
        bool testStewartPyattIPDLowTemperatureConvergence() const
        {
            float_64 const electronDensity = 2.4e30;
            uint8_t const chargeState = 3u;

            float_64 const coldValue = scflyExtendedIPD(electronDensity, 0., chargeState);
            float_64 const nearColdValue = scflyExtendedIPD(electronDensity, 1.e-3, chargeState);

            return testRelativeError<T_consoleOutput>(
                coldValue,
                nearColdValue,
                "Stewart-Pyatt IPD, low temperature converges to cold limit",
                1e-2);
        }

        /** finite temperature must stay on the Debye-Huckel branch
         *
         * For lambda_D >> r_ion the ion-sphere bracket tends to 2/(3r), so
         *  dE -> 1.44e-7[eV cm] * z / lambda_D, and the min() picks the 1.45e-7 sqrt branch.
         * This guards against the cold-limit fix leaking into the hot regime.
         */
        bool testStewartPyattIPDFiniteTemperature() const
        {
            float_64 const electronDensity = 4.e28;
            uint8_t const chargeState = 2u;

            float_64 const hotValue = scflyExtendedIPD(electronDensity, 1.e3, chargeState);
            float_64 const coldValue = scflyExtendedIPD(electronDensity, 0., chargeState);

            // hot IPD must be strictly below the cold ion-sphere ceiling, and strictly positive
            bool const pass = (hotValue > 0.) && (hotValue < coldValue);

            if constexpr(T_consoleOutput)
                std::cout << "  Stewart-Pyatt IPD, finite temperature below cold ceiling: " << hotValue << " eV < "
                          << coldValue << " eV -> " << (pass ? "pass" : "FAIL") << std::endl;
            return pass;
        }

        //! zero electron density must not produce NaN/Inf
        bool testStewartPyattIPDZeroDensity() const
        {
            float_64 const ipd = scflyExtendedIPD(1.e-30, 0., 1u);
            bool const pass = std::isfinite(ipd) && (ipd >= 0.);

            if constexpr(T_consoleOutput)
                std::cout << "  Stewart-Pyatt IPD, vanishing density finite: " << ipd << " eV -> "
                          << (pass ? "pass" : "FAIL") << std::endl;
            return pass;
        }

        bool testAll()
        {
            bool passTotal = testStewartPyattIPD();
            passTotal = testStewartPyattIPDColdLimit() && passTotal;
            passTotal = testStewartPyattIPDLowTemperatureConvergence() && passTotal;
            passTotal = testStewartPyattIPDFiniteTemperature() && passTotal;
            passTotal = testStewartPyattIPDZeroDensity() && passTotal;

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
