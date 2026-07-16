#!/usr/bin/env python
"""
atomicPhysics(FLYonPIC) reference rate calculation
This file is part of the PIConGPU.
Copyright 2023-2026 PIConGPU contributors
Authors: Brian Marre, Prashant Sharma
License: GPLv3+
"""

import numpy as np
import scipy.constants as const
import scipy.special as scipy

""" @file reference implementation of the rate calculation for bound-free radiative transitions

Spontaneous radiative recombination: upperState + free electron -> lowerState + photon

The recombination cross section is the Milne(detailed balance) relation applied to the photoionization
cross section of the same bound-free transition, following the SCFLY/FLYCHK conventions.
The photoionization cross section is the Scofield-type fit stored in the bound-free transition input
data(cxin1..cxin8), with a hydrogenic Kramers fallback chosen such that the Maxwellian average
reproduces SCFLY's hydrogenic recombination rate coefficient "alfxx" exactly.
"""

# cm/(s*eV^0.5), Maxwellian f(E)*v(E) coefficient, v = sqrt(2E/m_e) nonrelativistic
_F_V = 2.0 / np.sqrt(np.pi) * np.sqrt(2.0 * const.elementary_charge / const.electron_mass) * 100.0

# 1/eV, Milne relation prefactor, from the SCFLY spontaneous recombination constants,
#  numerically equal to the textbook 1/(2 m_e c^2) within 0.06%
MILNE_CONSTANT = 1.656415e-22 * 2.4179e14 * 1.636e9 / _F_V

# cm^2*eV^0.5, Kramers fallback constant, reproduces SCFLY's alfxx under Maxwellian averaging
KRAMERS_CONSTANT_CM2 = 1.917e-15 / (_F_V * MILNE_CONSTANT)


class BoundFreeRadiativeTransitions:
    @staticmethod
    def _stateMultiplicity(levelVector):
        """statistical weight of an atomic state given by its occupation number level vector"""
        result = 1.0
        for i, occupation in enumerate(levelVector):
            n = i + 1
            result *= scipy.comb(2 * n**2, occupation)
        return result

    @staticmethod
    def scofieldPhotoIonizationCrossSection(energyPhoton, cxin1, cxin2, cxin3, cxin4, cxin5, cxin6, cxin8):
        """photoionization cross section fit of a bound-free transition, SCFLY/Scofield-type

        @param energyPhoton float photon energy, [eV]
        @param cxin1 ... cxin4 float fit polynomial coefficients
        @param cxin5 float fit amplitude
        @param cxin6 float fit edge energy, [eV]
        @param cxin8 float upper validity limit, [eV]

        @return unit: 1e6b
        """
        if energyPhoton <= 0.0 or cxin6 <= 0.0:
            return 0.0
        if energyPhoton > cxin8:
            return 0.0

        L = max(np.log(energyPhoton / cxin6), 0.0)
        # 1e-18 cm^2 = 1e6b, the fit's natural unit
        return cxin5 * np.exp(cxin1 + L * (cxin2 + L * (cxin3 + L * cxin4))) * (13.606 / cxin6)

    @staticmethod
    def kramersPhotoIonizationCrossSection(energyPhoton, deltaEnergyTransition, screenedCharge, transitionMultiplicity):
        """hydrogenic Kramers photoionization cross section, fallback for transitions without a valid fit

        @param energyPhoton float photon energy, [eV]
        @param deltaEnergyTransition float IPD-shifted energy difference of the transition, [eV]
        @param screenedCharge float screened charge of the lower(recombined) state, [e]
        @param transitionMultiplicity float combinatorial multiplicity of the bound-free transition

        @return unit: 1e6b
        """
        if energyPhoton <= 0.0 or deltaEnergyTransition <= 0.0 or screenedCharge <= 0.0:
            return 0.0

        # cm^2 / (1e-18 cm^2/1e6b) = 1e6b
        return (
            KRAMERS_CONSTANT_CM2
            * transitionMultiplicity
            * deltaEnergyTransition**2.5
            / (screenedCharge * energyPhoton**3)
        ) / 1.0e-18

    @staticmethod
    def photoIonizationCrossSection(
        energyPhoton,
        deltaEnergyTransition,
        screenedCharge,
        transitionMultiplicity,
        cxin1,
        cxin2,
        cxin3,
        cxin4,
        cxin5,
        cxin6,
        cxin8,
    ):
        """photoionization cross section, fit with hydrogenic fallback

        fit is valid if cxin5 > 1e-30 and cxin6 > 0 and cxin8 > deltaEnergyTransition, following SCFLY

        @return unit: 1e6b
        """
        fitValid = (cxin5 > 1.0e-30) and (cxin6 > 0.0) and (cxin8 > deltaEnergyTransition)
        if fitValid:
            return BoundFreeRadiativeTransitions.scofieldPhotoIonizationCrossSection(
                energyPhoton, cxin1, cxin2, cxin3, cxin4, cxin5, cxin6, cxin8
            )
        return BoundFreeRadiativeTransitions.kramersPhotoIonizationCrossSection(
            energyPhoton, deltaEnergyTransition, screenedCharge, transitionMultiplicity
        )

    @staticmethod
    def radiativeRecombinationCrossSection(
        energyElectron,
        deltaEnergyTransition,
        screenedCharge,
        lowerStateLevelVector,
        upperStateLevelVector,
        transitionMultiplicity,
        cxin1,
        cxin2,
        cxin3,
        cxin4,
        cxin5,
        cxin6,
        cxin8,
    ):
        """spontaneous radiative recombination cross section via the Milne relation

        sigma_RR(E_e) = MILNE_CONSTANT * g_lower/g_upper * E_gamma^2/E_e * sigma_PI(E_gamma),
        E_gamma = E_e + deltaEnergyTransition

        @param energyElectron float kinetic energy of the captured free electron, [eV]
        @param deltaEnergyTransition float IPD-shifted energy difference of the transition, [eV]
        @param screenedCharge float screened charge of the lower(recombined) state, [e]
        @param lowerStateLevelVector occupation number level vector of the lower(recombined) state
        @param upperStateLevelVector occupation number level vector of the upper(ionized) state
        @param transitionMultiplicity float combinatorial multiplicity of the bound-free transition

        @return unit: 1e6b
        """
        if energyElectron <= 0.0 or deltaEnergyTransition <= 0.0:
            return 0.0

        energyPhoton = energyElectron + deltaEnergyTransition

        sigmaPhotoIonization = BoundFreeRadiativeTransitions.photoIonizationCrossSection(
            energyPhoton,
            deltaEnergyTransition,
            screenedCharge,
            transitionMultiplicity,
            cxin1,
            cxin2,
            cxin3,
            cxin4,
            cxin5,
            cxin6,
            cxin8,
        )

        degeneracyRatio = BoundFreeRadiativeTransitions._stateMultiplicity(
            lowerStateLevelVector
        ) / BoundFreeRadiativeTransitions._stateMultiplicity(upperStateLevelVector)

        # 1/eV * unitless * eV^2/eV * 1e6b = 1e6b
        return MILNE_CONSTANT * degeneracyRatio * energyPhoton**2 / energyElectron * sigmaPhotoIonization

    @staticmethod
    def rateRadiativeRecombination(
        energyElectron,
        energyElectronBinWidth,
        densityElectrons,
        deltaEnergyTransition,
        screenedCharge,
        lowerStateLevelVector,
        upperStateLevelVector,
        transitionMultiplicity,
        cxin1,
        cxin2,
        cxin3,
        cxin4,
        cxin5,
        cxin6,
        cxin8,
    ):
        """rate of spontaneous radiative recombination for one free electron bin

        @param energyElectron float central energy of electron bin, [eV]
        @param energyElectronBinWidth float width of energy bin, [eV]
        @param densityElectrons float number density of physical electrons in bin, 1/(m^3 * eV)

        @return unit: 1/s
        """
        sigma = BoundFreeRadiativeTransitions.radiativeRecombinationCrossSection(
            energyElectron,
            deltaEnergyTransition,
            screenedCharge,
            lowerStateLevelVector,
            upperStateLevelVector,
            transitionMultiplicity,
            cxin1,
            cxin2,
            cxin3,
            cxin4,
            cxin5,
            cxin6,
            cxin8,
        )  # 1e6b

        electronRestMassEnergy = const.value("electron mass energy equivalent in MeV") * 1e6  # eV

        # dE * sigma(E) * rho_e * v, same relativistic velocity as the collisional reference
        return (
            energyElectronBinWidth
            * sigma
            * 1e-22
            * densityElectrons
            * const.value("speed of light in vacuum")
            * np.sqrt(1.0 - 1.0 / (1.0 + energyElectron / electronRestMassEnergy) ** 2)
        )  # 1/s
