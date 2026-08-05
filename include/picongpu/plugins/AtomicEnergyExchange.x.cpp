/* Copyright 2026 Prashant Sharma
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

/** @file process-resolved atomic-energy exchange and escaped-radiation diagnostics */

#include "picongpu/defines.hpp"
#include "picongpu/particles/atomicPhysics/ParticleType.hpp"
#include "picongpu/particles/atomicPhysics/localHelperFields/AtomicEnergyExchange.hpp"
#include "picongpu/particles/atomicPhysics/localHelperFields/AtomicEnergyExchangeField.hpp"
#include "picongpu/particles/traits/GenerateSolversIfSpeciesEligible.hpp"
#include "picongpu/particles/traits/SpeciesEligibleForSolver.hpp"
#include "picongpu/plugins/PluginRegistry.hpp"
#include "picongpu/plugins/common/txtFileHandling.hpp"
#include "picongpu/plugins/multi/multi.hpp"

#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/math/operation.hpp>
#include <pmacc/mpi/MPIReduce.hpp>
#include <pmacc/mpi/reduceMethods/Reduce.hpp>
#include <pmacc/particles/meta/FindByNameOrType.hpp>
#include <pmacc/traits/HasIdentifiers.hpp>

#include <array>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>

namespace picongpu
{
    template<typename T_Particles>
    class AtomicEnergyExchange : public plugins::multi::IInstance
    {
    public:
        using Particles = T_Particles;
        using LedgerEntry = particles::atomicPhysics::localHelperFields::AtomicEnergyExchange;
        static constexpr uint32_t numberProcesses = LedgerEntry::numberProcesses;

        struct Help : public plugins::multi::IHelp
        {
            std::shared_ptr<IInstance> create(
                std::shared_ptr<IHelp>& help,
                size_t const id,
                MappingDesc* cellDescription) override
            {
                return std::shared_ptr<IInstance>(new AtomicEnergyExchange<Particles>(help, id, cellDescription));
            }

            plugins::multi::Option<std::string> notifyPeriod
                = {"period", "write cumulative atomic-energy exchange and escaped radiation [for each n-th step]"};

            void registerHelp(
                boost::program_options::options_description& desc,
                std::string const& masterPrefix = std::string{}) override
            {
                notifyPeriod.registerHelp(desc, masterPrefix + prefix);
            }

            void expandHelp(
                boost::program_options::options_description&,
                std::string const& = std::string{}) override
            {
            }

            void validateOptions() override
            {
            }

            size_t getNumPlugins() const override
            {
                return notifyPeriod.size();
            }

            std::string getDescription() const override
            {
                return description;
            }

            std::string getOptionPrefix() const
            {
                return prefix;
            }

            std::string getName() const override
            {
                return name;
            }

            std::string const name = "AtomicEnergyExchange";
            std::string const description = "process-resolved atomic energy and escaped radiation";
            std::string const prefix = Particles::FrameType::getName() + std::string("_atomicEnergyExchange");
        };

        static std::shared_ptr<plugins::multi::IHelp> getHelp()
        {
            return std::shared_ptr<plugins::multi::IHelp>(new Help{});
        }

        AtomicEnergyExchange(
            std::shared_ptr<plugins::multi::IHelp>& help,
            size_t const id,
            MappingDesc* cellDescription)
            : m_cellDescription(cellDescription)
            , m_help(std::static_pointer_cast<Help>(help))
            , m_id(id)
        {
            std::string const speciesName = Particles::FrameType::getName();
            atomicFilename = speciesName + "_atomic_energy_exchange_by_process.dat";
            radiationFilename = speciesName + "_radiated_energy.dat";
            writeToFile = reduce.hasResult(mpi::reduceMethods::Reduce());

            if(writeToFile)
            {
                atomicFile.open(atomicFilename.c_str(), std::ofstream::out | std::ostream::trunc);
                radiationFile.open(radiationFilename.c_str(), std::ofstream::out | std::ostream::trunc);
                if(!atomicFile || !radiationFile)
                {
                    std::cerr << "Can't open atomic-energy diagnostic files; disable plugin output.\n";
                    writeToFile = false;
                }
                else
                {
                    atomicFile
                        << "# Cumulative signed energy changes from accepted events. Bound-free entries use the "
                           "same dynamic IPD-shifted transition energies as the solver.\n";
                    atomicFile
                        << "#step spontaneousDeexcitation_J electronicExcitation_J electronicDeexcitation_J "
                           "electronicIonization_J autonomousIonization_J fieldIonization_J ipdIonization_J "
                           "threeBodyRecombination_J radiativeRecombination_J total_atomic_exchange_J\n";
                    radiationFile
                        << "# Cumulative escaped photon energy. RR uses the selected electron-histogram bin "
                           "energy plus the IPD-shifted binding energy.\n";
                    radiationFile << "#step spontaneousDeexcitation_J radiativeRecombination_J total_radiated_J\n";
                }
            }

            Environment<>::get().PluginConnector().setNotificationPeriod(this, m_help->notifyPeriod.get(id));
        }

        ~AtomicEnergyExchange() override
        {
            if(atomicFile.is_open())
                atomicFile.close();
            if(radiationFile.is_open())
                radiationFile.close();
        }

        void notify(uint32_t currentStep) override
        {
            calculate(currentStep);
        }

        void restart(uint32_t restartStep, std::string const& restartDirectory) override
        {
            if(!writeToFile)
                return;

            readAtomicRestartOffset(restartStep, restartDirectory);
            readRadiationRestartOffset(restartStep, restartDirectory);
            writeToFile = restoreTxtFile(atomicFile, atomicFilename, restartStep, restartDirectory)
                          && restoreTxtFile(radiationFile, radiationFilename, restartStep, restartDirectory);
        }

        void checkpoint(uint32_t currentStep, std::string const& checkpointDirectory) override
        {
            if(writeToFile)
            {
                checkpointTxtFile(atomicFile, atomicFilename, currentStep, checkpointDirectory);
                checkpointTxtFile(radiationFile, radiationFilename, currentStep, checkpointDirectory);
            }
        }

    private:
        static std::string checkpointPath(
            std::string const& directory,
            std::string const& filename,
            uint32_t const step)
        {
            return directory + "/" + filename + "." + std::to_string(step);
        }

        static bool getLastDataLine(std::string const& path, std::string& lastLine)
        {
            std::ifstream input(path);
            if(!input)
                return false;
            std::string line;
            while(std::getline(input, line))
                if(!line.empty() && line.front() != '#')
                    lastLine = line;
            return !lastLine.empty();
        }

        void readAtomicRestartOffset(uint32_t const step, std::string const& directory)
        {
            std::string line;
            if(!getLastDataLine(checkpointPath(directory, atomicFilename, step), line))
                return;
            std::istringstream values(line);
            uint32_t fileStep = 0u;
            values >> fileStep;
            for(uint32_t process = 1u; process < numberProcesses; ++process)
                values >> atomicRestartOffset[process];
        }

        void readRadiationRestartOffset(uint32_t const step, std::string const& directory)
        {
            std::string line;
            if(!getLastDataLine(checkpointPath(directory, radiationFilename, step), line))
                return;
            std::istringstream values(line);
            uint32_t fileStep = 0u;
            values >> fileStep;
            values >> radiationRestartOffset[1u] >> radiationRestartOffset[9u];
        }

        void calculate(uint32_t const currentStep)
        {
            using Field = particles::atomicPhysics::localHelperFields::AtomicEnergyExchangeField<
                picongpu::MappingDesc,
                Particles>;
            DataConnector& dc = Environment<>::get().DataConnector();
            auto field = dc.get<Field>(Particles::FrameType::getName() + "_atomicEnergyExchangeField");
            field->synchronize();

            std::array<float_64, numberProcesses> localAtomic{};
            std::array<float_64, numberProcesses> localRadiated{};
            auto const hostBox = field->superCellField->getHostBuffer().getDataBox();
            auto const gridSuperCells = m_cellDescription->getGridSuperCellsWithoutGuards();
            int const numberSuperCells = gridSuperCells.productOfComponents();
            for(int linearIdx = 0; linearIdx < numberSuperCells; ++linearIdx)
            {
                auto const idx = pmacc::math::mapToND(gridSuperCells, linearIdx);
                auto const& entry = hostBox(idx);
                for(uint32_t process = 1u; process < numberProcesses; ++process)
                {
                    localAtomic[process] += entry.getAtomicEnergy(process);
                    localRadiated[process] += entry.getRadiatedEnergy(process);
                }
            }

            std::array<float_64, numberProcesses> globalAtomic{};
            std::array<float_64, numberProcesses> globalRadiated{};
            reduce(
                pmacc::math::operation::Add(),
                globalAtomic.data(),
                localAtomic.data(),
                numberProcesses,
                mpi::reduceMethods::Reduce());
            reduce(
                pmacc::math::operation::Add(),
                globalRadiated.data(),
                localRadiated.data(),
                numberProcesses,
                mpi::reduceMethods::Reduce());

            if(!writeToFile)
                return;

            using dbl = std::numeric_limits<float_64>;
            atomicFile.precision(dbl::digits10);
            radiationFile.precision(dbl::digits10);

            float_64 totalAtomic = 0.0;
            atomicFile << currentStep << std::scientific;
            for(uint32_t process = 1u; process < numberProcesses; ++process)
            {
                float_64 const valueJ
                    = sim.si.conv().eV2Joule(globalAtomic[process]) + atomicRestartOffset[process];
                totalAtomic += valueJ;
                atomicFile << " " << valueJ;
            }
            atomicFile << " " << totalAtomic << std::endl;

            float_64 const spontaneousJ
                = sim.si.conv().eV2Joule(globalRadiated[1u]) + radiationRestartOffset[1u];
            float_64 const radiativeRecombinationJ
                = sim.si.conv().eV2Joule(globalRadiated[9u]) + radiationRestartOffset[9u];
            radiationFile << currentStep << " " << std::scientific << spontaneousJ << " "
                          << radiativeRecombinationJ << " " << spontaneousJ + radiativeRecombinationJ
                          << std::endl;
        }

        MappingDesc* m_cellDescription;
        std::string atomicFilename;
        std::string radiationFilename;
        std::ofstream atomicFile;
        std::ofstream radiationFile;
        bool writeToFile = false;
        mpi::MPIReduce reduce;
        std::shared_ptr<Help> m_help;
        size_t m_id;
        std::array<float_64, numberProcesses> atomicRestartOffset{};
        std::array<float_64, numberProcesses> radiationRestartOffset{};
    };

    namespace particles::traits
    {
        template<typename T_Species, typename T_UnspecifiedSpecies>
        struct SpeciesEligibleForSolver<T_Species, AtomicEnergyExchange<T_UnspecifiedSpecies>>
        {
            using FrameType = typename T_Species::FrameType;
            using RequiredIdentifiers = MakeSeq_t<weighting, atomicStateCollectionIndex>;
            using SpeciesHasIdentifiers = typename pmacc::traits::HasIdentifiers<FrameType, RequiredIdentifiers>::type;
            using IsAtomicIon = atomicPhysics::traits::IsParticleType_t<
                atomicPhysics::traits::GetParticleType_t<FrameType>,
                atomicPhysics::Tags::Ion>;
            using type = pmacc::mp_and<SpeciesHasIdentifiers, IsAtomicIon>;
        };
    } // namespace particles::traits
} // namespace picongpu

PIC_REGISTER_SPECIES_PLUGIN(picongpu::plugins::multi::Master<picongpu::AtomicEnergyExchange<boost::mpl::_1>>);
