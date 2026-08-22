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
#include "picongpu/particles/atomicPhysics/atomicData/AtomicData.hpp"
#include "picongpu/particles/atomicPhysics/localHelperFields/AtomicEnergyExchange.hpp"
#include "picongpu/particles/atomicPhysics/localHelperFields/AtomicEnergyExchangeField.hpp"
#include "picongpu/particles/traits/GenerateSolversIfSpeciesEligible.hpp"
#include "picongpu/particles/traits/GetAtomicDataType.hpp"
#include "picongpu/particles/traits/SpeciesEligibleForSolver.hpp"
#include "picongpu/plugins/PluginRegistry.hpp"
#include "picongpu/plugins/atomicPhysics/KernelAtomicInternalEnergy.hpp"
#include "picongpu/plugins/common/txtFileHandling.hpp"
#include "picongpu/plugins/multi/multi.hpp"

#include <pmacc/communication/manager_common.hpp>
#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
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
#include <stdexcept>
#include <string>
#include <vector>

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
            referenceEnergyBuffer = std::make_unique<GridBuffer<float_64, DIM1>>(DataSpace<DIM1>(2));

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
                        << "# externalReconciliation_J covers charge state changes made outside atomicPhysics that "
                           "left the atomic state index valid; unperturbed table energies, no IPD shift.\n";
                    atomicFile
                        << "# externalIonizationWeight is macro ion weight that arrived from an ionization module "
                           "outside atomicPhysics (BSI/ADK/Keldysh/ThomasFermi). SetChargeState destroys their "
                           "previous atomic state before atomicPhysics sees them, so their internal energy is NOT "
                           "in this ledger. Any non-zero value means every energy column below is incomplete.\n";
                    atomicFile
                        << "# internalEnergy_J is the absolute atomic internal energy: the reference energy taken "
                           "once at the first output step, advanced from there by the ledger. This is the column to "
                           "sum with the EnergyParticles/EnergyFields diagnostics.\n";
                    atomicFile
                        << "# referenceEnergy_J is the unperturbed-table energy of the current ion population, with "
                           "no IPD shift. Do not difference it in time for a budget; its excess over "
                           "internalEnergy_J measures the continuum-lowering reservoir.\n";
                    atomicFile
                        << "#step spontaneousDeexcitation_J electronicExcitation_J electronicDeexcitation_J "
                           "electronicIonization_J autonomousIonization_J fieldIonization_J ipdIonization_J "
                           "threeBodyRecombination_J radiativeRecombination_J externalReconciliation_J "
                           "total_atomic_exchange_J internalEnergy_J referenceEnergy_J externalIonizationWeight\n";
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

        /** flush a data line for the checkpoint step itself
         *
         * restart() recovers the accumulators from the last data line of the checkpointed file, so that line has to
         * belong to the checkpoint step. If the notify period does not divide the checkpoint period the newest line
         * is older than the checkpoint, and everything the ledger recorded in between is dropped on restart: the
         * device accumulators are not checkpointed and come back zeroed.
         *
         * @attention collective, calculate() reduces; every rank must reach this.
         */
        void onCheckpointStep(uint32_t const currentStep)
        {
            if(currentStep != lastWrittenStep)
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
            // collective, must be outside the master-only guard below
            onCheckpointStep(currentStep);

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

        /** number of columns in a data line of the by-process file
         *
         * step, one per process, then externalReconciliation_J, total_atomic_exchange_J, internalEnergy_J,
         * referenceEnergy_J, externalIonizationWeight.
         */
        static constexpr uint32_t numberAtomicColumns = 1u + (numberProcesses - 1u) + 5u;

        void readAtomicRestartOffset(uint32_t const step, std::string const& directory)
        {
            std::string line;
            if(!getLastDataLine(checkpointPath(directory, atomicFilename, step), line))
                return;

            /* Counted before anything is consumed. A file written by an older revision shares all of its leading
             * columns with this schema and differs only in the trailing ones, so a positional read of it succeeds
             * silently: the old total_atomic_exchange_J lands in reconciliationRestartOffset, where it is counted a
             * second time, and the remaining reads fail leaving internalEnergy_J anchored at zero. Restarting
             * across a schema change is not supported, so say so rather than producing a plausible wrong ledger.
             */
            std::vector<std::string> columns;
            {
                std::istringstream tokens(line);
                std::string token;
                while(tokens >> token)
                    columns.push_back(token);
            }
            if(columns.size() != numberAtomicColumns)
            {
                /* Aborts the communicator instead of throwing. restart() returns early on every rank but the
                 * reduce master, so only the master reaches this check; a throw here would unwind one rank while
                 * every other rank walks into the next initialisation barrier and waits there for a peer that is
                 * already gone. Re-reading the file on all ranks to make the check symmetric would put every rank
                 * on the same file for one line of text, so abort loudly from the one rank that knows instead.
                 */
                std::cerr << "[AtomicEnergyExchange] ERROR: " << atomicFilename << " in the checkpoint has "
                          << columns.size() << " columns, expected " << numberAtomicColumns
                          << ". It was written by a different revision of the plugin and cannot be restarted "
                             "from.\n";
                std::cerr.flush();
                MPI_Abort(MPI_COMM_WORLD, 1);
                throw std::runtime_error("AtomicEnergyExchange: unsupported checkpoint schema"); // unreachable
            }

            std::istringstream values(line);
            uint32_t fileStep = 0u;
            values >> fileStep;
            for(uint32_t process = 1u; process < numberProcesses; ++process)
                values >> atomicRestartOffset[process];
            float_64 total = 0.0;
            float_64 referenceEnergy = 0.0;
            /* externalIonizationWeight is cumulative and its completeness statement is permanent: once any ion has
             * been ionized outside atomicPhysics the ledger stays incomplete for the rest of the simulation. The
             * device accumulator restarts at zero, so without this offset the column would fall back to zero and
             * the warning would go quiet on a restart of an already incomplete run.
             */
            values >> reconciliationRestartOffset >> total >> baselineEnergy >> referenceEnergy
                >> externalIonizationRestartOffset;

            /* Anchor to the restart point rather than to step 0: from here on
             *   internalEnergy(t) = internalEnergy(restart) + (total(t) - total(restart))
             * which is exact regardless of how the run was segmented, and needs no separate constant column.
             * Re-anchoring on a fresh reference energy instead would step the curve by the reservoir accrued so far.
             *
             * Only reached when the checkpointed line was found; otherwise the offsets stay zero and the first
             * output step takes a fresh baseline, consistent with a ledger that also restarts from zero.
             */
            baselineTotal = total;
            baselineKnown = true;
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

        /** absolute unperturbed-table reference energy of the species right now, in J
         *
         * @attention collective, calls an MPI reduce; every rank must enter it. Only the reduce master gets a
         *  meaningful return value, which is all the caller uses it for.
         */
        float_64 referenceEnergy()
        {
            DataConnector& dc = Environment<>::get().DataConnector();
            auto particles = dc.get<Particles>(Particles::FrameType::getName());

            using AtomicDataType = typename picongpu::traits::GetAtomicDataType<Particles>::type;
            auto atomicData = dc.get<AtomicDataType>(Particles::FrameType::getName() + "_atomicData");

            referenceEnergyBuffer->getDeviceBuffer().setValue(0.0);
            auto const mapper = makeAreaMapper<CORE + BORDER>(*m_cellDescription);
            PMACC_LOCKSTEP_KERNEL(KernelAtomicInternalEnergy{})
                .config(mapper.getGridDim(), *particles)(
                    particles->getDeviceParticlesBox(),
                    referenceEnergyBuffer->getDeviceBuffer().getDataBox(),
                    mapper,
                    atomicData->template getAtomicStateDataDataBox<false>(),
                    atomicData->template getChargeStateDataDataBox<false>());

            referenceEnergyBuffer->deviceToHost();
            float_64 reduced[2] = {0.0, 0.0};
            reduce(
                pmacc::math::operation::Add(),
                reduced,
                referenceEnergyBuffer->getHostBuffer().data(),
                2,
                mpi::reduceMethods::Reduce());

            return sim.si.conv().eV2Joule(reduced[0] + reduced[1]);
        }

        void calculate(uint32_t const currentStep)
        {
            // rank uniform, and set before any early return: onCheckpointStep() branches a collective call on it
            lastWrittenStep = currentStep;

            using Field = particles::atomicPhysics::localHelperFields::AtomicEnergyExchangeField<
                picongpu::MappingDesc,
                Particles>;
            DataConnector& dc = Environment<>::get().DataConnector();
            auto field = dc.get<Field>(Particles::FrameType::getName() + "_atomicEnergyExchangeField");
            field->synchronize();

            std::array<float_64, numberProcesses> localAtomic{};
            std::array<float_64, numberProcesses> localRadiated{};
            float_64 localReconciliation = 0.0;
            float_64 localExternalWeight = 0.0;
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
                localReconciliation += entry.getReconciliationEnergy();
                localExternalWeight += entry.getExternalIonizationWeight();
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
            float_64 globalReconciliation = 0.0;
            reduce(
                pmacc::math::operation::Add(),
                &globalReconciliation,
                &localReconciliation,
                1u,
                mpi::reduceMethods::Reduce());
            float_64 globalExternalWeight = 0.0;
            reduce(
                pmacc::math::operation::Add(),
                &globalExternalWeight,
                &localExternalWeight,
                1u,
                mpi::reduceMethods::Reduce());

            /* Collective, so it must happen on every rank and before the master-only guard below. Evaluated
             * unconditionally rather than only when a baseline is needed: the "do we need one" flag is master-only
             * state, and branching a collective on it would hang every other rank in the reduce. The cost is one
             * kernel and one reduce per output step.
             */
            float_64 const referenceEnergyNow = referenceEnergy();

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
            float_64 const reconciliationJ
                = sim.si.conv().eV2Joule(globalReconciliation) + reconciliationRestartOffset;
            totalAtomic += reconciliationJ;

            /* Anchor the cumulative ledger to an absolute internal energy, once. The reference energy is a valid
             * absolute value at any single instant; only its change in time is unusable, so evaluating it exactly
             * once and advancing with the ledger from there is correct.
             */
            if(!baselineKnown)
            {
                baselineEnergy = referenceEnergyNow;
                baselineTotal = totalAtomic;
                baselineKnown = true;
            }

            float_64 const externalWeight = globalExternalWeight + externalIonizationRestartOffset;

            atomicFile << " " << reconciliationJ << " " << totalAtomic << " "
                       << baselineEnergy + (totalAtomic - baselineTotal) << " " << referenceEnergyNow << " "
                       << externalWeight << std::endl;

            /* Loud once, rather than a silently wrong budget. Reaching this means an ionization module outside
             * atomicPhysics is active on this species; the internal energy those ions carried away is unrecoverable
             * by the time FixAtomicState sees them, so the ledger under-reports by an unknown amount.
             */
            if(externalWeight > 0.0 && !externalIonizationWarned)
            {
                externalIonizationWarned = true;
                std::cerr << "[AtomicEnergyExchange] WARNING: " << externalWeight
                          << " macro ion weight of " << Particles::FrameType::getName()
                          << " was ionized outside atomicPhysics by step " << currentStep
                          << ". Their internal energy is not in the ledger, so internalEnergy_J and "
                             "total_atomic_exchange_J are incomplete and must not be used for an energy budget.\n";
            }

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
        float_64 reconciliationRestartOffset = 0.0;
        //! weight, carried across restarts because the completeness statement it makes is permanent
        float_64 externalIonizationRestartOffset = 0.0;

        //! scratch for the one-time reference energy evaluation, see referenceEnergy()
        std::unique_ptr<GridBuffer<float_64, DIM1>> referenceEnergyBuffer;
        //! J, absolute internal energy at the anchor point, and the ledger total there
        float_64 baselineEnergy = 0.0;
        float_64 baselineTotal = 0.0;
        bool baselineKnown = false;
        bool externalIonizationWarned = false;
        //! @attention must stay rank uniform, see onCheckpointStep()
        uint32_t lastWrittenStep = std::numeric_limits<uint32_t>::max();
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
