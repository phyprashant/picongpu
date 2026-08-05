/* Copyright 2026 Prashant Sharma
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

/** @file species plugin summing energy stored in atomic ion states */

#include "picongpu/defines.hpp"
#include "picongpu/particles/atomicPhysics/ParticleType.hpp"
#include "picongpu/particles/atomicPhysics/atomicData/AtomicData.hpp"
#include "picongpu/particles/traits/GenerateSolversIfSpeciesEligible.hpp"
#include "picongpu/particles/traits/GetAtomicDataType.hpp"
#include "picongpu/particles/traits/SpeciesEligibleForSolver.hpp"
#include "picongpu/plugins/PluginRegistry.hpp"
#include "picongpu/plugins/common/txtFileHandling.hpp"
#include "picongpu/plugins/multi/multi.hpp"

#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/lockstep.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/math/operation.hpp>
#include <pmacc/mpi/MPIReduce.hpp>
#include <pmacc/mpi/reduceMethods/Reduce.hpp>
#include <pmacc/particles/algorithm/ForEach.hpp>
#include <pmacc/traits/HasIdentifiers.hpp>

#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <string>

namespace picongpu
{
    /** Sum unperturbed ionization and excitation energies for an atomic ion species.
     *
     * Atomic-state energies are relative to the ground state of the respective charge
     * state.  The absolute reference used here is the neutral ground state, therefore
     * the ground-state ionization potentials below the particle's charge state must be
     * added.  Values accumulated by the kernel are weighted eV.
     */
    struct KernelAtomicInternalEnergy
    {
        template<
            typename T_Worker,
            typename T_ParBox,
            typename T_DBox,
            typename T_Mapping,
            typename T_AtomicStateDataBox,
            typename T_ChargeStateDataBox>
        DINLINE void operator()(
            T_Worker const& worker,
            T_ParBox particleBox,
            T_DBox globalEnergy,
            T_Mapping mapper,
            T_AtomicStateDataBox const atomicStateBox,
            T_ChargeStateDataBox const chargeStateBox) const
        {
            PMACC_SMEM(worker, sharedIonizationEnergy, float_64);
            PMACC_SMEM(worker, sharedExcitationEnergy, float_64);

            auto masterOnly = lockstep::makeMaster(worker);
            masterOnly(
                [&]()
                {
                    sharedIonizationEnergy = 0.0;
                    sharedExcitationEnergy = 0.0;
                });
            worker.sync();

            DataSpace<simDim> const superCellIdx(mapper.getSuperCellIndex(worker.blockDomIdxND()));
            auto forEachParticle
                = pmacc::particles::algorithm::acc::makeForEach(worker, particleBox, superCellIdx);
            if(!forEachParticle.hasParticles())
                return;

            float_64 localIonizationEnergy = 0.0;
            float_64 localExcitationEnergy = 0.0;

            forEachParticle(
                [&atomicStateBox,
                 &chargeStateBox,
                 &localIonizationEnergy,
                 &localExcitationEnergy](auto const&, auto& particle)
                {
                    uint32_t const stateIndex = particle[atomicStateCollectionIndex_];
                    auto const configNumber = atomicStateBox.configNumber(stateIndex);
                    uint8_t const chargeState = T_AtomicStateDataBox::ConfigNumber::getChargeState(configNumber);
                    float_64 const weighting = static_cast<float_64>(particle[weighting_]);

                    float_64 ionizationEnergy = 0.0;
                    for(uint8_t q = 0u; q < chargeState; ++q)
                        ionizationEnergy += static_cast<float_64>(chargeStateBox.ionizationEnergy(q));

                    localIonizationEnergy += weighting * ionizationEnergy;
                    localExcitationEnergy
                        += weighting * static_cast<float_64>(atomicStateBox.energy(stateIndex));
                });

            alpaka::atomicAdd(
                worker.getAcc(),
                &sharedIonizationEnergy,
                localIonizationEnergy,
                ::alpaka::hierarchy::Threads{});
            alpaka::atomicAdd(
                worker.getAcc(),
                &sharedExcitationEnergy,
                localExcitationEnergy,
                ::alpaka::hierarchy::Threads{});
            worker.sync();

            masterOnly(
                [&]()
                {
                    alpaka::atomicAdd(
                        worker.getAcc(),
                        &globalEnergy[0],
                        sharedIonizationEnergy,
                        ::alpaka::hierarchy::Blocks{});
                    alpaka::atomicAdd(
                        worker.getAcc(),
                        &globalEnergy[1],
                        sharedExcitationEnergy,
                        ::alpaka::hierarchy::Blocks{});
                });
        }
    };

    template<typename T_Particles>
    class AtomicInternalEnergy : public plugins::multi::IInstance
    {
    public:
        using Particles = T_Particles;

        struct Help : public plugins::multi::IHelp
        {
            std::shared_ptr<IInstance> create(
                std::shared_ptr<IHelp>& help,
                size_t const id,
                MappingDesc* cellDescription) override
            {
                return std::shared_ptr<IInstance>(new AtomicInternalEnergy<Particles>(help, id, cellDescription));
            }

            plugins::multi::Option<std::string> notifyPeriod
                = {"period", "compute atomic internal energy [for each n-th step]"};

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

            std::string const name = "AtomicInternalEnergy";
            std::string const description = "sum energy stored in an atomic ion species";
            std::string const prefix = Particles::FrameType::getName() + std::string("_atomicInternalEnergy");
        };

        static std::shared_ptr<plugins::multi::IHelp> getHelp()
        {
            return std::shared_ptr<plugins::multi::IHelp>(new Help{});
        }

        AtomicInternalEnergy(
            std::shared_ptr<plugins::multi::IHelp>& help,
            size_t const id,
            MappingDesc* cellDescription)
            : m_cellDescription(cellDescription)
            , m_help(std::static_pointer_cast<Help>(help))
            , m_id(id)
        {
            filename = Particles::FrameType::getName() + std::string("_atomic_internal_energy.dat");
            writeToFile = reduce.hasResult(mpi::reduceMethods::Reduce());
            globalEnergy = std::make_unique<GridBuffer<float_64, DIM1>>(DataSpace<DIM1>(2));

            if(writeToFile)
            {
                outFile.open(filename.c_str(), std::ofstream::out | std::ostream::trunc);
                if(!outFile)
                {
                    std::cerr << "Can't open file [" << filename << "] for output; disable plugin output.\n";
                    writeToFile = false;
                }
                else
                {
                    outFile << "# Atomic-state energies use the neutral-ground-state reference and the "
                               "unperturbed input tables; dynamic IPD shifts are not included.\n";
                    outFile << "#step ionization_J excitation_J total_atomic_J\n";
                }
            }

            Environment<>::get().PluginConnector().setNotificationPeriod(this, m_help->notifyPeriod.get(id));
        }

        ~AtomicInternalEnergy() override
        {
            if(writeToFile)
            {
                outFile.flush();
                outFile.close();
            }
        }

        void notify(uint32_t currentStep) override
        {
            calculate(currentStep);
        }

        void restart(uint32_t restartStep, std::string const& restartDirectory) override
        {
            if(writeToFile)
                writeToFile = restoreTxtFile(outFile, filename, restartStep, restartDirectory);
        }

        void checkpoint(uint32_t currentStep, std::string const& checkpointDirectory) override
        {
            if(writeToFile)
                checkpointTxtFile(outFile, filename, currentStep, checkpointDirectory);
        }

    private:
        void calculate(uint32_t currentStep)
        {
            DataConnector& dc = Environment<>::get().DataConnector();
            auto particles = dc.get<Particles>(Particles::FrameType::getName());

            using AtomicDataType = typename picongpu::traits::GetAtomicDataType<Particles>::type;
            auto atomicData = dc.get<AtomicDataType>(Particles::FrameType::getName() + "_atomicData");

            globalEnergy->getDeviceBuffer().setValue(0.0);
            auto const mapper = makeAreaMapper<CORE + BORDER>(*m_cellDescription);
            PMACC_LOCKSTEP_KERNEL(KernelAtomicInternalEnergy{})
                .config(mapper.getGridDim(), *particles)(
                    particles->getDeviceParticlesBox(),
                    globalEnergy->getDeviceBuffer().getDataBox(),
                    mapper,
                    atomicData->template getAtomicStateDataDataBox<false>(),
                    atomicData->template getChargeStateDataDataBox<false>());

            globalEnergy->deviceToHost();
            float_64 reducedEnergy[2] = {0.0, 0.0};
            reduce(
                pmacc::math::operation::Add(),
                reducedEnergy,
                globalEnergy->getHostBuffer().data(),
                2,
                mpi::reduceMethods::Reduce());

            if(writeToFile)
            {
                using dbl = std::numeric_limits<float_64>;
                float_64 const ionizationJ = sim.si.conv().eV2Joule(reducedEnergy[0]);
                float_64 const excitationJ = sim.si.conv().eV2Joule(reducedEnergy[1]);
                outFile.precision(dbl::digits10);
                outFile << currentStep << " " << std::scientific << ionizationJ << " " << excitationJ << " "
                        << ionizationJ + excitationJ << std::endl;
            }
        }

        std::unique_ptr<GridBuffer<float_64, DIM1>> globalEnergy;
        MappingDesc* m_cellDescription;
        std::string filename;
        std::ofstream outFile;
        bool writeToFile = false;
        mpi::MPIReduce reduce;
        std::shared_ptr<Help> m_help;
        size_t m_id;
    };

    namespace particles::traits
    {
        template<typename T_Species, typename T_UnspecifiedSpecies>
        struct SpeciesEligibleForSolver<T_Species, AtomicInternalEnergy<T_UnspecifiedSpecies>>
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

PIC_REGISTER_SPECIES_PLUGIN(picongpu::plugins::multi::Master<picongpu::AtomicInternalEnergy<boost::mpl::_1>>);
