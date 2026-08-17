// -----------------------------------------------------------------------------
//
// Copyright (C) 2021 CERN & University of Surrey for the benefit of the
// BioDynaMo collaboration. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
// See the LICENSE file distributed with this work for details.
// See the NOTICE file distributed with this work for additional information
// regarding copyright ownership.
//
// -----------------------------------------------------------------------------
#ifndef PREDATOR_PREY_H_
#define PREDATOR_PREY_H_

#include "biodynamo.h"
#include "core/environment/uniform_grid_environment.h"
#include <atomic>
#include <random>
#include <thread>
#include <fstream>

namespace bdm {


// Creates a CSV file and writes the header row
inline void CreateLog(const std::string& filename) {
  std::ofstream file(filename);
  if (file.is_open()) {
    file << "Step,Prey,Predators\n"; // Header row
    file.close();
  }
  else 
    std::cerr << "Error: could not create log file " << filename << "\n";
}

// Appends a row to the CSV file
inline void LogPopulation(const std::string& filename, int step, int prey, int predators) {
  std::ofstream file(filename, std::ios::app);
  if (file.is_open()) {
    file << step << "," << prey << "," << predators << "\n";
    file.close();
  }
  else 
    std::cerr << "Error: could not open log file " << filename << "\n";
}

// Returns a random double between the 0 and 1
inline double RandomDouble() {
  thread_local std::mt19937 rng(
    std::hash<std::thread::id>{}(std::this_thread::get_id()) ^
    (uint64_t(std::chrono::high_resolution_clock::now().time_since_epoch().count()))
  );
  thread_local std::uniform_real_distribution<double> dist(0.0, 1.0);
  return dist(rng);
}

// Returns a random double between the min and max
inline double RandomRange(double min, double max) {
  thread_local std::mt19937 rng(
    std::hash<std::thread::id>{}(std::this_thread::get_id()) ^
    (uint64_t(std::chrono::high_resolution_clock::now().time_since_epoch().count()))
  );
  thread_local std::uniform_real_distribution<double> dist(min, max);
  return dist(rng);
}

// Stops agents in their place if they try to leave the grid
inline void ClampPosition(Real3& position, double min, double max) {
  for (int i = 0; i < 2; i++) {
    if (position[i] < min) position[i] = min;
    if (position[i] > max) position[i] = max;
  }
}

// If agents try to leave the grid, they will instead wrap to the
// opposite side
inline void WrapPosition(Real3& position, double min, double max) {
  double size = max - min;
  for (int i = 0; i < 2; i++) {
    position[i] = fmod(position[i] - min, size);
    if (position[i] < 0) position[i] += size;
    position[i] += min;

    // Safety clamp
    if (position[i] >= max) position[i] = min;
    if (position[i] < min) position[i] = min;
  }
  position[2] = 0.0;
}

enum class AgentType { Prey, Predator };

// Lotka-Volterra parameters
struct Params {
  double preyBirthRate = 0.07;
  double predationRate = 0.32;
  double predDeathRate = 0.15;
  double predBirthRate = 0.75;
};

// Custom base class Animal
class Animal : public Cell {
public:
  // BDM_AGENT_HEADER(Animal, Cell, 1);
  Animal() {}
  explicit Animal(const Real3& position) : Cell(position) {}

  virtual AgentType GetAgentType() const { return AgentType::Prey; }
};

// Prey class
class Prey : public Animal {
public:
  BDM_AGENT_HEADER(Prey, Animal, 1);
  Prey() {}
  explicit Prey(const Real3& position) : Animal(position) {}

  // Copy constructor
  Prey(const Prey& other) : Animal(other) {
    // Copy the atomic values
    eaten_.store(other.eaten_.load());
    pendingRemoval_.store(other.pendingRemoval_.load());
  }

  // Copy assignment
  Prey& operator=(const Prey& other) {
    if (this != &other) {
      eaten_.store(other.eaten_.load());
      pendingRemoval_.store(other.pendingRemoval_.load());
    }
    return *this;
  }

  // Atomic boolean to prevent race condition between threads
  std::atomic<bool> eaten_{false};

  // Atomic boolean which never resets to prevent multiple removal
  std::atomic<bool> pendingRemoval_{false};

  AgentType GetAgentType() const override { return AgentType::Prey; }
};

// Predator class
class Predator : public Animal {
public:
  BDM_AGENT_HEADER(Predator, Animal, 1);
  Predator() {}
  explicit Predator(const Real3& position) : Animal(position), energy_(100) {}

  // Explicit copy constructor
  Predator(const Predator& other) : Animal(other) {
    energy_.store(other.energy_.load());
  }

  // Explicit copy assignment
  Predator& operator=(const Predator& other) {
    if (this != &other) {
      energy_.store(other.energy_.load());
    }
    return *this;
  }

  // Atomic int to prevent race condition between threads
  std::atomic<int> energy_{100};

  AgentType GetAgentType() const override { return AgentType::Predator; }
};

// Prey behaviour structure
struct PreyBehavior : public Behavior {
  BDM_BEHAVIOR_HEADER(PreyBehavior, Behavior, 1);

  void Run(Agent* agent) override {
    auto* animal = dynamic_cast<Animal*>(agent);
    if (!animal) return;
    if (animal->GetAgentType() != AgentType::Prey) return;

    auto* sim = Simulation::GetActive();
    auto* ctxt = sim->GetExecutionContext();
    Params p;

    auto* prey = static_cast<Prey*>(animal);
    prey->eaten_.store(false);

    // Check if prey is pending removal, if so, return
    if (prey->pendingRemoval_.load()) return;

    // Random movement
    Real3 position = prey->GetPosition();
    position[0] += RandomRange(-1.0, 1.0);
    position[1] += RandomRange(-1.0, 1.0);

    // Ensure the agent is in bounds
    // ClampPosition(position, 0.0, 50.0);
    WrapPosition(position, 0.0, 50.0);
    prey->SetPosition(position);

    // Reproduce
    if (RandomDouble() < p.preyBirthRate && !prey->eaten_.load()) {
      // std::cout << "Prey reproducing\n";
      auto* offspring = new Prey(position);
      offspring->AddBehavior(new PreyBehavior());
      ctxt->AddAgent(offspring);
    }
  }
};

// Prey search functor used in the predator behaviour structure
struct PreySearchFunctor : public Functor<void, Agent*, double> {
  Predator* predator;
  Animal* eatenPrey = nullptr;
  bool ate = false;
  Params p;

  // Pass the predator in so the functor can modify its state
  explicit PreySearchFunctor(Predator* pred) : predator(pred) {}

  void operator()(Agent* neighbour, double distance) override {
    if (ate) return;
    if (!predator) return;
    if (predator->energy_.load() <= 0) return;

    auto* animal = dynamic_cast<Animal*>(neighbour);
    if (!animal) return;
    if (animal->GetAgentType() != AgentType::Prey) return;

    auto* prey = static_cast<Prey*>(animal);

    // Check if this prey has been eaten by another predator
    // in the same timestep, if so, then don't eat this prey
    bool alreadyPending = prey->pendingRemoval_.exchange(true);
    if (alreadyPending) return;

    if (RandomDouble() < p.predationRate) {
      eatenPrey = animal;
      // predator->energy_.fetch_add(50);
      ate = true;
    }
    else {
      // Reset the flag if predator failed to eat despite finding prey
      prey->pendingRemoval_.store(false);
    }
  }
};

// Predator behaviour structure
struct PredatorBehavior : public Behavior {
  BDM_BEHAVIOR_HEADER(PredatorBehavior, Behavior, 1);

  void Run(Agent* agent) override {
    auto* animal = dynamic_cast<Animal*>(agent);
    if (!animal) return;
    if (animal->GetAgentType() != AgentType::Predator) return;

    auto* sim = Simulation::GetActive();
    auto* ctxt = sim->GetExecutionContext();
    Params p;

    auto* predator = static_cast<Predator*>(animal);
    Real3 position = predator->GetPosition();

    // Random movement
    position[0] += RandomRange(-1.0, 1.0);
    position[1] += RandomRange(-1.0, 1.0);

    // Ensure the agent is in bounds
    // ClampPosition(position, 0.0, 50.0);
    WrapPosition(position, 0.0, 50.0);
    predator->SetPosition(position);

    // predator->energy_.fetch_sub(1);

    // Use the functor for a neighbour search
    double search_radius = 2.0;
    PreySearchFunctor functor(predator);
    ctxt->ForEachNeighbor(functor, *predator, search_radius);
    
    // If prey was eaten, we need to remove the eaten prey
    if (functor.eatenPrey && functor.ate) {
      functor.eatenPrey->RemoveFromSimulation();

      // Reproduce after eating
      if (RandomDouble() < p.predBirthRate) {
        auto* offspring = new Predator(position);
        offspring->AddBehavior(new PredatorBehavior());
        ctxt->AddAgent(offspring);
        // std::cout << "Prey was eaten" << std::endl;
      }
    }

    // Death from starvation or natural causes
    if (predator->energy_.load() <= 0 || RandomDouble() < p.predDeathRate)
      predator->RemoveFromSimulation();
  }
};


// Simulation setup
inline int Simulate(int argc, const char** argv) {
  Simulation simulation(argc, argv);

  auto* rm = simulation.GetResourceManager();
  auto* random = simulation.GetRandom();

  const int kNumPrey = 400;
  const int kNumPredators = 50;
  const double kGridSize = 50.0;
  const int kSteps = 1000;
  const std::string kLogFile = "/media/sf_VirtualBoxSharedFolder/population_log.csv";

  // Seed prey
  for (int i = 0; i < kNumPrey; i++) {
    Real3 position = {random->Uniform(0, kGridSize),
                      random->Uniform(0, kGridSize),
                      0.0};
    auto* prey = new Prey(position);
    prey->AddBehavior(new PreyBehavior());
    rm->AddAgent(prey);
  }

  // Seed predators
  for (int i = 0; i < kNumPredators; i++) {
    Real3 position = {random->Uniform(0, kGridSize),
                      random->Uniform(0, kGridSize),
                      0.0};
    auto* predator = new Predator(position);
    predator->AddBehavior(new PredatorBehavior());
    rm->AddAgent(predator);
  }

  // Retrieve the environment and cast to UniformGridEnvironment
  auto* env = dynamic_cast<UniformGridEnvironment*>(simulation.GetEnvironment());
  if (!env) {
    std::cerr << "Error: Could not retrieve UniformGridEnvironment\n";
    return 1;
  }
  env->SetBoxLength(2); // Must be >= the search radius

  // Create the log file
  CreateLog(kLogFile);

  // Run the simulation one step at a time
  for (int t = 0; t < kSteps; t++) {
    simulation.GetScheduler()->Simulate(1);

    // Count populations at each step
    int preyCount = 0, predatorCount = 0;
    rm->ForEachAgent([&](Agent* agent) {
      auto* animal = dynamic_cast<Animal*>(agent);
      if (!animal) return;

      if (animal->GetAgentType() == AgentType::Prey) preyCount++;
      else predatorCount++;
    });

    LogPopulation(kLogFile, t, preyCount, predatorCount);

    std::cout << "t=" << t << " Prey=" << preyCount << 
              " Predators=" << predatorCount << "\n";

    if (preyCount == 0 || predatorCount == 0) {
      std::cout << "Population collapsed - ending simulation\n";
      break;
    }
  }

  std::cout << "Simulation completed successfully" << std::endl;
  return 0;
}

}  // namespace bdm

#endif  // PREDATOR_PREY_H_
