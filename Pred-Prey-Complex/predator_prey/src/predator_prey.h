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


enum class AgentType { Prey, Predator };

// Lotka-Volterra parameters
struct Params {
  double preyBirthRate = 0.09;
  double predationRate = 0.32;
  double predDeathRate = 0.05;
  double predBirthRate = 0.55;
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
  explicit Prey(const Real3& position) 
    : Animal(position), energy_(100), age_(0),
      speed_(1.0), perception_(5.0), isJuvenile_(true) {}

  // Pass traits explicitly when spawning offspring
  explicit Prey(const Real3& position, double speed, double perception)
    : Animal(position), energy_(100), age_(0),
      speed_(speed), perception_(perception), isJuvenile_(true) {}

  // Copy constructor
  Prey(const Prey& other) : Animal(other) {
    // Copy the values
    eaten_.store(other.eaten_.load());
    pendingRemoval_.store(other.pendingRemoval_.load());
    energy_.store(other.energy_.load());
    age_.store(other.age_.load());
    speed_ = other.speed_;
    perception_ = other.perception_;
    isJuvenile_.store(other.isJuvenile_.load());
  }

  // Copy assignment
  Prey& operator=(const Prey& other) {
    if (this != &other) {
      eaten_.store(other.eaten_.load());
      pendingRemoval_.store(other.pendingRemoval_.load());
      energy_.store(other.energy_.load());
      age_.store(other.age_.load());
      speed_ = other.speed_;
      perception_ = other.perception_;
      isJuvenile_.store(other.isJuvenile_.load());
    }
    return *this;
  }

  // Atomic boolean to prevent race condition between threads
  std::atomic<bool> eaten_{false};

  // Atomic boolean which never resets to prevent multiple removal
  std::atomic<bool> pendingRemoval_{false};

  // Atomic integer to represent prey hunger
  std::atomic<int> energy_{100};

  // Atomic integer for agent age
  std::atomic<int> age_{0};

  // Heritable traits - both slightly lower than predators
  double speed_ = 1.0;
  double perception_ = 5.0;

  // Atomic boolean indicating whether prey is juvenile
  std::atomic<bool> isJuvenile_{true};

  AgentType GetAgentType() const override { return AgentType::Prey; }
};

// Predator class
class Predator : public Animal {
public:
  BDM_AGENT_HEADER(Predator, Animal, 1);
  Predator() {}
  explicit Predator(const Real3& position) 
    : Animal(position), energy_(100), age_(0),
      speed_(1.0), perception_(5.0), isJuvenile_(true) {}

  // Pass traits explicitly when spawning offspring
  explicit Predator(const Real3& position, double speed, double perception)
    : Animal(position), energy_(100), age_(0),
      speed_(speed), perception_(perception), isJuvenile_(true) {}

  // Explicit copy constructor
  Predator(const Predator& other) : Animal(other) {
    pendingRemoval_.store(other.pendingRemoval_.load());
    energy_.store(other.energy_.load());
    age_.store(other.age_.load());
    speed_ = other.speed_;
    perception_ = other.perception_;
    isJuvenile_.store(other.isJuvenile_.load());
  }

  // Explicit copy assignment
  Predator& operator=(const Predator& other) {
    if (this != &other) {
      pendingRemoval_.store(other.pendingRemoval_.load());
      energy_.store(other.energy_.load());
      age_.store(other.age_.load());
      speed_ = other.speed_;
      perception_ = other.perception_;
      isJuvenile_.store(other.isJuvenile_.load());
    }
    return *this;
  }

  // Atomic boolean which never resets to prevent multiple removal
  std::atomic<bool> pendingRemoval_{false};

  // Atomic int to prevent race condition between threads
  std::atomic<int> energy_{100};

  // Atomic integer for agent age
  std::atomic<int> age_{0};

  // Heritable traits - both slightly higher than prey
  double speed_ = 1.0;
  double perception_ = 5.0;

  // Atomic boolean indicating whether prey is juvenile
  std::atomic<bool> isJuvenile_{true};

  AgentType GetAgentType() const override { return AgentType::Predator; }
};


/******************** Helper functions ****************************/

// Creates a CSV file and writes the header row
inline void CreateLog(const std::string& filename) {
  std::ofstream file(filename);
  if (file.is_open()) {
    file << "Step,Prey,Predators,Season,AvgPreySpd,AvgPreyPcp,AvgPredSpd,AvgPredPcp\n"; // Header row
    file.close();
  }
  else 
    std::cerr << "Error: could not create log file " << filename << "\n";
}

// Appends a row to the CSV file
inline void LogPopulation(const std::string& filename, int step, int prey, 
                            int predators, double season,
                            double avgPreySpd, double avgPreyPcp,
                            double avgPredSpd, double avgPredPcp) {
  std::ofstream file(filename, std::ios::app);
  if (file.is_open()) {
    file << step << "," << prey << "," << predators << "," << season << ","
        << avgPreySpd << "," << avgPreyPcp << ","
        << avgPredSpd << "," << avgPredPcp << "\n";
    file.close();
  }
  else 
    std::cerr << "Error: could not open log file " << filename << "\n";
}

struct TraitAverages {
  double avgPreySpeed = -1.0;
  double avgPreyPerception = -1.0;
  double avgPredSpeed = -1.0;
  double avgPredPerception = -1.0;
};

inline TraitAverages ComputeTraitAverages(ResourceManager* rm) {
  double preySumSpeed = 0.0, preySumPerception = 0.0;
  double predSumSpeed = 0.0, predSumPerception = 0.0;
  int preyCount = 0, predCount = 0;

  rm->ForEachAgent([&](Agent* agent) {
    auto* animal = dynamic_cast<Animal*>(agent);
    if (!animal) return;

    if (animal->GetAgentType() == AgentType::Prey) {
      auto* prey = static_cast<Prey*>(animal);
      preySumSpeed += prey->speed_;
      preySumPerception += prey->perception_;
      preyCount++;
    }
    else {
      auto* predator = static_cast<Predator*>(animal);
      predSumSpeed += predator->speed_;
      predSumPerception += predator->perception_;
      predCount++;
    }
  });

  TraitAverages result;
  if (preyCount > 0) {
    result.avgPreySpeed = preySumSpeed / preyCount;
    result.avgPreyPerception = preySumPerception / preyCount;
  }
  if (predCount > 0) {
    result.avgPredSpeed = predSumSpeed / predCount;
    result.avgPredPerception = predSumPerception / predCount;
  }
  return result;
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
  std::uniform_real_distribution<double> dist(min, max);
  return dist(rng);
}


// Normalises a 2D direction vector, returns zero vector if near-zero magnitude
inline Real3 Normalise2D(Real3 v) {
  double mag = std::sqrt(v[0]*v[0] + v[1]*v[1]);
  if (mag < 1e-6) return {0.0, 0.0, 0.0};
  return {v[0]/mag, v[1]/mag, 0.0};
}

// Blends two direction vectors with a given weight (0=all a, 1=all b)
inline Real3 BlendDirections(Real3 a, Real3 b, double weight) {
  return {a[0]*(1-weight) + b[0]*weight,
          a[1]*(1-weight) + b[1]*weight,
          0.0};
}


// Stops agents in their place if they try to leave the grid
inline void ClampPosition(Real3& position, double min, double max) {
  for (int i = 0; i < 2; i++) {
    if (position[i] <= min) position[i] = min;
    if (position[i] >= max) position[i] = max;
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

// Returns a seasonal multiplier in the range [minVal, 1.0]
// Period is in timesteps
// Phase shifts the peak - 0.0 means peak at timestep 0
inline double SeasonalMultiplier(int step, double period = 365.0, 
                              double minVal = 0.25, double phase = 0.0) {
  // Since oscillations between -1 and 1 is mapped to [minVal, 1.0]
  double raw = std::sin(2.0 * M_PI * (step / period) + phase);
  return minVal + (1.0 - minVal) * (raw + 1.0) / 2.0;
}

// Refuges defined as circles with a centre and radius
struct RefugeZone {
  double cx, cy, radius;
};

// Refuge positions
inline const std::vector<RefugeZone>& GetRefuges() {
  static std::vector<RefugeZone> refuges = {
    {12.5, 12.5, 4.0}, // Bottom-left quadrant
    {37.5, 37.5, 4.0}, // Top-right quadrant
    {12.5, 37.5, 3.0}, // Top-left quadrant
  };
  return refuges;
}

// Returns true if position is inside a refuge zone
inline bool InRefuge(double x, double y) {
  for (const auto& r : GetRefuges()) {
    double dx = x - r.cx;
    double dy = y - r.cy;
    if (dx*dx + dy*dy < r.radius*r.radius) return true;
  }
  return false;
}

// Water source defined the same way as the refuge zones
struct WaterZone {
  double cx, cy, radius;
};

// Water zones
inline const std::vector<WaterZone>& GetWaterSources() {
  static std::vector<WaterZone> water = {
    {25.0, 25.0, 5.0}, // Central water source
    {10.0, 40.0, 3.0}, // Top-left pool
    {40.0, 10.0, 3.0}, // Bottom-right pool
  };
  return water;
}

// Returns true if position is inside a water source
inline bool InWater(double x, double y) {
  for (const auto& w : GetWaterSources()) {
    double dx = x - w.cx;
    double dy = y - w.cy;
    if (dx*dx + dy*dy < w.radius*w.radius) return true;
  }
  return false;
}

/************************ End of helper functions ***************************/


/****************************** Resource grid ****************************/

// Resource grid with vegetation available for prey
struct ResourceGrid {

  static constexpr int kGridCells = 25; // 25x25 cells over 50x50 space
  static constexpr int kCellSize = 2; // Size of resource patch
  static constexpr int kRegenRate = 3; // Resource added per cell per step
  static constexpr int kEatCost = 15; // Resource consumed per prey per step

  std::vector<std::atomic<int>> cells_;
  std::vector<int> cellMax_; // Per cell carrying capacity

  ResourceGrid() : cells_(kGridCells * kGridCells), cellMax_(kGridCells * kGridCells) {
    // Fixed seed for a constant landscape across runs
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(20, 100);

    for (int i = 0; i < kGridCells * kGridCells; i++) {
      // Randomly assign each cell a max capacity between 20 and 100
      cellMax_[i] = dist(rng);
    }

    // Run multiple passes for smoother gradients to create clusters
    const int kSmoothingPasses = 4;
    for (int pass = 0; pass < kSmoothingPasses; pass++) {
      std::vector<int> smoothed(kGridCells * kGridCells);
      for (int row = 0; row < kGridCells; row++) {
        for (int col = 0; col < kGridCells; col++) {
          int idx = row * kGridCells + col;

          // Gather the cell and its four cardinal neighbours
          // Wrap at edges to avoid border artefacts
          int left = row * kGridCells + (col - 1 + kGridCells) % kGridCells;
          int right = row * kGridCells + (col + 1) % kGridCells;
          int up = ((row - 1 + kGridCells) % kGridCells) * kGridCells + col;
          int down = ((row + 1) % kGridCells) * kGridCells + col;

          // Weighted average
          smoothed[idx] = (cellMax_[idx] * 2 +
                          cellMax_[left] + cellMax_[right] +
                          cellMax_[up] + cellMax_[down]) / 6;
        }
      }
      cellMax_ = smoothed;
    }

    // Initialise cell resources to their cell maximum
    for (int i = 0; i < kGridCells * kGridCells; i++)
      cells_[i].store(cellMax_[i]);
  }

  int Index(double x, double y) const {
    auto clamp = [](int val, int lo, int hi) {
      return val < lo ? lo : (val > hi ? hi : val);
    };
    int ix = clamp((int)(x / kCellSize), 0, kGridCells - 1);
    int iy = clamp((int)(y / kCellSize), 0, kGridCells - 1);
    return iy * kGridCells + ix;
  }

  bool Consume(double x, double y) {
    int idx = Index(x, y);
    int current = cells_[idx].load();
    while (current >= kEatCost) {
      if (cells_[idx].compare_exchange_weak(current, current - kEatCost))
        return true;
      // Current is updated by compare_exchange_weak on failure - retry
    }
    return false; // Cell was depleted
  }

  void Regenerate(int step) {
    // Resources grow faster in spring/summer and slower in autumn/winter
    double season = SeasonalMultiplier(step, 365.0, 0.1, 0.0);
    int seasonalRegen = std::max(1, (int)(kRegenRate * season));

    for (int i = 0; i < kGridCells * kGridCells; i++) {
      int val = cells_[i].load();
      if (val < cellMax_[i])
        cells_[i].store(std::min(val + seasonalRegen, cellMax_[i]));
    }
  }
};

// Global resource grid
static ResourceGrid gResourceGrid;

/************************** End of Resource grid *****************************/


// Global step counter
static std::atomic<int> gCurrentStep{0};

// Counts how many agents are within a search radius - used for carrying capacity
struct NeighbourCountFunctor : public Functor<void, Agent*, double> {
  int count = 0;
  AgentType targetType;

  explicit NeighbourCountFunctor(AgentType t) : targetType(t) {}

  void operator()(Agent* neighbour, double distance) override {
    auto* animal = dynamic_cast<Animal*>(neighbour);
    if (!animal) return;
    if (animal->GetAgentType() == targetType) count++;
  }
};


/***************************** Prey behaviour *************************/

// Prey search functor to detect nearby predators
struct NearestPredatorFunctor : public Functor<void, Agent*, double> {
  Real3 nearestPos = {0.0, 0.0, 0.0};
  double nearestDist = std::numeric_limits<double>::max();
  bool foundThreat = false;

  void operator()(Agent* neighbour, double distance) override {
    auto* animal = dynamic_cast<Animal*>(neighbour);
    if (!animal) return;
    if (animal->GetAgentType() != AgentType::Predator) return;

    if (distance < nearestDist) {
      nearestDist = distance;
      nearestPos = animal->GetPosition();
      foundThreat = true;
    }
  }
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

    Real3 position = prey->GetPosition();
    Real3 moveDir = {0.0, 0.0, 0.0};

    double kFleeRadius = prey->perception_; // Threat detection range
    const double kFleeBias = 0.8; // 0=Random, 1=Pure avoidance

    const int kMaxAge = 730; // Lifespan cap
    const double kMinSpeedFactor = 0.5; // Min. speed is 50%
    int age = prey->age_.load();
    double ageFactor = 1.0 - (1.0 - kMinSpeedFactor) * ((double) age / kMaxAge);
    double kStepSize = prey->speed_ * ageFactor; // Movement distance per step

    // Scan for nearest predators within a detection range
    NearestPredatorFunctor fleeFunctor;
    ctxt->ForEachNeighbor(fleeFunctor, *prey, kFleeRadius);

    // If a predator is found within the search radius
    if (fleeFunctor.foundThreat) {
      // Direction directly away from the nearest predator
      Real3 awayFromPred = {position[0] - fleeFunctor.nearestPos[0],
                            position[1] - fleeFunctor.nearestPos[1],
                            0.0};
      Real3 fleeDir = Normalise2D(awayFromPred);

      // Random influence so fleeing is not perfectly deterministic
      Real3 randDir = Normalise2D({RandomRange(-1, 1), RandomRange(-1, 1), 0});

      moveDir = BlendDirections(randDir, fleeDir, kFleeBias);
    }
    else {
      // No predators nearby - so random movement
      moveDir = Normalise2D({RandomRange(-1, 1), RandomRange(-1, 1), 0});
    }

    position[0] += moveDir[0] * kStepSize;
    position[1] += moveDir[1] * kStepSize;

    // Ensure the agent is in bounds
    // ClampPosition(position, 0.0, 50.0);
    WrapPosition(position, 0.0, 50.0);
    prey->SetPosition(position);

    // Resource consumption - prey eat from the local cell
    bool ate = gResourceGrid.Consume(position[0], position[1]);

    if (ate) {
      prey->energy_.fetch_add(10);
      if (prey->energy_.load() > 100) prey->energy_.store(100); // Cap at 100
    }
    else {
      prey->energy_.fetch_sub(3); // Starve if cell is depleted
    }

    // Water source bonus - small energy boost
    if (InWater(position[0], position[1])) {
      prey->energy_.fetch_add(2);
      if (prey->energy_.load() > 100) prey->energy_.store(100);
    }

    // Starvation death check
    if (prey->energy_.load() <= 0) {
      bool alreadyPending = prey->pendingRemoval_.exchange(true);
      if (!alreadyPending)
        prey->RemoveFromSimulation();
      return;
    }

    // Increment age and check for age-related death
    prey->age_.fetch_add(1);

    const double kAgeDeathScale = 0.000007; // Scale probability with age
    const int kJuvenileAge = 60; // Steps before prey can reproduce - lower than predators

    // Maturation check
    if (age >= kJuvenileAge && prey->isJuvenile_.load())
      prey->isJuvenile_.store(false);

    if (age >= kMaxAge || RandomDouble() < kAgeDeathScale * age * age) {
      bool alreadyPending = prey->pendingRemoval_.exchange(true);
      if (!alreadyPending)
        prey->RemoveFromSimulation();
      return;
    }

    const double kCarryingRadius = 5.0; // Search radius to count nearby agents
    const int kMaxLocalDensity = 15; // Max local prey

    NeighbourCountFunctor countFunctor(AgentType::Prey);
    ctxt->ForEachNeighbor(countFunctor, *prey, kCarryingRadius);

    // Seasonal reproduction
    double season = SeasonalMultiplier(gCurrentStep.load());
    double seasonalBirth = p.preyBirthRate * season;

    if (/* !prey->isJuvenile_.load() && */
        countFunctor.count < kMaxLocalDensity &&
        RandomDouble() < seasonalBirth && 
        !prey->eaten_.load()) {

      // Litter size with 75% of 1 offspring, 20% chance of 2, and 5% chance of 3
      int litterSize = 1;
      double litterRoll = RandomDouble();
      if (litterRoll < 0.05) litterSize = 3;
      else if (litterRoll < 0.25) litterSize = 2;

      for (int i = 0; i < litterSize; i++) {
        // Inherit traits with small random mutation
        const double kMutationStrength = 0.02;
        double newSpeed = std::max(0.3, prey->speed_ + RandomRange(
                                              -kMutationStrength, kMutationStrength));
        double newPerception = std::max(1.0, prey->perception_ + RandomRange(
                                              -kMutationStrength, kMutationStrength));
        
        auto* offspring = new Prey(position, newSpeed, newPerception);
        offspring->AddBehavior(new PreyBehavior());
        ctxt->AddAgent(offspring);
      }      
    }
  }
};

/************************* End of prey behaviour *************************/


/************************* Predator behaviour ****************************/

// Scans the surrounding neighbours and move towards the closest prey found
struct NearestPreyFunctor : public Functor<void, Agent*, double> {
  Real3 nearestPos = {0.0, 0.0, 0.0};
  double nearestDist = std::numeric_limits<double>::max();
  bool foundPrey = false;

  void operator()(Agent* neighbour, double distance) override {
    auto* animal = dynamic_cast<Animal*>(neighbour);
    if (!animal) return;
    if (animal->GetAgentType() != AgentType::Prey) return;

    // Skip prey already marked for removal
    auto* prey = static_cast<Prey*>(animal);
    if (prey->pendingRemoval_.load()) return;

    if (distance < nearestDist) {
      nearestDist = distance;
      nearestPos = animal->GetPosition();
      foundPrey = true;
    }
  }
};

// Predator search functor to eat nearby prey
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

    // Prey inside refuge cannot be eaten
    Real3 preyPos = animal->GetPosition();
    if (InRefuge(preyPos[0], preyPos[1])) return;

    // Juvenile prey cannot be hunted
    // if (prey->isJuvenile_.load()) return;

    // Check if this prey has been eaten by another predator
    // in the same timestep, if so, then don't eat this prey
    bool alreadyPending = prey->pendingRemoval_.exchange(true);
    if (alreadyPending) return;

    if (RandomDouble() < p.predationRate) {
      eatenPrey = animal;
      predator->energy_.fetch_add(50);
      if (predator->energy_.load() > 100) predator->energy_.store(100); // Cap at 100
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

    // predator->pendingRemoval_.store(false);
    if (predator->pendingRemoval_.load()) return;

    Real3 position = predator->GetPosition();
    Real3 moveDir = {0.0, 0.0, 0.0};

    double kHuntRadius = predator->perception_; // Detection range
    const double kHuntBias = 0.6; // 0=Random, 1=Pure pursuit

    const int kMaxAge = 925; // Lifespan cap
    const double kMinSpeedFactor = 0.5;
    int age = predator->age_.load();
    double ageFactor = 1.0 - (1.0 - kMinSpeedFactor) * ((double) age / kMaxAge);
    double kStepSize = predator->speed_ * ageFactor; // Movement distance per step

    // Predators slow down in water zones
    if (InWater(position[0], position[1])) 
      kStepSize *= 0.5; // Half speed through water

    // Scan for nearest prey within a detection range
    NearestPreyFunctor huntFunctor;
    ctxt->ForEachNeighbor(huntFunctor, *predator, kHuntRadius);

    // If prey is found within the search radius
    if (huntFunctor.foundPrey) {
      // Direction towards nearest prey
      Real3 towardPrey = {huntFunctor.nearestPos[0] - position[0],
                          huntFunctor.nearestPos[1] - position[1],
                          0.0};
      Real3 pursuitDir = Normalise2D(towardPrey);

      // Random influence so pursuit is not perfectly deterministic
      Real3 randDir = Normalise2D({RandomRange(-1, 1), RandomRange(-1, 1), 0});

      moveDir = BlendDirections(randDir, pursuitDir, kHuntBias);
    }
    else {
      // No prey nearby, so random movement
      moveDir = Normalise2D({RandomRange(-1, 1), RandomRange(-1, 1), 0});
    }

    position[0] += moveDir[0] * kStepSize;
    position[1] += moveDir[1] * kStepSize;

    // Ensure the agent is in bounds
    // ClampPosition(position, 0.0, 50.0);
    WrapPosition(position, 0.0, 50.0);
    predator->SetPosition(position);

    // If predator has entered a refuge, push it back out
    if (InRefuge(position[0], position[1])) {
      // Move away from the centre
      for (const auto& r : GetRefuges()) {
        double dx = position[0] - r.cx;
        double dy = position[1] - r.cy;
        double dist = std::sqrt(dx*dx + dy*dy);
        if (dist < r.radius) {
          // Push to the boundary
          position[0] = r.cx + (dx/dist) * (r.radius + 0.1);
          position[1] = r.cy + (dy/dist) * (r.radius + 0.1);
          WrapPosition(position, 0.0, 50.0);
          predator->SetPosition(position);
          break;
        }
      }
    }

    predator->energy_.fetch_sub(1);

    // Increment age and check for age-related death
    predator->age_.fetch_add(1);

    const double kAgeDeathScale = 0.000004; // Scale probability with age
    const int kJuvenileAge = 75; // Steps before predators can reproduce - higher than prey

    // Maturation check
    if (age >= kJuvenileAge && predator->isJuvenile_.load())
      predator->isJuvenile_.store(false);

    if (age >= kMaxAge || RandomDouble() < kAgeDeathScale * age * age) {
      bool alreadyPending = predator->pendingRemoval_.exchange(true);
      if (!alreadyPending)
        predator->RemoveFromSimulation();
      return;
    }

    // Use the functor for a neighbour search
    double search_radius = 2.0;
    PreySearchFunctor functor(predator);
    ctxt->ForEachNeighbor(functor, *predator, search_radius);
    
    // If prey was eaten, we need to remove the eaten prey
    if (functor.eatenPrey && functor.ate) {
      functor.eatenPrey->RemoveFromSimulation();

      // Reproduce after eating
      if (/* !predator->isJuvenile_.load() && */ RandomDouble() < p.predBirthRate) {
        // Predators rarely have more than one offspring
        int litterSize = 1;
        if (RandomDouble() < 0.08) litterSize = 2; // 8% chance of twins

        for (int i = 0; i < litterSize; i++) {
          // Inherit traits with small random mutation
          const double kMutationStrength = 0.02;
          double newSpeed = std::max(0.3, predator->speed_ + RandomRange(
                                                -kMutationStrength, kMutationStrength));
          double newPerception = std::max(1.0, predator->perception_ + RandomRange(
                                                -kMutationStrength, kMutationStrength));

          auto* offspring = new Predator(position, newSpeed, newPerception);
          offspring->AddBehavior(new PredatorBehavior());
          ctxt->AddAgent(offspring);
          // std::cout << "Predator reproduced\n";
        }        
      }
    }

    // Death from starvation or natural causes
    if (predator->energy_.load() <= 0 || RandomDouble() < p.predDeathRate) {
      bool alreadyPending = predator->pendingRemoval_.exchange(true);
      if (!alreadyPending)
        predator->RemoveFromSimulation();
    }
  }
};

/********************** End of predator behaviour ************************/


// Simulation setup
inline int Simulate(int argc, const char** argv) {
  Simulation simulation(argc, argv);

  auto* rm = simulation.GetResourceManager();
  auto* random = simulation.GetRandom();

  const int kNumPrey = 400;
  const int kNumPredators = 50;
  const double kGridSize = 50.0;
  const int kSteps = 2000;
  const std::string kLogFile = "/media/sf_VirtualBoxSharedFolder/population_log.csv";

  // Seed prey
  for (int i = 0; i < kNumPrey; i++) {
    Real3 position = {random->Uniform(0, kGridSize),
                      random->Uniform(0, kGridSize),
                      0.0};
    auto* prey = new Prey(position);
    // prey->isJuvenile_.store(false); // Initial population starts as adults
    // prey->age_.store(60);
    prey->AddBehavior(new PreyBehavior());
    rm->AddAgent(prey);
  }

  // Seed predators
  for (int i = 0; i < kNumPredators; i++) {
    Real3 position = {random->Uniform(0, kGridSize),
                      random->Uniform(0, kGridSize),
                      0.0};
    auto* predator = new Predator(position);
    // predator->isJuvenile_.store(false); // Initial population starts as adults
    // predator->age_.store(75);
    predator->AddBehavior(new PredatorBehavior());
    rm->AddAgent(predator);
  }

  // Retrieve the environment and cast to UniformGridEnvironment
  auto* env = dynamic_cast<UniformGridEnvironment*>(simulation.GetEnvironment());
  if (!env) {
    std::cerr << "Error: Could not retrieve UniformGridEnvironment\n";
    return 1;
  }
  env->SetBoxLength(11); // Must be >= the search radius

  // Create the log file
  CreateLog(kLogFile);

  // Compute trait averages every 100 timesteps
  const int kStoreTraitsInterval = 100;
  TraitAverages traits;

  // Run the simulation one step at a time
  for (int t = 0; t < kSteps; t++) {
    gCurrentStep.store(t);
    gResourceGrid.Regenerate(t); // Regenerate the resources for prey
    simulation.GetScheduler()->Simulate(1);

    // Count populations at each step
    int preyCount = 0, predatorCount = 0;
    rm->ForEachAgent([&](Agent* agent) {
      auto* animal = dynamic_cast<Animal*>(agent);
      if (!animal) return;

      if (animal->GetAgentType() == AgentType::Prey) preyCount++;
      else predatorCount++;
    });

    // Compute trait averages every kStoreTraitsInterval
    if (t % kStoreTraitsInterval == 0)
      traits = ComputeTraitAverages(rm);

    double season = SeasonalMultiplier(t);
    LogPopulation(kLogFile, t, preyCount, predatorCount, season,
                  traits.avgPreySpeed, traits.avgPreyPerception,
                  traits.avgPredSpeed, traits.avgPredPerception);

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
