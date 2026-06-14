# AI-Enhanced Ride-Hailing Management System

An AI-Enhanced Ride-Hailing System in Python featuring intelligent driver matching via CSP (backtracking + MRV heuristic), batch ride optimization via Genetic Algorithm, and rule-based decisions via Prolog. Includes dynamic pricing, behavioral analytics, and a Tkinter GUI.

---

## Features

### Core System
- **Rider Management** — Request rides, cancel, view history, rate drivers, AI behavior insights
- **Driver Management** — Accept rides, start/complete rides, rate riders, AI performance insights
- **Admin Panel** — View all riders/drivers/rides, system statistics, surge pricing toggle, AI optimization tools
- **Payment System** — Wallet, cash, and card support with balance validation
- **Rating System** — Mutual post-ride rating between riders and drivers with running average
- **Data Persistence** — Pickle-based file storage; data survives across sessions

### AI Techniques
- **CSP** — Constraint Satisfaction Problem for single and batch driver-ride matching
- **Genetic Algorithm** — Batch optimization of multiple pending rides simultaneously
- **Prolog Rule Engine** — Declarative rule-based decisions for surge pricing, driver eligibility, and premium riders
- **AI Driver Matcher** — Multi-factor match scoring (distance, rating, performance, completion rate)
- **AI Pricing Engine** — Dynamic demand-based fare multipliers (peak hours, late night)
- **AI Behavior Analyzer** — Rider reliability scoring, cancellation rate tracking, driver tier classification

---

## AI Techniques — How They Work

### CSP (Constraint Satisfaction Problem)
**Problem Formulation:**
- Variables → Rides that need drivers
- Domains → Available drivers satisfying unary constraints
- Constraints → AllDifferent (no driver assigned twice) + Quality constraints

**Unary Constraints (domain pruning):**
- Driver must be available
- Rating ≥ 3.5
- Distance to pickup ≤ 10 km
- Completion rate ≥ 0.75
- Vehicle capacity ≥ 1

**Algorithm:** Backtracking search with MRV (Minimum Remaining Values) heuristic and Forward Checking

**Usage:**
- Single ride booking → `find_valid_drivers()` (unary constraints only)
- Batch assignment (Admin) → `demonstrate_csp_optimization()` (full CSP with all constraints)

---

### Genetic Algorithm
**Representation:**
- Individual → One complete batch assignment `[driver_idx_for_R1, driver_idx_for_R2, ...]`
- Population → 50 random assignments, evolved over 30 generations

**Fitness Function:**
- Reward for high driver ratings
- Penalty for reusing the same driver across rides
- Penalty for simulated distance

**Operations:** Two-point crossover, uniform integer mutation, tournament selection

**Usage:** Admin-only → `optimize_pending_rides_ga()` — assigns all pending/queued rides at once

**Key difference from CSP:** GA guarantees full assignments but may bend rules (adds penalty instead of hard rejection). CSP guarantees rule satisfaction but may leave some rides unassigned.

---

### Prolog Rule Engine
**Rules defined:**
```prolog
eligible_driver(Driver, Rating, Available) :- Rating >= 3.5, Available = true
apply_surge(Hour) :- Hour >= 7, Hour =< 9
apply_surge(Hour) :- Hour >= 17, Hour =< 20
premium_rider(ReliabilityScore) :- ReliabilityScore >= 0.8
high_value_ride(Fare) :- Fare >= 500
```

**Applied in:**
- Surge detection during ride booking
- Premium rider discount (5% off fare)
- Driver eligibility checks

Falls back to equivalent Python logic if PySWIP is not installed.

---

## Project Structure (20 Classes)

```
├── Core Models
│   ├── User (ABC)              # Abstract base class for all users
│   ├── Rider                   # Extends User — ride requests, wallet, behavioral tracking
│   ├── Driver                  # Extends User — availability, performance tracking
│   ├── Admin                   # Extends User — password auth, system management
│   ├── Vehicle                 # Composed inside Driver (Car, Bike, Van)
│   ├── Location                # Coordinates, Islamabad validation, Haversine distance
│   ├── Ride                    # Core entity — composes Rider, Driver, Vehicle, Location
│   └── Payment                 # Payment details, method, success status
│
├── Services
│   ├── RouteService            # Distance calculation, ETA estimation
│   ├── FareCalculator (ABC)    # Abstract fare calculator
│   ├── NormalFareCalculator    # Base fare + per-km rate
│   ├── SurgeFareCalculator     # Normal fare × surge multiplier
│   ├── PaymentService          # Wallet/cash/card processing with validation
│   ├── RatingService           # Mutual rider-driver rating with running average
│   └── NotificationService     # Console-based event notifications
│
├── AI Components
│   ├── AIDriverMatcher         # Multi-factor match scoring and ranking
│   ├── AIPricingEngine         # Demand multiplier, loyalty discount
│   ├── AIBehaviorAnalyzer      # Rider reliability, driver tier classification
│   ├── CSPDriverMatcher        # Complete CSP with backtracking, MRV, forward checking
│   ├── GeneticRideOptimizer    # DEAP-based GA for batch ride assignment
│   └── PrologRuleEngine        # PySWIP rule engine with Python fallback
│
├── Repositories
│   ├── UserRepository          # In-memory storage for riders, drivers, admins
│   └── RideRepository          # In-memory storage for all rides
│
├── Storage
│   └── ObjectFileStorageService # Pickle-based persistence (.dat files)
│
└── UI
    ├── main()                  # CLI interface with full menu system
    └── launch_gui()            # Tkinter GUI with scrollable dashboards
```

---

## System Flow

```
Rider Books a Ride:
1. PENDING ride created immediately
2. CSP filters valid drivers (unary constraints)
3. Prolog checks for surge pricing
4. AI ranks remaining drivers by match score
5. AI Pricing calculates dynamic fare
6. Prolog checks if rider is premium (discount)
7. Rider selects driver → confirmed → payment processed
   OR rider queues ride → stays PENDING for GA optimization

Admin GA Optimization:
1. Fetch all PENDING rides
2. Fetch all available drivers
3. GA evolves 50 populations × 30 generations
4. Best individual applied as batch assignment

Admin CSP Optimization:
1. Build CSP variables (one per pending ride)
2. Build domains (drivers passing unary constraints)
3. Add AllDifferent + quality constraints
4. Backtracking with MRV + forward checking
5. Solution applied if found
```

---

## OOP Concepts Demonstrated

| Concept | Where Applied |
|---|---|
| **Abstraction** | `User` (ABC), `FareCalculator` (ABC) — cannot be instantiated directly |
| **Inheritance** | `Rider`, `Driver`, `Admin` extend `User`; `NormalFareCalculator`, `SurgeFareCalculator` extend `FareCalculator` |
| **Encapsulation** | Private methods (`_validate`, `_save_to_file`, `_load_from_file`, `_build_domain_for_ride`) |
| **Composition** | `Ride` contains `Rider`, `Driver`, `Vehicle`, `Location`; `Driver` contains `Vehicle` |
| **Polymorphism** | `FareCalculator.calculate_fare()` overridden in Normal and Surge variants |
| **Singleton** | `Config` — single system-wide configuration instance |
| **Enums** | `VehicleType`, `PaymentMethod`, `RideStatus` |
| **Exception Handling** | `NoDriverAvailableException`, `PaymentFailedException` |

---

## Demo Login Credentials

| Role | Phone | Password |
|---|---|---|
| Rider (Noah) | 03001234567 | — |
| Rider (Emma) | 03009876543 | — |
| Driver (Ali) | 03211223344 | — |
| Driver (Sara) | 03331234567 | — |
| Driver (Ahmed) | 03445566778 | — |
| Admin | 03000000000 | admin123 |

**Sample Islamabad coordinates for testing:**
- Bahria Town: `33.5325, 72.9748`
- F-10 Islamabad: `33.6938, 73.0080`
- Blue Area: `33.7215, 73.0587`

---

## How to Run

**Requirements:** Python 3.8+

**Install dependencies:**
```bash
pip install deap numpy pyswip python-constraint
```

> The system runs fine without these — AI techniques fall back to Python equivalents if libraries are missing.

**Run GUI (recommended):**
```bash
python main.py
```

**Run CLI:**
In `main.py`, change the last line from `launch_gui()` to `main()`:
```python
if __name__ == "__main__":
    main()  # CLI mode
```

**Generated files** (auto-created, do not upload):
```
riders.dat       # Saved rider data
drivers.dat      # Saved driver data
rides.dat        # Saved ride data
```

---

## Course Info

**Course:** Artificial Intelligence  
**University:** COMSATS University Islamabad  
**Department:** Software Engineering — BSE-3A
