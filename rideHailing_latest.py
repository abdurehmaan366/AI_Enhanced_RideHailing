"""
AI-Enhanced Ride-Hailing Management System
Implements intelligent driver matching, predictive pricing, and behavioral analysis
Uses: CSP (Constraint Satisfaction), Genetic Algorithm, and Prolog (PySWIP)
"""

import pickle
import math
import random
from datetime import datetime
from enum import Enum
from typing import List, Optional, Tuple, Dict, Callable
from abc import ABC, abstractmethod

# AI Libraries
try:
    from constraint import Problem, AllDifferentConstraint
    CSP_AVAILABLE = True
except ImportError:
    CSP_AVAILABLE = False
    print("⚠️  python-constraint not installed. Install: pip install python-constraint")

try:
    from deap import base, creator, tools, algorithms
    import numpy as np
    GA_AVAILABLE = True
except ImportError:
    GA_AVAILABLE = False
    print("⚠️  DEAP not installed. Install: pip install deap numpy")

try:
    from pyswip import Prolog
    PROLOG_AVAILABLE = True
except ImportError:
    PROLOG_AVAILABLE = False
    print("⚠️  PySWIP not installed. Install: pip install pyswip")

# ==================== ENUMS ====================

class VehicleType(Enum):
    CAR = "CAR"
    BIKE = "BIKE"
    VAN = "VAN"

class PaymentMethod(Enum):
    WALLET = "WALLET"
    CASH = "CASH"
    CARD = "CARD"

class RideStatus(Enum):
    PENDING = "PENDING"
    ACCEPTED = "ACCEPTED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"

# ==================== EXCEPTIONS ====================

class NoDriverAvailableException(Exception):
    pass

class PaymentFailedException(Exception):
    pass

# ==================== CORE CLASSES ====================

class Vehicle:
    def __init__(self, id: str, reg_number: str, model: str, vehicle_type: VehicleType, capacity: int):
        self.id = id
        self.reg_number = reg_number
        self.model = model
        self.type = vehicle_type
        self.capacity = capacity

    def __str__(self):
        return f"{self.id} {self.reg_number} {self.model} {self.type.value}"

class Location:
    def __init__(self, latitude: float, longitude: float, address: str):
        self.latitude = latitude
        self.longitude = longitude
        self.address = address
        self._validate()

    def _validate(self):
        if not (33.53 <= self.latitude <= 35.86):
            raise ValueError("Latitude must be between 33.53 and 35.86 for Islamabad.")
        if not (72.78 <= self.longitude <= 73.23):
            raise ValueError("Longitude must be between 72.78 and 73.23 for Islamabad.")

    def calculate_distance(self, destination: 'Location') -> float:
        R = 6371  # Earth radius in km
        lat_dis = math.radians(destination.latitude - self.latitude)
        lon_dis = math.radians(destination.longitude - self.longitude)
        a = (math.sin(lat_dis / 2) ** 2 +
             math.cos(math.radians(self.latitude)) * 
             math.cos(math.radians(destination.latitude)) *
             math.sin(lon_dis / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    def __str__(self):
        return f"({self.latitude}, {self.longitude}) {self.address}"

class Config:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.surge_on = False
            cls._instance.surge_multiplier = 1.0
            cls._instance.currency_code = "PKR"
        return cls._instance

    @staticmethod
    def get_instance():
        if Config._instance is None:
            Config()
        return Config._instance

# ==================== USER CLASSES ====================

class User(ABC):
    def __init__(self, id: str, name: str, phone: str, email: str):
        self.id = id
        self.name = name
        self.phone = phone
        self.email = email

    @abstractmethod
    def get_role(self) -> str:
        pass

    def __str__(self):
        return f"{self.id} {self.name} {self.phone} {self.email}"

class Rider(User):
    def __init__(self, id: str, name: str, phone: str, email: str):
        super().__init__(id, name, phone, email)
        self.wallet_balance = 0.0
        self.ride_ids = []
        self.average_rating = 0.0
        self.rating_count = 0
        
        # AI Features: Behavioral tracking
        self.total_rides = 0
        self.cancelled_rides = 0
        self.preferred_vehicle_type = None
        self.avg_ride_distance = 0.0
        self.peak_hour_rides = 0

    def get_role(self) -> str:
        return "RIDER"

    def add_to_wallet(self, amount: float):
        if amount > 0:
            self.wallet_balance += amount

    def deduct_from_wallet(self, amount: float) -> bool:
        if amount <= 0:
            return True
        if amount > self.wallet_balance:
            return False
        self.wallet_balance -= amount
        return True

    def add_ride_id(self, ride_id: str):
        self.ride_ids.append(ride_id)

    def update_average_rating(self, new_rating: int):
        if 1 <= new_rating <= 5:
            self.rating_count += 1
            self.average_rating = ((self.average_rating * (self.rating_count - 1)) + new_rating) / self.rating_count

    def get_cancellation_rate(self) -> float:
        """AI Feature: Calculate rider's cancellation behavior"""
        if self.total_rides == 0:
            return 0.0
        return self.cancelled_rides / self.total_rides

    def get_reliability_score(self) -> float:
        """AI Feature: Composite score for rider reliability"""
        cancellation_penalty = self.get_cancellation_rate() * 0.3
        rating_score = self.average_rating / 5.0
        return max(0, rating_score - cancellation_penalty)

class Driver(User):
    def __init__(self, id: str, name: str, phone: str, email: str, vehicle: Vehicle, available: bool = True):
        super().__init__(id, name, phone, email)
        self.vehicle = vehicle
        self.available = available
        self.average_rating = 0.0   # no rating yet
        self.rating_count = 0
        self.assigned_ride_ids = []
        
        # AI Features: Performance tracking
        self.total_rides = 0
        self.completed_rides = 0
        self.cancelled_rides = 0
        self.total_earnings = 0.0
        self.avg_acceptance_time = 0.0
        self.peak_hour_availability = 0.0

    def get_role(self) -> str:
        return "DRIVER"

    def add_ride_id(self, ride_id: str):
        self.assigned_ride_ids.append(ride_id)

    def update_average_rating(self, new_rating: int):
        if 1 <= new_rating <= 5:
            total = self.average_rating * self.rating_count
            self.rating_count += 1
            self.average_rating = (total + new_rating) / self.rating_count


    def get_completion_rate(self) -> float:
        """AI Feature: Calculate driver's completion rate"""
        if self.total_rides == 0:
            return 1.0
        return self.completed_rides / self.total_rides

    def get_performance_score(self) -> float:
        """AI Feature: Composite performance metric"""
        rating_score = self.average_rating / 5.0
        completion_score = self.get_completion_rate()
        return (rating_score * 0.6) + (completion_score * 0.4)

# ==================== RIDE & PAYMENT ====================

class Payment:
    def __init__(self, id: str, ride_id: str, amount: float, method: PaymentMethod):
        self.id = id
        self.ride_id = ride_id
        self.amount = amount
        self.method = method
        self.successful = False
        self.timestamp = datetime.now()

    def mark_success(self):
        self.successful = True

    def mark_failed(self):
        self.successful = False

class Ride:
    def __init__(self, ride_id: str, rider: Rider, pickup: Location, drop: Location):
        self.ride_id = ride_id
        self.rider = rider
        self.pickup = pickup
        self.drop = drop
        self.driver = None
        self.vehicle = None
        self.fare = 0.0
        self.status = RideStatus.PENDING
        self.distance_km = 0.0
        self.estimated_time_minutes = 0
        self.request_time = datetime.now()
        self.start_time = None
        self.end_time = None
        self.rider_rating = 0
        self.driver_rating = 0
        self.payment = None

    def assign_driver(self, driver: Driver):
        self.driver = driver
        self.vehicle = driver.vehicle
        self.status = RideStatus.ACCEPTED

    def mark_started(self):
        self.status = RideStatus.IN_PROGRESS
        self.start_time = datetime.now()

    def mark_completed(self):
        self.status = RideStatus.COMPLETED
        self.end_time = datetime.now()

    def mark_cancelled(self):
        self.status = RideStatus.CANCELLED
        self.end_time = datetime.now()

    def __str__(self):
        driver_str = self.driver.name if self.driver else "NO_DRIVER"
        return f"{self.ride_id} | Rider: {self.rider.name} | Driver: {driver_str} | Status: {self.status.value} | Fare: {self.fare}"

# ==================== SERVICES ====================

class RouteService:
    SPEED = 40.0  # km/h

    def calculate_distance_km(self, start: Location, end: Location) -> float:
        return start.calculate_distance(end)

    def estimate_time_minutes(self, start: Location, end: Location) -> int:
        distance = self.calculate_distance_km(start, end)
        if distance == 0:
            return 0
        time = (distance / self.SPEED) * 60
        return int(time)

class FareCalculator(ABC):
    @abstractmethod
    def calculate_fare(self, ride: Ride, route_service: RouteService, config: Config) -> float:
        pass

class NormalFareCalculator(FareCalculator):
    def __init__(self, base_fare: float, per_km_rate: float):
        self.base_fare = base_fare
        self.per_km_rate = per_km_rate

    def calculate_fare(self, ride: Ride, route_service: RouteService, config: Config) -> float:
        distance = route_service.calculate_distance_km(ride.pickup, ride.drop)
        ride.distance_km = distance
        fare = self.base_fare + (self.per_km_rate * distance)
        ride.fare = max(0, fare)
        return ride.fare

class SurgeFareCalculator(FareCalculator):
    def __init__(self, base_fare: float, per_km_rate: float):
        self.base_fare = base_fare
        self.per_km_rate = per_km_rate

    def calculate_fare(self, ride: Ride, route_service: RouteService, config: Config) -> float:
        distance = route_service.calculate_distance_km(ride.pickup, ride.drop)
        ride.distance_km = distance
        fare = self.base_fare + (self.per_km_rate * distance)
        if config.surge_on:
            fare *= config.surge_multiplier
        ride.fare = max(0, fare)
        return ride.fare

class PaymentService:
    def process_payment(self, ride: Ride, rider: Rider, method: PaymentMethod) -> Payment:
        amount = ride.fare
        if amount < 0:
            raise PaymentFailedException("Fare cannot be negative")

        payment = Payment(f"PAY-{ride.ride_id}", ride.ride_id, amount, method)

        if method == PaymentMethod.WALLET:
            if rider.wallet_balance < amount:
                payment.mark_failed()
                raise PaymentFailedException("Insufficient balance")
            if not rider.deduct_from_wallet(amount):
                payment.mark_failed()
                raise PaymentFailedException("Failed to deduct payment!")
            payment.mark_success()
        elif method in [PaymentMethod.CASH, PaymentMethod.CARD]:
            payment.mark_success()
        else:
            payment.mark_failed()
            raise PaymentFailedException("Unsupported payment method")

        return payment

class RatingService:
    def rate_driver(self, ride: Ride, rating: int):
        if ride and ride.driver and 1 <= rating <= 5:
            ride.driver_rating = rating
            ride.driver.update_average_rating(rating)

    def rate_rider(self, ride: Ride, rating: int):
        if ride and ride.rider and 1 <= rating <= 5:
            ride.rider_rating = rating
            ride.rider.update_average_rating(rating)

class NotificationService:
    def notify_rider(self, rider: Rider, message: str):
        print(f"[RIDER NOTIFICATION to {rider.name}] {message}")

    def notify_driver(self, driver: Driver, message: str):
        print(f"[DRIVER NOTIFICATION to {driver.name}] {message}")

# ==================== ADMIN CLASS ====================

class Admin(User):
    def __init__(self, id: str, name: str, phone: str, email: str, password: str):
        super().__init__(id, name, phone, email)
        self.password = password

    def get_role(self) -> str:
        return "ADMIN"
    
    def verify_password(self, password: str) -> bool:
        return self.password == password

# ==================== AI-ENHANCED SERVICES ====================

class AIDriverMatcher:
    """AI Feature: Intelligent driver matching based on multiple factors"""
    
    def calculate_match_score(self, driver: Driver, ride: Ride, distance_to_pickup: float) -> float:
        """
        Calculate a comprehensive match score considering:
        - Driver performance score
        - Distance to pickup location
        - Driver rating
        - Completion rate
        """
        # Normalize distance (closer is better, max 10km considered)
        distance_score = max(0, 1 - (distance_to_pickup / 10.0))
        
        # Performance metrics
        performance_score = driver.get_performance_score()
        rating_score = driver.average_rating / 5.0
        completion_score = driver.get_completion_rate()
        
        # Weighted combination
        match_score = (
            distance_score * 0.35 +
            performance_score * 0.25 +
            rating_score * 0.25 +
            completion_score * 0.15
        )
        
        return match_score

    def rank_drivers(self, available_drivers: List[Driver], ride: Ride) -> List[Tuple[Driver, float]]:
        """Rank drivers by match score"""
        ranked = []
        for driver in available_drivers:
            # Simulate distance to pickup (in real system, would use actual location)
            distance = random.uniform(0.5, 8.0)
            score = self.calculate_match_score(driver, ride, distance)
            ranked.append((driver, score))
        
        # Sort by score descending
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

class AIPricingEngine:
    """AI Feature: Predictive and dynamic pricing"""
    
    def predict_demand_multiplier(self, pickup: Location, current_time: datetime) -> float:
        """
        Predict demand based on:
        - Time of day (peak hours)
        - Day of week
        - Location patterns
        """
        hour = current_time.hour
        
        # Peak hours: 7-9 AM, 5-8 PM
        if (7 <= hour <= 9) or (17 <= hour <= 20):
            return 1.3
        # Late night premium
        elif hour >= 23 or hour <= 5:
            return 1.2
        else:
            return 1.0

    def calculate_dynamic_fare(self, base_fare: float, distance: float, 
                              demand_multiplier: float, rider: Rider) -> float:
        """Calculate fare with AI adjustments"""
        # Base calculation
        fare = base_fare + (25 * distance)
        
        # Apply demand multiplier
        fare *= demand_multiplier
        
        # Loyalty discount for reliable riders
        reliability_score = rider.get_reliability_score()
        if reliability_score > 0.8 and rider.total_rides > 10:
            fare *= 0.95  # 5% discount
        
        return fare

class AIBehaviorAnalyzer:
    """AI Feature: Analyze user behavior patterns"""
    
    def analyze_rider_pattern(self, rider: Rider) -> dict:
        """Analyze rider behavior and return insights"""
        return {
            'reliability_score': rider.get_reliability_score(),
            'cancellation_rate': rider.get_cancellation_rate(),
            'total_rides': rider.total_rides,
            'average_rating': rider.average_rating,
            'risk_level': 'LOW' if rider.get_cancellation_rate() < 0.1 else 'MEDIUM' if rider.get_cancellation_rate() < 0.25 else 'HIGH'
        }
    
    def analyze_driver_pattern(self, driver: Driver) -> dict:
        """Analyze driver performance and return insights"""
        return {
            'performance_score': driver.get_performance_score(),
            'completion_rate': driver.get_completion_rate(),
            'total_rides': driver.total_rides,
            'average_rating': driver.average_rating,
            'tier': 'GOLD' if driver.get_performance_score() > 0.9 else 'SILVER' if driver.get_performance_score() > 0.75 else 'BRONZE'
        }

# ==================== COMPLETE CSP IMPLEMENTATION FROM SCRATCH ====================
"""
Complete Constraint Satisfaction Problem (CSP) Implementation
NO EXTERNAL CSP LIBRARY REQUIRED

Features:
1. Variables, Domains, and Constraints explicitly defined
2. Backtracking search algorithm
3. Forward checking optimization
4. Constraint propagation
5. Heuristic ordering (Most Constrained Variable)
"""

class CSPVariable:
    """Represents a CSP variable (a ride that needs driver assignment)"""
    def __init__(self, name: str, domain: List):
        self.name = name
        self.domain = domain.copy()  # List of possible driver IDs
        self.value = None  # Assigned driver ID
    
    def __repr__(self):
        return f"Var({self.name}, domain_size={len(self.domain)}, value={self.value})"


class CSPConstraint:
    """Represents a constraint between variables"""
    def __init__(self, variables: List[str], constraint_fn: Callable, name: str = ""):
        self.variables = variables  # List of variable names involved
        self.constraint_fn = constraint_fn
        self.name = name
    
    def is_satisfied(self, assignment: Dict[str, any]) -> bool:
        """Check if constraint is satisfied given current assignment"""
        return self.constraint_fn(assignment, self.variables)
    
    def __repr__(self):
        return f"Constraint({self.name}, vars={self.variables})"


class CSPSolver:
    """
    Complete CSP Solver with Backtracking and Optimizations
    
    Algorithm:
    1. Select unassigned variable (using MRV heuristic)
    2. Try values from domain
    3. Check constraints
    4. Forward check (prune domains)
    5. Recursively solve
    6. Backtrack if needed
    """
    
    def __init__(self):
        self.variables: Dict[str, CSPVariable] = {}
        self.constraints: List[CSPConstraint] = []
        self.assignment: Dict[str, any] = {}
        self.backtrack_count = 0
        self.constraint_checks = 0
    
    def add_variable(self, var: CSPVariable):
        """Add a variable to the CSP"""
        self.variables[var.name] = var
    
    def add_constraint(self, constraint: CSPConstraint):
        """Add a constraint to the CSP"""
        self.constraints.append(constraint)
    
    def is_consistent(self, var_name: str, value: any) -> bool:
        """Check if assigning value to var_name is consistent with constraints"""
        self.constraint_checks += 1
        temp_assignment = self.assignment.copy()
        temp_assignment[var_name] = value
        
        for constraint in self.constraints:
            # Only check constraints involving this variable
            if var_name in constraint.variables:
                # Only check if all variables in constraint are assigned
                if all(v in temp_assignment for v in constraint.variables):
                    if not constraint.is_satisfied(temp_assignment):
                        return False
        
        return True
    
    def select_unassigned_variable(self) -> Optional[str]:
        """
        MRV (Minimum Remaining Values) Heuristic:
        Choose variable with smallest domain (most constrained)
        """
        unassigned = [name for name in self.variables.keys() if name not in self.assignment]
        
        if not unassigned:
            return None
        
        # Choose variable with smallest domain (MRV heuristic)
        return min(unassigned, key=lambda name: len(self.variables[name].domain))
    
    def forward_check(self, var_name: str, value: any) -> Dict[str, List]:
        """
        Forward Checking: Remove inconsistent values from future variables
        Returns: domains_removed (for backtracking)
        """
        domains_removed = {}
        
        for constraint in self.constraints:
            if var_name not in constraint.variables:
                continue
            
            # Find other variables in this constraint
            other_vars = [v for v in constraint.variables if v != var_name and v not in self.assignment]
            
            for other_var in other_vars:
                removed = []
                for val in self.variables[other_var].domain[:]:
                    temp_assignment = self.assignment.copy()
                    temp_assignment[var_name] = value
                    temp_assignment[other_var] = val
                    
                    if not constraint.is_satisfied(temp_assignment):
                        self.variables[other_var].domain.remove(val)
                        removed.append(val)
                
                if removed:
                    domains_removed[other_var] = removed
        
        return domains_removed
    
    def restore_domains(self, domains_removed: Dict[str, List]):
        """Restore domains after backtracking"""
        for var_name, values in domains_removed.items():
            self.variables[var_name].domain.extend(values)
    
    def backtrack(self) -> bool:
        """
        Backtracking search algorithm with forward checking
        Returns: True if solution found, False otherwise
        """
        self.backtrack_count += 1
        
        # Check if assignment is complete
        if len(self.assignment) == len(self.variables):
            return True
        
        # Select unassigned variable (MRV heuristic)
        var_name = self.select_unassigned_variable()
        if var_name is None:
            return True
        
        # Try each value in domain
        for value in self.variables[var_name].domain[:]:
            if self.is_consistent(var_name, value):
                # Assign value
                self.assignment[var_name] = value
                self.variables[var_name].value = value
                
                # Forward checking
                domains_removed = self.forward_check(var_name, value)
                
                # Check if any domain became empty
                if all(len(self.variables[v].domain) > 0 for v in self.variables if v not in self.assignment):
                    # Recursively solve
                    if self.backtrack():
                        return True
                
                # Backtrack: restore domains and remove assignment
                self.restore_domains(domains_removed)
                del self.assignment[var_name]
                self.variables[var_name].value = None
        
        return False
    
    def solve(self) -> Optional[Dict[str, any]]:
        """
        Solve the CSP and return solution
        Returns: assignment dict if solution found, None otherwise
        """
        self.assignment = {}
        self.backtrack_count = 0
        self.constraint_checks = 0
        
        if self.backtrack():
            return self.assignment.copy()
        else:
            return None
    
    def get_statistics(self) -> Dict:
        """Get solver statistics"""
        return {
            'backtrack_count': self.backtrack_count,
            'constraint_checks': self.constraint_checks,
            'variables': len(self.variables),
            'constraints': len(self.constraints)
        }


# ==================== DRIVER MATCHING CSP IMPLEMENTATION ====================

class CSPDriverMatcher:
    """
    COMPLETE CSP IMPLEMENTATION for Driver-Ride Matching
    
    Problem Formulation:
    - Variables: Rides that need drivers (ride1, ride2, ...)
    - Domains: Available drivers for each ride
    - Constraints:
        1. AllDifferent: Each driver assigned to at most one ride
        2. Rating: Driver rating >= 3.5
        3. Distance: Driver within 10km of pickup
        4. Availability: Driver must be available
        5. Performance: Driver completion rate >= 0.75
    """
    
    def __init__(self):
        self.min_rating = 3.5
        self.max_distance = 10.0
        self.min_completion_rate = 0.75
        self.solver = None
    
    def _calculate_distance_to_pickup(self, driver, pickup: 'Location') -> float:
        """Simulate distance from driver to pickup (in real system, use GPS)"""
        # For demo: random distance between 1-12 km
        return random.uniform(1.0, 12.0)
    
    def _build_domain_for_ride(self, ride: 'Ride', available_drivers: List['Driver']) -> List[str]:
        """
        Build domain (list of valid driver IDs) for a ride
        Apply unary constraints here to prune domain
        """
        valid_driver_ids = []
        
        for driver in available_drivers:
            # Unary Constraint 1: Driver must be available
            if not driver.available:
                continue
            
            # Unary Constraint 2: Rating threshold
            if driver.average_rating < self.min_rating:
                continue
            
            # Unary Constraint 3: Completion rate
            if driver.get_completion_rate() < self.min_completion_rate:
                continue
            
            # Unary Constraint 4: Distance to pickup
            distance = self._calculate_distance_to_pickup(driver, ride.pickup)
            if distance > self.max_distance:
                continue
            
            # Unary Constraint 5: Vehicle capacity
            if driver.vehicle.capacity < 1:
                continue
            
            # All unary constraints satisfied
            valid_driver_ids.append(driver.id)
        
        return valid_driver_ids
    
    def find_valid_drivers(self, available_drivers: List['Driver'], 
                          pickup: 'Location', distance_km: float,
                          min_rating: float = 3.5) -> List['Driver']:
        """
        BACKWARD COMPATIBLE: Simple filtering (for single ride)
        This maintains compatibility with existing code
        """
        print("🔍 CSP: Using single-ride filtering mode")
        
        valid_drivers = []
        for driver in available_drivers:
            if not driver.available:
                continue
            if driver.average_rating < min_rating:
                continue
            if driver.get_completion_rate() < self.min_completion_rate:
                continue
            
            distance_to_pickup = self._calculate_distance_to_pickup(driver, pickup)
            if distance_to_pickup > self.max_distance:
                continue
            
            if driver.vehicle.capacity < 1:
                continue
            
            valid_drivers.append(driver)
        
        print(f"✅ CSP: Found {len(valid_drivers)} drivers satisfying constraints")
        return valid_drivers
    
    def solve_multi_ride_assignment(self, pending_rides: List['Ride'], 
                                    available_drivers: List['Driver']) -> Dict[str, str]:
        """
        NEW METHOD: Solve multi-ride assignment using complete CSP
        
        Returns: Dict mapping ride_id -> driver_id
        """
        if not pending_rides or not available_drivers:
            return {}
        
        print(f"\n🔍 CSP SOLVER: Processing {len(pending_rides)} rides with {len(available_drivers)} drivers")
        print("=" * 70)
        
        # Create CSP solver
        self.solver = CSPSolver()
        
        # Create driver lookup
        driver_map = {d.id: d for d in available_drivers}
        
        # STEP 1: Create variables and domains
        print("📊 Building CSP Variables and Domains...")
        for ride in pending_rides:
            domain = self._build_domain_for_ride(ride, available_drivers)
            
            if not domain:
                print(f"⚠️  Ride {ride.ride_id}: No valid drivers in domain (over-constrained)")
                continue
            
            var = CSPVariable(ride.ride_id, domain)
            self.solver.add_variable(var)
            print(f"  Variable: {ride.ride_id}, Domain size: {len(domain)}")
        
        if not self.solver.variables:
            print("❌ No valid variables (all rides over-constrained)")
            return {}
        
        # STEP 2: Add AllDifferent constraint
        print("\n🔗 Adding Constraints...")
        
        # AllDifferent: No two rides can have the same driver
        def all_different_constraint(assignment: Dict, variables: List[str]) -> bool:
            assigned_values = [assignment[v] for v in variables if v in assignment]
            return len(assigned_values) == len(set(assigned_values))
        
        all_vars = list(self.solver.variables.keys())
        all_diff = CSPConstraint(all_vars, all_different_constraint, "AllDifferent")
        self.solver.add_constraint(all_diff)
        print(f"  ✓ AllDifferent constraint (across {len(all_vars)} variables)")
        
        # Binary constraints: Driver quality constraints
        def driver_quality_constraint(assignment: Dict, variables: List[str]) -> bool:
            """Ensure assigned drivers meet quality standards"""
            for var in variables:
                if var in assignment:
                    driver_id = assignment[var]
                    driver = driver_map.get(driver_id)
                    if not driver:
                        return False
                    # Additional quality check
                    if driver.average_rating < self.min_rating:
                        return False
            return True
        
        for var_name in all_vars:
            quality_constraint = CSPConstraint([var_name], driver_quality_constraint, f"Quality_{var_name}")
            self.solver.add_constraint(quality_constraint)
        
        print(f"  ✓ {len(all_vars)} Quality constraints added")
        
        # STEP 3: Solve using backtracking with forward checking
        print("\n🔄 Running CSP Backtracking Algorithm...")
        print("   (Using MRV heuristic + Forward Checking + Constraint Propagation)")
        
        solution = self.solver.solve()
        
        # STEP 4: Report results
        stats = self.solver.get_statistics()
        print("\n" + "=" * 70)
        print("📈 CSP SOLVER STATISTICS:")
        print(f"   Backtracking steps: {stats['backtrack_count']}")
        print(f"   Constraint checks: {stats['constraint_checks']}")
        print(f"   Variables: {stats['variables']}")
        print(f"   Constraints: {stats['constraints']}")
        
        if solution:
            print(f"\n✅ CSP Solution Found: {len(solution)} rides assigned")
            print("=" * 70)
            
            # Print assignment details
            print("\n📋 ASSIGNMENT DETAILS:")
            for ride_id, driver_id in solution.items():
                driver = driver_map[driver_id]
                print(f"  {ride_id} → Driver {driver.name} (Rating: {driver.average_rating:.1f}⭐)")
            
            return solution
        else:
            print("\n❌ No CSP solution found (problem over-constrained)")
            print("=" * 70)
            return {}
    
    def demonstrate_csp(self, pending_rides: List['Ride'], available_drivers: List['Driver']):
        """
        Demonstration method to show CSP working
        Call this from admin menu to see full CSP in action
        """
        print("\n" + "🎯" * 35)
        print("CSP DEMONSTRATION - COMPLETE CONSTRAINT SATISFACTION SOLVER")
        print("🎯" * 35)
        
        solution = self.solve_multi_ride_assignment(pending_rides, available_drivers)
        
        if solution:
            print("\n✨ CSP successfully solved the assignment problem!")
            print(f"Assigned {len(solution)} out of {len(pending_rides)} rides")
        else:
            print("\n⚠️  CSP could not find a valid assignment")
            print("This could mean:")
            print("  - Not enough drivers available")
            print("  - Constraints are too strict")
            print("  - Problem is over-constrained")
        
        print("\n" + "🎯" * 35)

# ==================== GENETIC ALGORITHM ====================

class GeneticRideOptimizer:
    """GA: Optimize multiple ride assignments simultaneously"""
    
    def __init__(self):
        self.population_size = 50
        self.generations = 30
        self.mutation_rate = 0.2
    
    def optimize_batch_assignment(self, pending_rides: List[Ride], 
                                  available_drivers: List[Driver]) -> List[Tuple[Ride, Driver]]:
        """
        Use GA to find optimal driver-ride assignments for multiple rides
        Fitness: Minimize total distance + Maximize average rating
        """
        if not GA_AVAILABLE:
            print("⚠️  GA not available, using greedy assignment")
            return self._greedy_assignment(pending_rides, available_drivers)
        
        if not pending_rides or not available_drivers:
            return []
        
        print(f"🧬 GA: Optimizing {len(pending_rides)} rides with {len(available_drivers)} drivers...")
        
        # Setup GA
        if hasattr(creator, "FitnessMax"):
            del creator.FitnessMax
        if hasattr(creator, "Individual"):
            del creator.Individual
            
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        
        # Gene: driver index for each ride
        def create_individual():
            return [random.randint(0, len(available_drivers)-1) for _ in range(len(pending_rides))]
        
        toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        def evaluate(individual):
            """Fitness function: higher is better"""
            total_score = 0
            used_drivers = set()
            
            for ride_idx, driver_idx in enumerate(individual):
                ride = pending_rides[ride_idx]
                driver = available_drivers[driver_idx]
                
                # Penalty for reusing drivers
                if driver.id in used_drivers:
                    total_score -= 50
                else:
                    used_drivers.add(driver.id)
                
                # Reward for high-rated drivers
                total_score += driver.average_rating * 10
                
                # Penalty for distance (simulated)
                distance_penalty = random.uniform(0, 5)
                total_score -= distance_penalty
            
            return (total_score,)
        
        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutUniformInt, low=0, up=len(available_drivers)-1, indpb=self.mutation_rate)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Run GA
        pop = toolbox.population(n=self.population_size)
        hof = tools.HallOfFame(1)
        
        algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=self.generations, 
                           halloffame=hof, verbose=False)
        
        # Get best solution
        best_individual = hof[0]
        assignments = []
        used_drivers = set()
        
        for ride_idx, driver_idx in enumerate(best_individual):
            ride = pending_rides[ride_idx]
            driver = available_drivers[driver_idx]
            
            # Only assign if driver not already used
            if driver.id not in used_drivers:
                assignments.append((ride, driver))
                used_drivers.add(driver.id)
        
        print(f"✅ GA: Optimized {len(assignments)} assignments")
        return assignments
    
    def _greedy_assignment(self, rides: List[Ride], drivers: List[Driver]) -> List[Tuple[Ride, Driver]]:
        """Fallback: Simple greedy assignment"""
        assignments = []
        available = drivers.copy()
        
        for ride in rides:
            if not available:
                break
            driver = max(available, key=lambda d: d.average_rating)
            assignments.append((ride, driver))
            available.remove(driver)
        
        return assignments

# ==================== PROLOG RULES ====================

class PrologRuleEngine:
    """Prolog: Rule-based decision making"""
    
    def __init__(self):
        self.prolog = None
        if PROLOG_AVAILABLE:
            try:
                self.prolog = Prolog()
                self._initialize_rules()
            except Exception as e:
                print(f"⚠️  Prolog initialization failed: {e}")
                self.prolog = None
    
    def _initialize_rules(self):
        """Define Prolog rules for ride-hailing system"""
        if not self.prolog:
            return
        
        try:
            # Rule 1: Driver eligibility
            self.prolog.assertz("eligible_driver(Driver, Rating, Available) :- Rating >= 3.5, Available = true")
            
            # Rule 2: Surge pricing condition
            self.prolog.assertz("apply_surge(Hour) :- Hour >= 7, Hour =< 9")
            self.prolog.assertz("apply_surge(Hour) :- Hour >= 17, Hour =< 20")
            
            # Rule 3: Premium rider (high reliability)
            self.prolog.assertz("premium_rider(ReliabilityScore) :- ReliabilityScore >= 0.8")
            
            # Rule 4: High-value ride
            self.prolog.assertz("high_value_ride(Fare) :- Fare >= 500")
            
        except Exception as e:
            print(f"⚠️  Prolog rule definition failed: {e}")
    
    def check_driver_eligibility(self, rating: float, available: bool) -> bool:
        """Query Prolog: Is driver eligible?"""
        if not self.prolog:
            # Fallback to simple Python logic
            return rating >= 3.5 and available
        
        try:
            available_atom = "true" if available else "false"
            query = f"eligible_driver(driver, {rating}, {available_atom})"
            result = list(self.prolog.query(query))
            return len(result) > 0
        except:
            return rating >= 3.5 and available
    
    def should_apply_surge(self, hour: int) -> bool:
        """Query Prolog: Should surge pricing apply?"""
        if not self.prolog:
            # Fallback
            return (7 <= hour <= 9) or (17 <= hour <= 20)
        
        try:
            query = f"apply_surge({hour})"
            result = list(self.prolog.query(query))
            return len(result) > 0
        except:
            return (7 <= hour <= 9) or (17 <= hour <= 20)
    
    def is_premium_rider(self, reliability_score: float) -> bool:
        """Query Prolog: Is rider premium?"""
        if not self.prolog:
            return reliability_score >= 0.8
        
        try:
            query = f"premium_rider({reliability_score})"
            result = list(self.prolog.query(query))
            return len(result) > 0
        except:
            return reliability_score >= 0.8

# ==================== REPOSITORIES ====================

class UserRepository:
    def __init__(self):
        self.riders: List[Rider] = []
        self.drivers: List[Driver] = []
        self.admins: List[Admin] = []

    def add_rider(self, rider: Rider):
        self.riders.append(rider)

    def add_driver(self, driver: Driver):
        self.drivers.append(driver)
    
    def add_admin(self, admin: Admin):
        self.admins.append(admin)

    def find_rider_by_phone(self, phone: str) -> Optional[Rider]:
        return next((r for r in self.riders if r.phone == phone), None)

    def find_driver_by_phone(self, phone: str) -> Optional[Driver]:
        return next((d for d in self.drivers if d.phone == phone), None)
    
    def find_admin_by_phone(self, phone: str) -> Optional[Admin]:
        return next((a for a in self.admins if a.phone == phone), None)

    def get_all_riders(self) -> List[Rider]:
        return self.riders

    def get_all_drivers(self) -> List[Driver]:
        return self.drivers

class RideRepository:
    def __init__(self):
        self.rides: List[Ride] = []

    def add_ride(self, ride: Ride):
        self.rides.append(ride)

    def find_ride_by_id(self, ride_id: str) -> Optional[Ride]:
        return next((r for r in self.rides if r.ride_id == ride_id), None)

    def get_rides_for_rider(self, rider_id: str) -> List[Ride]:
        return [r for r in self.rides if r.rider.id == rider_id]

    def get_rides_for_driver(self, driver_id: str) -> List[Ride]:
        return [r for r in self.rides if r.driver and r.driver.id == driver_id]

    def get_all_rides(self) -> List[Ride]:
        return self.rides

# ==================== STORAGE SERVICE ====================

class ObjectFileStorageService:
    RIDERS_FILE = "riders.dat"
    DRIVERS_FILE = "drivers.dat"
    RIDES_FILE = "rides.dat"

    def save_all(self, user_repo: UserRepository, ride_repo: RideRepository):
        self._save_to_file(self.RIDERS_FILE, user_repo.get_all_riders())
        self._save_to_file(self.DRIVERS_FILE, user_repo.get_all_drivers())
        self._save_to_file(self.RIDES_FILE, ride_repo.get_all_rides())

    def _save_to_file(self, filename: str, data: list):
        try:
            with open(filename, 'wb') as f:
                pickle.dump(data, f)
            print(f"✅ Saved {len(data)} object(s) to {filename}")
        except Exception as e:
            print(f"❌ Error saving to {filename}: {e}")

    def load_all(self, user_repo: UserRepository, ride_repo: RideRepository):
        riders = self._load_from_file(self.RIDERS_FILE, [])
        drivers = self._load_from_file(self.DRIVERS_FILE, [])
        rides = self._load_from_file(self.RIDES_FILE, [])
        
        for rider in riders:
            user_repo.add_rider(rider)
        for driver in drivers:
            user_repo.add_driver(driver)
        for ride in rides:
            ride_repo.add_ride(ride)

    def _load_from_file(self, filename: str, default):
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            print(f"No {filename} found, starting fresh.")
            return default
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")
            return default

# ==================== MAIN RIDE SERVICE ====================

class RideService:
    class DriverFareOption:
        def __init__(self, driver: Driver, fare: float, match_score: float = 0.0):
            self.driver = driver
            self.fare = fare
            self.match_score = match_score

    def __init__(self, user_repo: UserRepository, ride_repo: RideRepository,
                 route_service: RouteService, fare_calculator: FareCalculator,
                 payment_service: PaymentService, rating_service: RatingService,
                 notification_service: NotificationService, config: Config):
        self.user_repo = user_repo
        self.ride_repo = ride_repo
        self.route_service = route_service
        self.fare_calculator = fare_calculator
        self.payment_service = payment_service
        self.rating_service = rating_service
        self.notification_service = notification_service
        self.config = config
        
        # AI Components
        self.ai_matcher = AIDriverMatcher()
        self.ai_pricing = AIPricingEngine()
        self.ai_analyzer = AIBehaviorAnalyzer()
        
        # CSP, GA, and Prolog components
        self.csp_matcher = CSPDriverMatcher()
        self.ga_optimizer = GeneticRideOptimizer()
        self.prolog_engine = PrologRuleEngine()

    def login_rider(self, phone: str) -> Optional[Rider]:
        return self.user_repo.find_rider_by_phone(phone)

    def login_driver(self, phone: str) -> Optional[Driver]:
        return self.user_repo.find_driver_by_phone(phone)
    
    def login_admin(self, phone: str, password: str) -> Optional[Admin]:
        admin = self.user_repo.find_admin_by_phone(phone)
        if admin and admin.verify_password(password):
            return admin
        return None

    def get_driver_fare_options(self, pickup: Location, drop: Location, 
                                rider: Rider = None) -> Tuple[Ride, List[DriverFareOption]]:
        """
        AI-Enhanced with CSP + Prolog: Get intelligently matched and priced driver options
        Returns: (pending_ride, driver_options)
        Creates a PENDING ride that can either be:
        - Assigned immediately if rider selects a driver
        - Left PENDING for GA optimization if rider cancels
        """
        
        # STEP 1: Create PENDING ride immediately
        ride_id = f"R{len(self.ride_repo.get_all_rides()) + 1}"
        pending_ride = Ride(ride_id, rider, pickup, drop)
        pending_ride.status = RideStatus.PENDING
        
        # Calculate distance and fare
        distance = self.route_service.calculate_distance_km(pickup, drop)
        pending_ride.distance_km = distance
        pending_ride.estimated_time_minutes = self.route_service.estimate_time_minutes(pickup, drop)
        
        # Add to repository NOW (even before driver selection)
        self.ride_repo.add_ride(pending_ride)
        rider.add_ride_id(ride_id)
        # DON'T increment rider.total_rides yet (only when confirmed/cancelled)
        
        # STEP 2: CSP - Filter drivers by constraints
        all_drivers = self.user_repo.get_all_drivers()
        available_drivers = [d for d in all_drivers if d.available]
        
        valid_drivers = self.csp_matcher.find_valid_drivers(available_drivers, pickup, distance)
        
        if not valid_drivers:
            # Ride stays PENDING for GA optimization
            print("⚠️ No drivers available now. Ride queued for optimization.")
            return pending_ride, []

        # STEP 3: Prolog - Check surge pricing eligibility
        current_hour = datetime.now().hour
        should_surge = self.prolog_engine.should_apply_surge(current_hour)
        
        if should_surge and not self.config.surge_on:
            print("🤖 Prolog suggests surge pricing for current hour")

        # STEP 4: AI - Rank drivers by match score
        ranked_drivers = self.ai_matcher.rank_drivers(valid_drivers, pending_ride)

        # STEP 5: AI - Calculate dynamic pricing with Prolog premium check
        demand_multiplier = self.ai_pricing.predict_demand_multiplier(pickup, datetime.now())
        
        # Prolog: Check if rider is premium
        is_premium = False
        if rider:
            is_premium = self.prolog_engine.is_premium_rider(rider.get_reliability_score())
            if is_premium:
                print("🌟 Prolog: Premium rider detected - applying discount")

        options = []
        for driver, match_score in ranked_drivers[:5]:  # Top 5 matches
            temp_ride = Ride("TEMP", rider, pickup, drop)
            temp_ride.distance_km = distance
            
            base_fare = self.fare_calculator.calculate_fare(temp_ride, self.route_service, self.config)
            
            # Add variance and apply demand
            variance = random.uniform(0.9, 1.1)
            final_fare = base_fare * variance * demand_multiplier
            
            # Premium rider discount (Prolog rule)
            if is_premium:
                final_fare *= 0.95  # 5% discount
            
            final_fare = round(final_fare)
            
            options.append(self.DriverFareOption(driver, final_fare, match_score))

        return pending_ride, options

    def confirm_driver_for_pending_ride(self, pending_ride: Ride, selected_driver: Driver, 
                                       agreed_fare: float, payment_method: PaymentMethod) -> bool:
        """
        Assigns driver to an existing PENDING ride
        """
        driver_in_repo = self.user_repo.find_driver_by_phone(selected_driver.phone)
        
        if not driver_in_repo or not driver_in_repo.available:
            print("❌ Driver no longer available")
            return False
        
        # Assign driver to existing PENDING ride
        pending_ride.assign_driver(driver_in_repo)
        pending_ride.fare = agreed_fare
        
        # Process payment
        try:
            payment = self.payment_service.process_payment(pending_ride, pending_ride.rider, payment_method)
            pending_ride.payment = payment
        except PaymentFailedException as e:
            pending_ride.status = RideStatus.PENDING  # Revert to PENDING
            print(f"❌ Payment failed: {e}")
            return False
        
        # Update records
        pending_ride.rider.total_rides += 1  # NOW increment
        driver_in_repo.add_ride_id(pending_ride.ride_id)
        driver_in_repo.total_rides += 1
        driver_in_repo.available = False
        
        # Notifications
        self.notification_service.notify_driver(driver_in_repo, f"New Ride Assigned {pending_ride.ride_id}")
        self.notification_service.notify_rider(pending_ride.rider, f"Ride confirmed! Driver: {driver_in_repo.name}")
        
        return True

    def leave_ride_pending_for_optimization(self, ride: Ride):
        """
        Rider chose not to select any driver - leave for GA optimization
        """
        print(f"🤖 Ride {ride.ride_id} left PENDING for GA optimization")
        self.notification_service.notify_rider(
            ride.rider, 
            f"Ride {ride.ride_id} queued. We'll find you the best driver shortly!"
        )
        # Ride stays PENDING, no driver assigned

    def cancel_ride(self, rider: Rider, ride_id: str) -> bool:
        ride = self.ride_repo.find_ride_by_id(ride_id)
        if not ride:
            return False
        if ride.status in [RideStatus.CANCELLED, RideStatus.COMPLETED]:
            return False
        if ride.status == RideStatus.PENDING:
            # Can cancel PENDING rides
            ride.mark_cancelled()
            rider.cancelled_rides += 1
            return True
        if ride.status != RideStatus.ACCEPTED:
            return False

        ride.mark_cancelled()
        rider.cancelled_rides += 1

        if ride.driver:
            ride.driver.cancelled_rides += 1
            ride.driver.available = True
            self.notification_service.notify_driver(
                ride.driver, f"Ride {ride_id} was cancelled by {rider.name}"
            )

        return True

    def start_ride(self, driver: Driver, ride_id: str) -> bool:
        ride = self.ride_repo.find_ride_by_id(ride_id)
        
        if not ride:
            print("❌ Ride not found.")
            return False
        if not ride.driver or ride.driver.id != driver.id:
            print("❌ This ride does NOT belong to you.")
            return False
        if ride.status != RideStatus.ACCEPTED:
            print("❌ Ride is not in ACCEPTED state, cannot start.")
            return False

        ride.mark_started()
        self.notification_service.notify_rider(ride.rider, f"Your ride {ride_id} has started.")
        print(f"✅ Ride {ride_id} started successfully.")
        return True

    def complete_ride(self, driver: Driver, ride_id: str) -> bool:
        ride = self.ride_repo.find_ride_by_id(ride_id)
        
        if not ride:
            return False
        if ride.status != RideStatus.IN_PROGRESS:
            print("❌ Can't complete a ride not yet started")
            return False

        ride.mark_completed()
        driver.available = True
        driver.completed_rides += 1
        driver.total_earnings += ride.fare
        
        return True

    def get_rides_for_rider(self, rider: Rider) -> List[Ride]:
        return self.ride_repo.get_rides_for_rider(rider.id)

    def get_rides_for_driver(self, driver: Driver) -> List[Ride]:
        return self.ride_repo.get_rides_for_driver(driver.id)

    def show_ai_insights_rider(self, rider: Rider):
        """AI Feature: Show rider behavior analysis"""
        insights = self.ai_analyzer.analyze_rider_pattern(rider)
        print("\n🤖 AI INSIGHTS - RIDER PROFILE")
        print(f"Reliability Score: {insights['reliability_score']:.2f}")
        print(f"Cancellation Rate: {insights['cancellation_rate']:.1%}")
        print(f"Total Rides: {insights['total_rides']}")
        print(f"Average Rating: {insights['average_rating']:.2f}⭐")
        print(f"Risk Level: {insights['risk_level']}")

    def show_ai_insights_driver(self, driver: Driver):
        """AI Feature: Show driver performance analysis"""
        insights = self.ai_analyzer.analyze_driver_pattern(driver)
        print("\n🤖 AI INSIGHTS - DRIVER PERFORMANCE")
        print(f"Performance Score: {insights['performance_score']:.2f}")
        print(f"Completion Rate: {insights['completion_rate']:.1%}")
        print(f"Total Rides: {insights['total_rides']}")
        print(f"Average Rating: {insights['average_rating']:.2f}⭐")
        print(f"Driver Tier: {insights['tier']}")
    
    def optimize_pending_rides_ga(self) -> int:
        """
        GA Feature: Optimize multiple pending rides at once (Admin function)
        Returns number of optimized assignments
        """
        # Find all pending rides
        all_rides = self.ride_repo.get_all_rides()
        pending_rides = [r for r in all_rides if r.status == RideStatus.PENDING]
        
        if not pending_rides:
            print("❌ No pending rides to optimize")
            return 0
        
        # Get available drivers
        available_drivers = [d for d in self.user_repo.get_all_drivers() if d.available]
        
        if not available_drivers:
            print("❌ No available drivers")
            return 0
        
        print(f"\n🧬 Running Genetic Algorithm for {len(pending_rides)} pending rides...")
        
        # Use GA to optimize assignments
        assignments = self.ga_optimizer.optimize_batch_assignment(pending_rides, available_drivers)
        
        # Apply optimized assignments
        assigned_count = 0
        for ride, driver in assignments:
            ride.assign_driver(driver)
            driver.available = False
            ride.rider.total_rides += 1
            driver.add_ride_id(ride.ride_id)
            driver.total_rides += 1
            
            # Set a calculated fare
            ride.fare = 50 + (25 * ride.distance_km)
            
            self.notification_service.notify_driver(driver, f"Assigned to ride {ride.ride_id} (GA Optimized)")
            self.notification_service.notify_rider(ride.rider, f"Driver {driver.name} assigned (GA Optimized)")
            assigned_count += 1
        
        print(f"✅ GA optimized and assigned {assigned_count} rides")
        return assigned_count
    
    def demonstrate_csp_optimization(self):
        """
        NEW METHOD: Demonstrate CSP multi-ride optimization (Admin function)
        """
        # Find all pending rides
        all_rides = self.ride_repo.get_all_rides()
        pending_rides = [r for r in all_rides if r.status == RideStatus.PENDING]
        
        if not pending_rides:
            print("❌ No pending rides to demonstrate CSP optimization")
            return 0
        
        # Get available drivers
        available_drivers = [d for d in self.user_repo.get_all_drivers() if d.available]
        
        if not available_drivers:
            print("❌ No available drivers")
            return 0
        
        # Use CSP to solve multi-ride assignment
        solution = self.csp_matcher.solve_multi_ride_assignment(pending_rides, available_drivers)
        
        if not solution:
            print("❌ CSP could not find a solution")
            return 0
        
        # Apply CSP assignments
        assigned_count = 0
        for ride_id, driver_id in solution.items():
            ride = self.ride_repo.find_ride_by_id(ride_id)
            driver = next((d for d in available_drivers if d.id == driver_id), None)
            
            if ride and driver:
                ride.assign_driver(driver)
                driver.available = False
                ride.rider.total_rides += 1
                driver.add_ride_id(ride.ride_id)
                driver.total_rides += 1
                
                # Set a calculated fare
                ride.fare = 50 + (25 * ride.distance_km)
                
                self.notification_service.notify_driver(driver, f"Assigned to ride {ride.ride_id} (CSP Optimized)")
                self.notification_service.notify_rider(ride.rider, f"Driver {driver.name} assigned (CSP Optimized)")
                assigned_count += 1
        
        print(f"✅ CSP optimized and assigned {assigned_count} rides")
        return assigned_count

# ==================== CLI PROGRAM ====================

def show_main_menu():
    """Display the main login menu"""
    print("\n" + "="*50)
    print("🚗 AI-ENHANCED RIDE HAILING SYSTEM")
    print("="*50)
    print("1) Login as Rider")
    print("2) Login as Driver")
    print("3) Login as Admin")
    print("4) Exit")
    print("="*50)

def show_rider_menu(rider_name: str):
    """Display rider-specific menu"""
    print("\n" + "="*50)
    print(f"🚗 RIDER MENU - Welcome {rider_name}!")
    print("="*50)
    print("1) Request Ride")
    print("2) Cancel Ride")
    print("3) View My Rides")
    print("4) View AI Insights")
    print("5) Logout")
    print("="*50)

def show_driver_menu(driver_name: str):
    """Display driver-specific menu"""
    print("\n" + "="*50)
    print(f"🚕 DRIVER MENU - Welcome {driver_name}!")
    print("="*50)
    print("1) View Assigned Rides")
    print("2) Start Ride")
    print("3) Complete Ride")
    print("4) View My Rides")
    print("5) View AI Insights")
    print("6) Logout")
    print("="*50)

def show_admin_menu():
    """Display admin-specific menu"""
    print("\n" + "="*50)
    print("👨‍💼 ADMIN MENU - System Management")
    print("="*50)
    print("1) View All Riders")
    print("2) View All Drivers")
    print("3) View All Rides")
    print("4) View System Statistics")
    print("5) Toggle Surge Pricing")
    print("6) 🧬 GA: Optimize Pending Rides")
    print("7) 🎯 CSP: Demonstrate Multi-Ride Assignment")  # NEW!
    print("8) 🤖 View AI Techniques Status")
    print("9) Logout")  # Changed from 8 to 9
    print("="*50)

def main():
    # Initialize repositories
    user_repo = UserRepository()
    ride_repo = RideRepository()
    route_service = RouteService()
    payment_service = PaymentService()
    rating_service = RatingService()
    notification_service = NotificationService()
    config = Config.get_instance()
    storage_service = ObjectFileStorageService()

    if not user_repo.admins:
        admin = Admin(
            "ADM1",
            "System Admin",
            "03000000000",
            "admin@system.com",
            "admin123"
        )
        user_repo.add_admin(admin)

    # Initialize fare calculator
    fare_calculator = SurgeFareCalculator(50, 25) if config.surge_on else NormalFareCalculator(50, 25)

    # Initialize main service
    ride_service = RideService(
        user_repo, ride_repo, route_service, fare_calculator,
        payment_service, rating_service, notification_service, config
    )

    # FIRST load existing data
    storage_service.load_all(user_repo, ride_repo)

    # ONLY seed demo data if repositories are empty
    if not user_repo.get_all_riders() and not user_repo.get_all_drivers():
        print("🔧 Seeding initial riders, drivers, and admin...")

        # Riders
        rider1 = Rider("RID1", "Noah", "03001234567", "noah@mail.com")
        rider1.add_to_wallet(100000)
        user_repo.add_rider(rider1)

        rider2 = Rider("RID2", "Emma", "03009876543", "emma@mail.com")
        rider2.add_to_wallet(50000)
        user_repo.add_rider(rider2)

        # Drivers
        v1 = Vehicle("V1", "ABC-123", "Honda Civic", VehicleType.CAR, 4)
        driver1 = Driver("DRV1", "Ali", "03211223344", "ali@mail.com", v1, True)
        user_repo.add_driver(driver1)

        v2 = Vehicle("V2", "XYZ-456", "Toyota Corolla", VehicleType.CAR, 4)
        driver2 = Driver("DRV2", "Sara", "03331234567", "sara@mail.com", v2, True)
        user_repo.add_driver(driver2)

        v3 = Vehicle("V3", "LMN-789", "Suzuki Cultus", VehicleType.CAR, 4)
        driver3 = Driver("DRV3", "Ahmed", "03445566778", "ahmed@mail.com", v3, True)
        user_repo.add_driver(driver3)

        # # Admin
        # admin = Admin("ADM1", "System Admin", "03000000000", "admin@system.com", "admin123")
        # user_repo.add_admin(admin)


    logged_rider = None
    logged_driver = None
    logged_admin = None
    current_role = None

    while True:
        if current_role is None:
            # Main menu - not logged in
            show_main_menu()
            choice = input("Choose option: ").strip()

            if choice == "1":
                phone = input("Enter rider phone: ").strip()
                logged_rider = ride_service.login_rider(phone)
                if logged_rider:
                    current_role = "RIDER"
                    print(f"✅ Rider logged in: {logged_rider.name}")
                else:
                    print("❌ Rider not found.")

            elif choice == "2":
                phone = input("Enter driver phone: ").strip()
                logged_driver = ride_service.login_driver(phone)
                if logged_driver:
                    current_role = "DRIVER"
                    print(f"✅ Driver logged in: {logged_driver.name}")
                else:
                    print("❌ Driver not found.")

            elif choice == "3":
                phone = input("Enter admin phone: ").strip()
                password = input("Enter admin password: ").strip()
                logged_admin = ride_service.login_admin(phone, password)
                if logged_admin:
                    current_role = "ADMIN"
                    print(f"✅ Admin logged in: {logged_admin.name}")
                else:
                    print("❌ Invalid credentials.")

            elif choice == "4":
                print("\n👋 Thank you for using AI Ride Hailing System!")
                storage_service.save_all(user_repo, ride_repo)
                break

            else:
                print("❌ Invalid choice. Try again.")

        elif current_role == "RIDER":
            # Rider menu
            show_rider_menu(logged_rider.name)
            choice = input("Choose option: ").strip()

            try:
                if choice == "1":
                    # Request Ride
                    print("\n📍 Enter Pickup Location:")
                    p_lat = float(input("  Latitude: "))
                    p_lon = float(input("  Longitude: "))
                    p_addr = input("  Address: ")

                    print("\n📍 Enter Drop Location:")
                    d_lat = float(input("  Latitude: "))
                    d_lon = float(input("  Longitude: "))
                    d_addr = input("  Address: ")

                    try:
                        pickup = Location(p_lat, p_lon, p_addr)
                        drop = Location(d_lat, d_lon, d_addr)

                        # Get PENDING ride + options
                        pending_ride, options = ride_service.get_driver_fare_options(pickup, drop, logged_rider)

                        if not options:
                            print("⚠️ No drivers available. Ride queued for optimization.")
                            storage_service.save_all(user_repo, ride_repo)
                            continue

                        print("\n🤖 AI-MATCHED DRIVERS (CSP Filtered + AI Ranked):")
                        print("-" * 70)
                        for i, opt in enumerate(options, 1):
                            d = opt.driver
                            print(f"{i}) {d.name}")
                            print(f"   Rating: {d.average_rating:.1f}⭐ | Vehicle: {d.vehicle.model}")
                            print(f"   Performance: {d.get_performance_score():.2f}")
                            print(f"   Match Score: {opt.match_score:.2f} | Fare: Rs.{opt.fare:.0f}")
                            print("-" * 70)

                        print("0) Cancel (Queue for Optimization)")
                        choice_driver = int(input("\nChoose driver (or 0 to queue): "))

                        if choice_driver == 0:
                            # Leave PENDING for GA
                            ride_service.leave_ride_pending_for_optimization(pending_ride)
                            print("✅ Ride queued! Admin will optimize soon.")
                            storage_service.save_all(user_repo, ride_repo)
                            continue

                        if choice_driver < 1 or choice_driver > len(options):
                            print("❌ Invalid choice.")
                            continue

                        selected_opt = options[choice_driver - 1]
                        
                        # Confirm driver for existing PENDING ride
                        success = ride_service.confirm_driver_for_pending_ride(
                            pending_ride,
                            selected_opt.driver,
                            selected_opt.fare,
                            PaymentMethod.WALLET
                        )

                        if success:
                            print(f"\n✅ Ride confirmed! ID: {pending_ride.ride_id}")
                            print(f"   Driver: {pending_ride.driver.name}")
                            print(f"   Fare: Rs.{pending_ride.fare:.0f}")
                            print(f"   Distance: {pending_ride.distance_km:.2f} km")
                            print(f"   ETA: {pending_ride.estimated_time_minutes} min")
                            storage_service.save_all(user_repo, ride_repo)
                        else:
                            print("❌ Could not confirm ride")

                    except NoDriverAvailableException as e:
                        print(f"❌ {e}")
                    except PaymentFailedException as e:
                        print(f"❌ Payment failed: {e}")
                    except ValueError as e:
                        print(f"❌ Invalid location: {e}")

                elif choice == "2":
                    # Cancel Ride
                    rides = ride_service.get_rides_for_rider(logged_rider)
                    active_rides = [r for r in rides if r.status in [RideStatus.PENDING, RideStatus.ACCEPTED]]
                    
                    if not active_rides:
                        print("❌ No active rides to cancel.")
                        continue
                    
                    print("\n📋 Your active rides:")
                    for r in active_rides:
                        driver_name = r.driver.name if r.driver else "Pending Assignment"
                        print(f"  {r.ride_id} - Driver: {driver_name} - Status: {r.status.value}")
                    
                    ride_id = input("\nEnter Ride ID to cancel: ").strip()
                    if ride_service.cancel_ride(logged_rider, ride_id):
                        print("✅ Ride cancelled successfully.")
                        storage_service.save_all(user_repo, ride_repo)
                    else:
                        print("❌ Failed to cancel ride.")

                elif choice == "3":
                    # View My Rides
                    rides = ride_service.get_rides_for_rider(logged_rider)
                    print("\n📋 YOUR RIDES:")
                    if not rides:
                        print("No rides found.")
                    else:
                        for r in rides:
                            print(f"\n{r}")
                            print(f"  Distance: {r.distance_km:.2f} km | Status: {r.status.value}")
                            
                            if r.status == RideStatus.COMPLETED and r.driver_rating == 0 and r.driver:
                                rating = int(input(f"Rate driver for {r.ride_id} (1-5): "))
                                rating_service.rate_driver(r, rating)
                                print("⭐ Driver rated!")
                        storage_service.save_all(user_repo, ride_repo)

                elif choice == "4":
                    # View AI Insights
                    ride_service.show_ai_insights_rider(logged_rider)

                elif choice == "5":
                    # Logout
                    print(f"👋 Goodbye, {logged_rider.name}!")
                    logged_rider = None
                    current_role = None

                else:
                    print("❌ Invalid choice. Try again.")

            except Exception as e:
                print(f"❌ An error occurred: {e}")

        elif current_role == "DRIVER":
            # Driver menu
            show_driver_menu(logged_driver.name)
            choice = input("Choose option: ").strip()

            try:
                if choice == "1":
                    # View Assigned Rides
                    rides = ride_service.get_rides_for_driver(logged_driver)
                    accepted = [r for r in rides if r.status == RideStatus.ACCEPTED]
                    
                    if not accepted:
                        print("❌ No assigned rides.")
                    else:
                        print("\n📋 Assigned rides:")
                        for r in accepted:
                            print(f"  {r.ride_id} - Rider: {r.rider.name} - Fare: Rs.{r.fare:.0f}")

                elif choice == "2":
                    # Start Ride
                    rides = ride_service.get_rides_for_driver(logged_driver)
                    accepted = [r for r in rides if r.status == RideStatus.ACCEPTED]
                    
                    if not accepted:
                        print("❌ No rides to start.")
                        continue

                    print("\n📋 Rides waiting to start:")
                    for r in accepted:
                        print(f"  {r.ride_id} - {r.rider.name}")

                    ride_id = input("\nEnter Ride ID to start: ").strip()
                    if ride_service.start_ride(logged_driver, ride_id):
                        storage_service.save_all(user_repo, ride_repo)

                elif choice == "3":
                    # Complete Ride
                    rides = ride_service.get_rides_for_driver(logged_driver)
                    in_progress = [r for r in rides if r.status == RideStatus.IN_PROGRESS]
                    
                    if not in_progress:
                        print("❌ No rides in progress.")
                        continue

                    print("\n📋 Rides in progress:")
                    for r in in_progress:
                        print(f"  {r.ride_id} - {r.rider.name}")

                    ride_id = input("\nEnter Ride ID to complete: ").strip()
                    if ride_service.complete_ride(logged_driver, ride_id):
                        print("✅ Ride completed successfully.")
                        
                        ride = ride_repo.find_ride_by_id(ride_id)
                        if ride:
                            rating = int(input("Rate the rider (1-5): "))
                            rating_service.rate_rider(ride, rating)
                            print("⭐ Rider rated successfully.")
                        
                        storage_service.save_all(user_repo, ride_repo)

                elif choice == "4":
                    # View My Rides
                    rides = ride_service.get_rides_for_driver(logged_driver)
                    print("\n📋 YOUR RIDES:")
                    if not rides:
                        print("No rides found.")
                    else:
                        for r in rides:
                            print(f"\n{r}")
                            print(f"  Distance: {r.distance_km:.2f} km | Status: {r.status.value}")

                elif choice == "5":
                    # View AI Insights
                    ride_service.show_ai_insights_driver(logged_driver)

                elif choice == "6":
                    # Logout
                    print(f"👋 Goodbye, {logged_driver.name}!")
                    logged_driver = None
                    current_role = None

                else:
                    print("❌ Invalid choice. Try again.")

            except Exception as e:
                print(f"❌ An error occurred: {e}")

        elif current_role == "ADMIN":
            # Admin menu
            show_admin_menu()
            choice = input("Choose option: ").strip()

            try:
                if choice == "1":
                    # View All Riders
                    riders = user_repo.get_all_riders()
                    print("\n👥 ALL RIDERS:")
                    print("-" * 80)
                    for r in riders:
                        print(f"ID: {r.id} | Name: {r.name} | Phone: {r.phone}")
                        print(f"Wallet: Rs.{r.wallet_balance:.0f} | Rating: {r.average_rating:.2f}⭐")
                        print(f"Total Rides: {r.total_rides} | Cancelled: {r.cancelled_rides}")
                        print(f"Reliability: {r.get_reliability_score():.2f}")
                        print("-" * 80)

                elif choice == "2":
                    # View All Drivers
                    drivers = user_repo.get_all_drivers()
                    print("\n🚗 ALL DRIVERS:")
                    print("-" * 80)
                    for d in drivers:
                        print(f"ID: {d.id} | Name: {d.name} | Phone: {d.phone}")
                        print(f"Vehicle: {d.vehicle.model} ({d.vehicle.reg_number})")
                        print(f"Available: {'Yes' if d.available else 'No'} | Rating: {d.average_rating:.2f}⭐")
                        print(f"Total Rides: {d.total_rides} | Completed: {d.completed_rides}")
                        print(f"Performance: {d.get_performance_score():.2f}")
                        print("-" * 80)

                elif choice == "3":
                    # View All Rides
                    rides = ride_repo.get_all_rides()
                    print("\n🚕 ALL RIDES:")
                    print("-" * 80)
                    for r in rides:
                        print(f"Ride ID: {r.ride_id} | Status: {r.status.value}")
                        print(f"Rider: {r.rider.name} | Driver: {r.driver.name if r.driver else 'N/A'}")
                        print(f"Fare: Rs.{r.fare:.0f} | Distance: {r.distance_km:.2f} km")
                        print(f"Pickup: {r.pickup.address}")
                        print(f"Drop: {r.drop.address}")
                        print("-" * 80)

                elif choice == "4":
                    # View System Statistics
                    total_riders = len(user_repo.get_all_riders())
                    total_drivers = len(user_repo.get_all_drivers())
                    total_rides = len(ride_repo.get_all_rides())
                    completed = len([r for r in ride_repo.get_all_rides() if r.status == RideStatus.COMPLETED])
                    cancelled = len([r for r in ride_repo.get_all_rides() if r.status == RideStatus.CANCELLED])
                    pending = len([r for r in ride_repo.get_all_rides() if r.status == RideStatus.PENDING])
                    active = len([r for r in ride_repo.get_all_rides() if r.status in [RideStatus.ACCEPTED, RideStatus.IN_PROGRESS]])
                    
                    total_revenue = sum(r.fare for r in ride_repo.get_all_rides() if r.status == RideStatus.COMPLETED)
                    
                    print("\n📊 SYSTEM STATISTICS:")
                    print("="*50)
                    print(f"Total Riders: {total_riders}")
                    print(f"Total Drivers: {total_drivers}")
                    print(f"Total Rides: {total_rides}")
                    print(f"  ✅ Completed: {completed}")
                    print(f"  ❌ Cancelled: {cancelled}")
                    print(f"  ⏳ Pending (GA Queue): {pending}")
                    print(f"  🚗 Active: {active}")
                    print(f"Total Revenue: Rs.{total_revenue:.0f}")
                    print(f"Surge Status: {'ON' if config.surge_on else 'OFF'}")
                    print("="*50)

                elif choice == "5":
                    # Toggle Surge Pricing
                    config.surge_on = not config.surge_on
                    multiplier = config.surge_multiplier if config.surge_on else 1.0
                    print(f"🔄 Surge pricing {'ENABLED' if config.surge_on else 'DISABLED'} (x{multiplier})")

                elif choice == "6":
                    # GA: Optimize Pending Rides
                    assigned = ride_service.optimize_pending_rides_ga()
                    if assigned > 0:
                        storage_service.save_all(user_repo, ride_repo)

                elif choice == "7":
                    # CSP Demonstration
                    pending_rides = [r for r in ride_repo.get_all_rides() if r.status == RideStatus.PENDING]
                    available_drivers = [d for d in user_repo.get_all_drivers() if d.available]
                    
                    if not pending_rides:
                        print("❌ No pending rides to demonstrate CSP")
                    elif not available_drivers:
                        print("❌ No available drivers")
                    else:
                        ride_service.csp_matcher.demonstrate_csp(pending_rides, available_drivers)
                        # Also actually apply the CSP optimization
                        apply = input("\nApply CSP optimization? (y/n): ").strip().lower()
                        if apply == 'y':
                            assigned = ride_service.demonstrate_csp_optimization()
                            if assigned > 0:
                                storage_service.save_all(user_repo, ride_repo)

                elif choice == "8":
                    # View AI Techniques Status
                    print("\n🤖 AI TECHNIQUES STATUS:")
                    print("="*50)
                    print(f"CSP (Constraint Satisfaction): {'✅ Active' if CSP_AVAILABLE else '❌ Not Available'}")
                    print(f"GA (Genetic Algorithm): {'✅ Active' if GA_AVAILABLE else '❌ Not Available'}")
                    print(f"Prolog (Rule Engine): {'✅ Active' if PROLOG_AVAILABLE else '❌ Not Available'}")
                    print("="*50)
                    if CSP_AVAILABLE:
                        print("📋 CSP: Filters drivers by constraints (rating, availability, distance)")
                    if GA_AVAILABLE:
                        print("🧬 GA: Optimizes batch ride assignments for efficiency")
                    if PROLOG_AVAILABLE:
                        print("🔮 Prolog: Rule-based decisions (surge, eligibility, premium)")

                elif choice == "9":
                    # Logout
                    print(f"👋 Goodbye, {logged_admin.name}!")
                    logged_admin = None
                    current_role = None

                else:
                    print("❌ Invalid choice. Try again.")

            except Exception as e:
                print(f"❌ An error occurred: {e}")
                import traceback
                traceback.print_exc()

# ==================== TKINTER GUI ====================

import tkinter as tk
from tkinter import messagebox

# ---------- UI THEME ----------
BG = "#eef2f7"
CARD = "#ffffff"
BTN = "#2563eb"
BTN_TEXT = "white"
BTN_CANCEL = "#f59e0b"
FONT_TITLE = ("Segoe UI", 20, "bold")
FONT_LABEL = ("Segoe UI", 11)
FONT_ENTRY = ("Segoe UI", 11)
FONT_BTN = ("Segoe UI", 11, "bold")

def launch_gui():
    # ---------------- Backend Initialization ----------------
    user_repo = UserRepository()
    ride_repo = RideRepository()
    route_service = RouteService()
    payment_service = PaymentService()
    rating_service = RatingService()
    notification_service = NotificationService()
    config = Config.get_instance()
    storage_service = ObjectFileStorageService()

    fare_calculator = NormalFareCalculator(50, 25)

    ride_service = RideService(
        user_repo, ride_repo, route_service,
        fare_calculator, payment_service,
        rating_service, notification_service, config
    )

    # FIRST load existing data
    storage_service.load_all(user_repo, ride_repo)

    if not user_repo.admins:
        admin = Admin(
            "ADM1",
            "System Admin",
            "03000000000",
            "admin@system.com",
            "admin123"
        )
        user_repo.add_admin(admin)

    # ONLY seed demo data if repositories are empty
    if not user_repo.get_all_riders() and not user_repo.get_all_drivers():
        print("🔧 Seeding initial users for GUI...")

        rider = Rider("RID1", "Noah", "03001234567", "noah@mail.com")
        rider.add_to_wallet(100000)
        user_repo.add_rider(rider)

        rider2 = Rider("RID2", "Emma", "03009876543", "emma@mail.com")
        rider2.add_to_wallet(50000)
        user_repo.add_rider(rider2)

        drivers = [
            Driver("DRV1", "Ali", "03211223344", "ali@mail.com",
                Vehicle("V1", "ABC-123", "Honda Civic", VehicleType.CAR, 4), True),
            Driver("DRV2", "Sara", "03331234567", "sara@mail.com",
                Vehicle("V2", "XYZ-456", "Toyota Corolla", VehicleType.CAR, 4), True),
            Driver("DRV3", "Ahmed", "03445566778", "ahmed@mail.com",
                Vehicle("V3", "LMN-789", "Suzuki Cultus", VehicleType.CAR, 4), True),
        ]
        for d in drivers:
            user_repo.add_driver(d)

        # admin = Admin("ADM1", "System Admin", "03000000000", "admin@system.com", "admin123")
        # user_repo.add_admin(admin)


    # ---------------- ROOT ----------------
    root = tk.Tk()
    root.title("AI Ride Hailing System")
    root.geometry("1000x650")
    root.configure(bg=BG)

    current_user = {"role": None, "obj": None}

    # ---------------- HELPERS ----------------
    def clear():
        for w in root.winfo_children():
            w.destroy()

    def card():
        f = tk.Frame(root, bg=CARD, padx=30, pady=25)
        f.pack(pady=30)
        return f

    def button(parent, text, cmd, bg_color=BTN):
        tk.Button(parent, text=text, bg=bg_color, fg=BTN_TEXT,
                  font=FONT_BTN, width=35,
                  command=cmd).pack(pady=6)

    # ---------------- MAIN MENU ----------------
    def main_menu():
        clear()
        frame = card()

        tk.Label(frame, text="🚗 AI Ride Hailing System",
                 font=FONT_TITLE, bg=CARD).pack(pady=10)

        tk.Label(frame, text="Phone Number", bg=CARD,
                 font=FONT_LABEL).pack(anchor="w")
        phone = tk.Entry(frame, font=FONT_ENTRY, width=35)
        phone.pack(pady=5)

        tk.Label(frame, text="Admin Password (admin only)", bg=CARD,
                 font=FONT_LABEL).pack(anchor="w")
        pwd = tk.Entry(frame, font=FONT_ENTRY, show="*", width=35)
        pwd.pack(pady=5)

        def login(role):
            if role == "RIDER":
                user = ride_service.login_rider(phone.get())
            elif role == "DRIVER":
                user = ride_service.login_driver(phone.get())
            else:
                user = ride_service.login_admin(phone.get(), pwd.get())

            if not user:
                messagebox.showerror("Login Failed", "Invalid credentials")
                return

            current_user["role"] = role
            current_user["obj"] = user

            if role == "RIDER":
                rider_dashboard()
            elif role == "DRIVER":
                driver_dashboard()
            else:
                admin_dashboard()

        button(frame, "Login as Rider", lambda: login("RIDER"))
        button(frame, "Login as Driver", lambda: login("DRIVER"))
        button(frame, "Login as Admin", lambda: login("ADMIN"))

    # ---------------- RIDER DASHBOARD ----------------
    def rider_dashboard():
        clear()
        rider = current_user["obj"]
        
        # Create scrollable frame
        canvas = tk.Canvas(root, bg=BG)
        scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=BG)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        frame = tk.Frame(scrollable_frame, bg=CARD, padx=30, pady=25)
        frame.pack(pady=30, padx=30)

        tk.Label(frame, text=f"👤 Rider Dashboard – {rider.name}",
                 font=FONT_TITLE, bg=CARD).pack(pady=10)

        fields = {}
        for label in ["Pickup Latitude", "Pickup Longitude", "Pickup Address",
                      "Drop Latitude", "Drop Longitude", "Drop Address"]:
            tk.Label(frame, text=label, bg=CARD, font=FONT_LABEL).pack(anchor="w")
            e = tk.Entry(frame, width=40, font=FONT_ENTRY)
            e.pack(pady=3)
            fields[label] = e

        def request_ride():
            try:
                pickup = Location(
                    float(fields["Pickup Latitude"].get()),
                    float(fields["Pickup Longitude"].get()),
                    fields["Pickup Address"].get()
                )
                drop = Location(
                    float(fields["Drop Latitude"].get()),
                    float(fields["Drop Longitude"].get()),
                    fields["Drop Address"].get()
                )

                pending_ride, options = ride_service.get_driver_fare_options(pickup, drop, rider)

                if not options:
                    messagebox.showinfo("Queued", 
                        f"No drivers available. Ride {pending_ride.ride_id} queued for GA optimization!")
                    storage_service.save_all(user_repo, ride_repo)
                    return

                win = tk.Toplevel(root)
                win.title("AI Matched Drivers")
                win.geometry("600x400")
                win.configure(bg=BG)

                tk.Label(win, text="🤖 AI-Matched Drivers", font=FONT_TITLE, bg=BG).pack(pady=10)

                for opt in options:
                    txt = f"{opt.driver.name} | ⭐ {opt.driver.average_rating:.1f} | Rs.{opt.fare} | Match: {opt.match_score:.2f}"
                    tk.Button(win, text=txt, width=60, font=FONT_LABEL,
                              command=lambda o=opt: confirm(o)).pack(pady=4)

                def cancel_queue():
                    ride_service.leave_ride_pending_for_optimization(pending_ride)
                    storage_service.save_all(user_repo, ride_repo)
                    messagebox.showinfo("Queued", 
                        f"Ride {pending_ride.ride_id} queued for GA optimization!")
                    win.destroy()

                tk.Button(win, text="❌ Cancel (Queue for Optimization)", 
                          bg=BTN_CANCEL, fg="white", font=FONT_BTN, 
                          command=cancel_queue).pack(pady=15)

                def confirm(opt):
                    success = ride_service.confirm_driver_for_pending_ride(
                        pending_ride, opt.driver, opt.fare, PaymentMethod.WALLET
                    )
                    if success:
                        storage_service.save_all(user_repo, ride_repo)
                        messagebox.showinfo("Success", 
                            f"Ride {pending_ride.ride_id} confirmed with {opt.driver.name}!")
                    else:
                        messagebox.showerror("Failed", "Could not confirm ride")
                    win.destroy()

            except Exception as e:
                messagebox.showerror("Error", str(e))

        from tkinter import simpledialog
        def rate_driver_popup(ride):
            rating = simpledialog.askinteger(
                "Rate Driver",
                f"Rate your driver {ride.driver.name} (1–5):",
                minvalue=1,
                maxvalue=5
            )

            if rating:
                rating_service.rate_driver(ride, rating)
                storage_service.save_all(user_repo, ride_repo)
                messagebox.showinfo("Thank You", "Driver rated successfully ⭐")


        def view_rides():
            rides = ride_service.get_rides_for_rider(rider)
            win = tk.Toplevel(root)
            win.title("My Rides")
            win.geometry("700x400")

            for r in rides:
                driver_name = r.driver.name if r.driver else "Pending Assignment"

                tk.Label(
                    win,
                    text=f"{r.ride_id} | {r.status.value} | Driver: {driver_name} | Rs.{r.fare:.0f}",
                    font=FONT_LABEL
                ).pack(anchor="w", padx=10, pady=3)

                # ⭐ ADD RATE DRIVER BUTTON (ONLY WHEN APPLICABLE)
                if (
                    r.status == RideStatus.COMPLETED
                    and r.driver
                    and r.driver_rating == 0
                ):
                    tk.Button(
                        win,
                        text="⭐ Rate Driver",
                        font=FONT_BTN,
                        bg=BTN,
                        fg=BTN_TEXT,
                        width=20,
                        command=lambda ride=r: rate_driver_popup(ride)
                    ).pack(padx=30, pady=5)

        def ai_insights():
            insights = ride_service.ai_analyzer.analyze_rider_pattern(rider)
            msg = f"""🤖 AI INSIGHTS - RIDER PROFILE

Reliability Score: {insights['reliability_score']:.2f}
Cancellation Rate: {insights['cancellation_rate']:.1%}
Total Rides: {insights['total_rides']}
Average Rating: {insights['average_rating']:.2f}⭐
Risk Level: {insights['risk_level']}"""
            messagebox.showinfo("AI Insights", msg)

        # Buttons in a more compact layout
        btn_frame = tk.Frame(frame, bg=CARD)
        btn_frame.pack(pady=10)
        
        tk.Button(btn_frame, text="🚕 Request Ride", bg=BTN, fg=BTN_TEXT,
                  font=FONT_BTN, width=17, command=request_ride).grid(row=0, column=0, padx=5, pady=5)
        tk.Button(btn_frame, text="📋 View My Rides", bg=BTN, fg=BTN_TEXT,
                  font=FONT_BTN, width=17, command=view_rides).grid(row=0, column=1, padx=5, pady=5)
        tk.Button(btn_frame, text="🤖 AI Insights", bg=BTN, fg=BTN_TEXT,
                  font=FONT_BTN, width=17, command=ai_insights).grid(row=1, column=0, padx=5, pady=5)
        tk.Button(btn_frame, text="🚪 Logout", bg="#dc2626", fg=BTN_TEXT,
                  font=FONT_BTN, width=17, command=main_menu).grid(row=1, column=1, padx=5, pady=5)

    # ---------------- DRIVER DASHBOARD ----------------
    def driver_dashboard():
        clear()
        driver = current_user["obj"]
        
        # Create scrollable frame
        canvas = tk.Canvas(root, bg=BG)
        scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=BG)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        frame = tk.Frame(scrollable_frame, bg=CARD, padx=30, pady=25)
        frame.pack(pady=30, padx=30)

        tk.Label(frame, text=f"🚕 Driver Dashboard – {driver.name}",
                 font=FONT_TITLE, bg=CARD).pack(pady=10)

        listbox = tk.Listbox(frame, width=70, height=10, font=FONT_LABEL)
        listbox.pack(pady=10)

        def refresh_rides():
            listbox.delete(0, tk.END)
            for r in ride_service.get_rides_for_driver(driver):
                listbox.insert(tk.END, f"{r.ride_id} | {r.status.value} | {r.rider.name}")

        refresh_rides()

        def start():
            try:
                selected = listbox.get(tk.ACTIVE)
                ride_id = selected.split()[0]
                if ride_service.start_ride(driver, ride_id):
                    storage_service.save_all(user_repo, ride_repo)
                    messagebox.showinfo("Started", f"Ride {ride_id} started!")
                    refresh_rides()
            except:
                messagebox.showerror("Error", "Select a ride first")

        def complete():
            try:
                selected = listbox.get(tk.ACTIVE)
                ride_id = selected.split()[0]
                if ride_service.complete_ride(driver, ride_id):
                    ride = ride_repo.find_ride_by_id(ride_id)

                    rating = tk.simpledialog.askinteger(
                        "Rate Rider",
                        "Rate the rider (1–5):",
                        minvalue=1,
                        maxvalue=5
                    )

                    if rating:
                        rating_service.rate_rider(ride, rating)

                    storage_service.save_all(user_repo, ride_repo)
                    messagebox.showinfo("Completed", f"Ride {ride_id} completed!")
                    refresh_rides()

            except:
                messagebox.showerror("Error", "Select a ride first")

        def ai_insights():
            insights = ride_service.ai_analyzer.analyze_driver_pattern(driver)
            msg = f"""🤖 AI INSIGHTS - DRIVER PERFORMANCE

Performance Score: {insights['performance_score']:.2f}
Completion Rate: {insights['completion_rate']:.1%}
Total Rides: {insights['total_rides']}
Average Rating: {insights['average_rating']:.2f}⭐
Driver Tier: {insights['tier']}"""
            messagebox.showinfo("AI Insights", msg)

        # Buttons in compact grid layout
        btn_frame = tk.Frame(frame, bg=CARD)
        btn_frame.pack(pady=10)
        
        tk.Button(btn_frame, text="▶️ Start Ride", bg=BTN, fg=BTN_TEXT,
                  font=FONT_BTN, width=17, command=start).grid(row=0, column=0, padx=5, pady=5)
        tk.Button(btn_frame, text="✅ Complete Ride", bg=BTN, fg=BTN_TEXT,
                  font=FONT_BTN, width=17, command=complete).grid(row=0, column=1, padx=5, pady=5)
        tk.Button(btn_frame, text="🤖 AI Insights", bg=BTN, fg=BTN_TEXT,
                  font=FONT_BTN, width=17, command=ai_insights).grid(row=1, column=0, padx=5, pady=5)
        tk.Button(btn_frame, text="🚪 Logout", bg="#dc2626", fg=BTN_TEXT,
                  font=FONT_BTN, width=17, command=main_menu).grid(row=1, column=1, padx=5, pady=5)

    # ---------------- ADMIN DASHBOARD ----------------
    def admin_dashboard():
        clear()
        
        # Create scrollable frame
        canvas = tk.Canvas(root, bg=BG)
        scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=BG)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        frame = tk.Frame(scrollable_frame, bg=CARD, padx=30, pady=25)
        frame.pack(pady=30, padx=30)

        tk.Label(frame, text="👨‍💼 Admin Dashboard",
                 font=FONT_TITLE, bg=CARD).pack(pady=10)

        def show_riders():
            win = tk.Toplevel(root)
            win.title("All Riders")
            win.geometry("700x400")
            for r in user_repo.get_all_riders():
                tk.Label(win, text=f"{r.name} | Wallet Rs.{r.wallet_balance:.0f} | ⭐ {r.average_rating:.2f} | Rides: {r.total_rides}",
                        font=FONT_LABEL).pack(anchor="w", padx=10, pady=2)

        def show_drivers():
            win = tk.Toplevel(root)
            win.title("All Drivers")
            win.geometry("700x400")
            for d in user_repo.get_all_drivers():
                avail = "✅" if d.available else "❌"
                tk.Label(win, text=f"{d.name} | {d.vehicle.model} | {avail} | ⭐ {d.average_rating:.2f} | Rides: {d.total_rides}",
                        font=FONT_LABEL).pack(anchor="w", padx=10, pady=2)

        def show_rides():
            win = tk.Toplevel(root)
            win.title("All Rides")
            win.geometry("700x400")
            for r in ride_repo.get_all_rides():
                driver_name = r.driver.name if r.driver else "N/A"
                tk.Label(win, text=f"{r.ride_id} | {r.status.value} | Rider: {r.rider.name} | Driver: {driver_name} | Rs.{r.fare:.0f}",
                        font=FONT_LABEL).pack(anchor="w", padx=10, pady=2)

        def toggle_surge():
            config.surge_on = not config.surge_on
            messagebox.showinfo("Surge Pricing",
                                f"Surge is now {'ON' if config.surge_on else 'OFF'}")

        def system_stats():
            total_rides = len(ride_repo.get_all_rides())
            completed = len([r for r in ride_repo.get_all_rides() if r.status == RideStatus.COMPLETED])
            pending = len([r for r in ride_repo.get_all_rides() if r.status == RideStatus.PENDING])
            msg = f"""📊 SYSTEM STATISTICS

Total Rides: {total_rides}
Completed: {completed}
Pending (GA Queue): {pending}
Total Riders: {len(user_repo.get_all_riders())}
Total Drivers: {len(user_repo.get_all_drivers())}"""
            messagebox.showinfo("System Stats", msg)

        def ai_status():
            msg = f"""🤖 AI TECHNIQUES STATUS

CSP: {'✅ Active' if CSP_AVAILABLE else '❌ Inactive'}
GA: {'✅ Active' if GA_AVAILABLE else '❌ Inactive'}
Prolog: {'✅ Active' if PROLOG_AVAILABLE else '❌ Inactive'}"""
            messagebox.showinfo("AI Status", msg)

        def run_ga():
            count = ride_service.optimize_pending_rides_ga()
            storage_service.save_all(user_repo, ride_repo)
            messagebox.showinfo("GA Optimization", f"✅ Optimized {count} rides!")

        def run_csp():
            count = ride_service.demonstrate_csp_optimization()
            storage_service.save_all(user_repo, ride_repo)
            messagebox.showinfo("CSP Optimization", f"✅ CSP optimized {count} rides!")

        # Buttons in compact grid layout (2 columns)
        btn_frame = tk.Frame(frame, bg=CARD)
        btn_frame.pack(pady=10)
        
        tk.Button(btn_frame, text="👥 View Riders", bg=BTN, fg=BTN_TEXT,
                  font=FONT_BTN, width=17, command=show_riders).grid(row=0, column=0, padx=5, pady=5)
        tk.Button(btn_frame, text="🚗 View Drivers", bg=BTN, fg=BTN_TEXT,
                  font=FONT_BTN, width=17, command=show_drivers).grid(row=0, column=1, padx=5, pady=5)
        tk.Button(btn_frame, text="🚕 View Rides", bg=BTN, fg=BTN_TEXT,
                  font=FONT_BTN, width=17, command=show_rides).grid(row=1, column=0, padx=5, pady=5)
        tk.Button(btn_frame, text="📊 Statistics", bg=BTN, fg=BTN_TEXT,
                  font=FONT_BTN, width=17, command=system_stats).grid(row=1, column=1, padx=5, pady=5)
        tk.Button(btn_frame, text="🔄 Toggle Surge", bg=BTN, fg=BTN_TEXT,
                  font=FONT_BTN, width=17, command=toggle_surge).grid(row=2, column=0, padx=5, pady=5)
        tk.Button(btn_frame, text="🧬 GA Optimize", bg=BTN_CANCEL, fg=BTN_TEXT,
                  font=FONT_BTN, width=17, command=run_ga).grid(row=2, column=1, padx=5, pady=5)
        tk.Button(btn_frame, text="🎯 CSP Optimize", bg="#8b5cf6", fg=BTN_TEXT,
                  font=FONT_BTN, width=17, command=run_csp).grid(row=3, column=0, padx=5, pady=5)
        tk.Button(btn_frame, text="🤖 AI Status", bg=BTN, fg=BTN_TEXT,
                  font=FONT_BTN, width=17, command=ai_status).grid(row=3, column=1, padx=5, pady=5)
        tk.Button(btn_frame, text="🚪 Logout", bg="#dc2626", fg=BTN_TEXT,
                  font=FONT_BTN, width=17, command=main_menu).grid(row=4, column=0, padx=5, pady=5, columnspan=2)

    main_menu()
    root.mainloop()


if __name__ == "__main__":
    launch_gui()



'''
CSP:
- variable = un-assigned ride
- domain = drivers fulfilling unary constraints -> available, rating>=3.5, distance<=10km, completion>=0.75, vehicle capacity>=1
- Constraints: AllDiff -> every rides gets unique driver, Driver quality -> rechecks driver rating (for safety purpose)

class CSPSolver -> implements CSP components -> MRV, backtracking, forward checking, Domain restoration
- If 3 rides exist and only 2 can be assigned -> No assignments, all should all fulfil constraints

CSP Logic flow:
1. Build domains (prune early)
2. Create variables
3. Add constraints
4. Backtracking search
5. If ALL variables assigned → SUCCESS
6. Else → FAIL

CSP Usage:
- single ride flow (normal ride booking) -> find_valid_drivers() -> only unary constraints apply to single ride assignments
- Batch assignment -> Admin only option -> demonstrate_csp_optimization -> unary, other constraints and all components apply
--------------------------------------------------------------------------------------------------------------------------------------

GA:
- Individual -> one complete batch assignment -> [driver_idx_for R1, driver_idx_for R2...]
- Population -> many assignments (evolves for 30 generations)
- Fitness -> evaluate(individual)
- Selection, Crossover, Mutation

Fitness logic:
- good rating -> reward
- distance, driver_re-use -> penalty

GA logic flow:
1. Generate random assignments
2. Score them (fitness)
3. Select best
4. Mutate & crossover
5. Repeat
6. Return best individual

GA Usage:
- optimize_pending_rides_ga() -> batch optimization only 
--------------------------------------------------------------------------------------------------------------------------------------

Difference between CSP and GA optimizarion:
- GA (flexible optimization) -> if a rule/condition not satisfied the assignment logic doesnot break, but a penalty is added to overall fitness value
- GA guarantees full assignments, but not always strictly according to rules
- CSP (strict) -> if any constraint not satisfied, Assignment doesnot happen
- CSP guarantees full rules satisfaction, but not always guarantees assignments
--------------------------------------------------------------------------------------------------------------------------------------

Prolog:

Rules:
- if rating>=3.5 AND available then eligible driver
- if hour>=7 AND hour <=9 then apply surge
- if hour>=17 AND hour <=20 then apply surge
- if reliability_score >= 0.8 then rider is premium (helps getting discounts for rides)
--> Prolog tells yes/no, no optimizations

Rules applications:
1. surge rule used in get_driver_fare_options()
2. 
--------------------------------------------------------------------------------------------------------------------------------------

Project flow:

Rider:
1. Create PENDING ride
2. CSP filters valid drivers
3. Prolog checks surge
4. AI ranks drivers
5. Fare calculated
6. Rider selects OR queues ride

Driver:
1. View assigned rides
2. Start ride
3. Complete ride
4. Get rated
5. Availability updated

--> Different AI concepts CSP, GA and Prolog are layered, on different steps different concepts apply
'''