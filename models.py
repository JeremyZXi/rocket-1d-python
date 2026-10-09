"""Vehicle inputs and environmental models. All quantities use SI units."""
from dataclasses import dataclass
from pathlib import Path
import csv
import numpy as np


@dataclass
class Motor:
    time: np.ndarray
    thrust: np.ndarray
    propellant_mass: np.ndarray

    def __post_init__(self):
        self.time, self.thrust, self.propellant_mass = (
            np.asarray(values, dtype=float) for values in
            (self.time, self.thrust, self.propellant_mass))
        if (self.time.ndim != 1 or len(self.time) < 2 or
            self.thrust.shape != self.time.shape or
            self.propellant_mass.shape != self.time.shape or
            not all(np.all(np.isfinite(a)) for a in
                    (self.time, self.thrust, self.propellant_mass)) or
            np.any(np.diff(self.time) <= 0) or self.time[0] < 0 or
            np.any(self.thrust < 0) or np.any(self.propellant_mass < 0) or
            np.any(np.diff(self.propellant_mass) > 0)):
            raise ValueError('Motor curves need matching, finite, ordered samples and nonnegative values; propellant must not increase.')

    def force(self, time):
        return np.interp(time, self.time, self.thrust, left=0.0, right=0.0)

    def fuel_mass(self, time):
        return np.interp(time, self.time, self.propellant_mass)


@dataclass
class Aerodynamics:
    mach: np.ndarray
    drag_coefficient: np.ndarray

    @classmethod
    def from_csv(cls, path: Path):
        with path.open(newline='') as file:
            rows = [row for row in csv.DictReader(file)
                    if abs(float(row['Alpha'])) < 1e-9]
        rows.sort(key=lambda row: float(row['Mach']))
        if len(rows) < 2:
            raise ValueError('CSV must contain at least two zero-angle samples')
        mach = np.array([float(row['Mach']) for row in rows])
        cd = np.array([float(row['CD']) for row in rows])
        if (not np.all(np.isfinite(mach)) or not np.all(np.isfinite(cd)) or
            np.any(np.diff(mach) <= 0) or np.any(cd < 0)):
            raise ValueError('Aerodynamic samples must be finite, unique, and nonnegative')
        return cls(mach, cd)

    def cd(self, mach):
        # Clamp to table endpoints; do not extrapolate beyond supplied data.
        return np.interp(mach, self.mach, self.drag_coefficient)


@dataclass
class Rocket:
    dry_mass: float
    reference_area: float
    motor: Motor
    aerodynamics: Aerodynamics

    def __post_init__(self):
        if not np.isfinite(self.dry_mass) or not np.isfinite(self.reference_area) or self.dry_mass <= 0 or self.reference_area < 0:
            raise ValueError('Dry mass must be positive and reference area nonnegative')

    def mass(self, time):
        return self.dry_mass + self.motor.fuel_mass(time)


def atmosphere(altitude):
    """Simplified atmosphere: lapse rate to 11 km, isothermal above it.

    Returns density and sound speed; this is not a full high-altitude model.
    """
    h = np.maximum(altitude, 0.0)
    temperature = np.maximum(288.15 - 0.0065 * h, 216.65)
    p_trop = 101325 * (216.65 / 288.15) ** (9.80665 / (0.0065 * 287.05))
    pressure = np.where(h <= 11000,
        101325 * (temperature / 288.15) ** (9.80665 / (0.0065 * 287.05)),
        p_trop * np.exp(-9.80665 * (h - 11000) / (287.05 * 216.65)))
    return pressure / (287.05 * temperature), np.sqrt(1.4 * 287.05 * temperature)
