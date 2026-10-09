"""Physics checks independent of the integration implementation."""
import unittest
import numpy as np
from models import Motor, Aerodynamics, Rocket
from solver import simulate


class FlightTests(unittest.TestCase):
    def rocket(self, thrust):
        return Rocket(1, 0, Motor([0, 2], [thrust, thrust], [0, 0]),
                      Aerodynamics(np.array([0, 1]), np.array([0, 0])))

    def test_ground_support(self):
        result = simulate(self.rocket(1), max_time=3)
        self.assertIsNone(result.liftoff_time)
        np.testing.assert_array_equal(result.altitude, 0)
        np.testing.assert_array_equal(result.velocity, 0)

    def test_propellant_boundary(self):
        motor = Motor([0.05, 2], [20, 0], [5, 0])
        self.assertEqual(motor.fuel_mass(0), 5)
        self.assertEqual(motor.fuel_mass(3), 0)
        self.assertEqual(motor.force(0), 0)

    def test_analytic_powered_and_ballistic_flight(self):
        # Cancel inverse-square variation to compare with constant-g equations.
        # Tiny flight height keeps inverse-square correction below 1e-4 m.
        rocket = self.rocket(10.80665)  # 1 m/s^2 net for two seconds
        result = simulate(rocket, max_time=4, rail_length=1)
        expected_peak = 2 + 2**2 / (2 * 9.80665)
        self.assertAlmostEqual(result.altitude[-1], expected_peak, delta=1e-4)
        self.assertAlmostEqual(result.apogee_time, 2 + 2 / 9.80665, delta=1e-5)
        self.assertAlmostEqual(result.rail_exit_time, np.sqrt(2), delta=1e-5)
        self.assertAlmostEqual(result.velocity[-1], 0, delta=1e-8)


if __name__ == '__main__':
    unittest.main()
