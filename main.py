"""Run this file in PyCharm or with .venv/bin/python main.py."""
from pathlib import Path
from models import Motor, Aerodynamics, Rocket
from solver import simulate


def main():
    directory = Path(__file__).resolve().parent
    motor = Motor([0.05, 8.5, 8.51, 9.5],
                  [8133, 7603, 1200, 1000], [44, 5, 5, 0])
    aero = Aerodynamics.from_csv(directory / 'Eureka3_Machnum_RASAero2_sim.CSV')
    rocket = Rocket(dry_mass=46.7, reference_area=0.03243, motor=motor, aerodynamics=aero)
    result = simulate(rocket)
    result.export_csv(directory / 'flight_data.csv')
    print(f'Maximum altitude: {result.altitude.max():.2f} m')
    print(f'Maximum speed: {result.velocity.max():.2f} m/s')
    print(f'Maximum Mach: {result.mach.max():.4f}')
    print(f'Liftoff: {result.liftoff_time} s')
    print(f'Apogee time: {result.apogee_time} s')
    if result.rail_exit_time is not None:
        print(f'Rail exit: {result.rail_exit_time:.3f} s at {result.rail_exit_speed:.2f} m/s')
    print(f'Results: {directory / "flight_data.csv"}')


if __name__ == '__main__':
    main()
