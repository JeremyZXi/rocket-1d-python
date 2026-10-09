# Vertical rocket simulator

Open this directory as a project in PyCharm. Select `.venv/bin/python` as the
existing project interpreter, then run `main.py`.

To set up elsewhere: `python3 -m venv .venv`, then
`.venv/bin/python -m pip install -r requirements.txt`.
Run `.venv/bin/python main.py`. Paths are resolved relative to the script.

Edit the motor samples and rocket parameters in `main.py`. `models.py` describes
inputs and the simplified atmosphere; `solver.py` integrates altitude and velocity
with SciPy and returns NumPy arrays. The output sampling interval is independent
of the adaptive solver step size. CSV output includes the exact apogee event.

Thrust is zero outside its sampled time range. Fuel mass holds its endpoint values
outside that range. The rocket stays on the ground before liftoff. Drag opposes
velocity. Rail exit is an altitude event; this vertical model includes no rail
friction, lateral motion, wind, attitude dynamics, or parachute descent. Atmosphere
above 11 km uses a simplified isothermal layer. Drag coefficients clamp to the
Mach table bounds. CG and stability coefficients are omitted because they do not
enter vertical translation. Maximum speed and Mach are sampled maxima.

Run validation with `.venv/bin/python -m unittest test_solver.py`.
