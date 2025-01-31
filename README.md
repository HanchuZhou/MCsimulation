# MCsimulation

Since the script needs to read the vehicle state and map topology from CARLA, it is necessary to run the environment in the carla first. Then the run MC simulation function in the script.

To calculate collision rate,
```python
collision_rate, _ = run_monte_carlo_simulation(world, ego_vehicle, background_vehicles, bg_speeds)
```

To debug and visualize the trajectories of vehicles,
```python
visualize_trajectories(world, ego_vehicle, background_vehicles, bg_speeds)
```
