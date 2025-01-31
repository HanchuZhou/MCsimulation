import carla
import random
import numpy as np
import matplotlib.pyplot as plt

NUM_SIMULATIONS = 1000  # Number of Monte Carlo simulations
EGO_SPEED_RANGE = (5, 50)  # Speed range in m/s for ego vehicle
TIME_STEP = 0.05  # Simulation time step in seconds
SIM_DURATION = 5  # Simulate for 5 seconds per run
COLLISION_DISTANCE = 2.5  # Threshold for collision detection in meters

def get_lane_trajectory(vehicle, world, num_points=100, step=2):
    """Retrieve waypoints along the lane for a vehicle."""
    map = world.get_map()
    traj = []
    waypoint = map.get_waypoint(vehicle.get_location())
    
    for _ in range(num_points):
        traj.append((waypoint.transform.location.x, waypoint.transform.location.y))
        next_waypoints = waypoint.next(step)
        if next_waypoints:
            waypoint = next_waypoints[0]
        else:
            break
            
    return traj

def precompute_background_trajectories(background_vehicles, bg_speeds, world):
    """Precompute the entire motion trajectory for background vehicles before simulations."""
    bg_trajectories = {}

    for i, vehicle in enumerate(background_vehicles):
        traj = get_lane_trajectory(vehicle, world)
        speed = bg_speeds[i]
        bg_trajectories[vehicle.id] = []

        total_time_steps = int(SIM_DURATION / TIME_STEP)

        for t in range(total_time_steps):
            distance_traveled = speed * t * TIME_STEP
            cumulative_distance = 0.0
            bg_x, bg_y = traj[0]  # Default to start position

            for j in range(1, len(traj)):
                prev_x, prev_y = traj[j - 1]
                curr_x, curr_y = traj[j]
                segment_distance = np.linalg.norm(np.array([curr_x, curr_y]) - np.array([prev_x, prev_y]))
                cumulative_distance += segment_distance

                if cumulative_distance >= distance_traveled:
                    ratio = (distance_traveled - (cumulative_distance - segment_distance)) / segment_distance
                    bg_x = prev_x + ratio * (curr_x - prev_x)
                    bg_y = prev_y + ratio * (curr_y - prev_y)
                    break

            bg_trajectories[vehicle.id].append((bg_x, bg_y))

    return bg_trajectories

def simulate_ego_trajectory(ego_start, ego_speed, bg_trajectories, background_vehicles):
    """Simulate ego vehicle motion and check for collision with precomputed background trajectories."""
    ego_x, ego_y = ego_start.location.x, ego_start.location.y
    ego_yaw = np.deg2rad(ego_start.rotation.yaw)
    
    ego_traj = []
    collision = False
    total_time_steps = int(SIM_DURATION / TIME_STEP)

    for t in range(total_time_steps):
        ego_x += ego_speed * TIME_STEP * np.cos(ego_yaw)
        ego_y += ego_speed * TIME_STEP * np.sin(ego_yaw)
        ego_traj.append((ego_x, ego_y))

        # Check for collisions with background vehicles
        for vehicle in background_vehicles:
            bg_x, bg_y = bg_trajectories[vehicle.id][t]
            if np.linalg.norm(np.array([ego_x, ego_y]) - np.array([bg_x, bg_y])) < COLLISION_DISTANCE:
                collision = True

    return ego_traj, collision

def run_monte_carlo_simulation(world, ego_vehicle, background_vehicles, bg_speeds):
    """
    Run Monte Carlo simulations to compute the collision rate of the ego vehicle.
    
    Parameters:
        world (carla.World): The CARLA simulation world.
        ego_vehicle (carla.Vehicle): The ego vehicle that is being evaluated.
        background_vehicles (list[carla.Vehicle]): A list of background vehicles.
        bg_speeds (list[float]): A list of speeds (in m/s) for each background vehicle.
    
    Returns:
        tuple:
            - collision_rate (float): The percentage of simulations where a collision occurs.
            - bg_trajectories (dict): A dictionary containing precomputed trajectories for background vehicles.
    """
    collision_count = 0
    ego_start = ego_vehicle.get_transform()

    bg_trajectories = precompute_background_trajectories(background_vehicles, bg_speeds, world)

    for _ in range(NUM_SIMULATIONS):
        random_speed = random.uniform(*EGO_SPEED_RANGE)
        _, collision = simulate_ego_trajectory(ego_start, random_speed, bg_trajectories, background_vehicles)
        if collision:
            collision_count += 1

    collision_rate = collision_count / NUM_SIMULATIONS
    print(f"Collision rate: {collision_rate:.2%}")
    return collision_rate, bg_trajectories

def visualize_trajectories(world, ego_vehicle, background_vehicles, bg_speeds):
    """
    Visualize Monte Carlo simulation results, including ego vehicle trajectories, background vehicle trajectories, and collision rates.
    
    Parameters:
        world (carla.World): The CARLA simulation world.
        ego_vehicle (carla.Vehicle): The ego vehicle that is being evaluated.
        background_vehicles (list[carla.Vehicle]): A list of background vehicles.
        bg_speeds (list[float]): A list of speeds (in m/s) for each background vehicle.
    """
    plt.figure(figsize=(12, 8))

    collision_rate, bg_trajectories = run_monte_carlo_simulation(world, ego_vehicle, background_vehicles, bg_speeds)

    # Background vehicle trajectories
    for i, vehicle in enumerate(background_vehicles):
        bg_x, bg_y = zip(*bg_trajectories[vehicle.id])
        plt.plot(bg_x, bg_y, linestyle='--', color='gray', alpha=0.7)

        # Background vehicle
        plt.scatter(bg_x[0], bg_y[0], color='purple', marker='x', s=120, 
                    label="BG Start" if i == 0 else "")

    # Run a few ego simulations for visualization
    ego_start = ego_vehicle.get_transform()
    safe_plotted, collision_plotted = False, False

    for _ in range(10):  # Plot only 10 trajectories for clarity
        ego_speed = random.uniform(*EGO_SPEED_RANGE)
        ego_traj, collision = simulate_ego_trajectory(ego_start, ego_speed, bg_trajectories, background_vehicles)
        x, y = zip(*ego_traj)

        if collision:
            plt.plot(x, y, 'r', alpha=0.5, label="Ego (Collision)" if not collision_plotted else "")
            collision_plotted = True
        else:
            plt.plot(x, y, 'b', alpha=0.5, label="Ego (Safe)" if not safe_plotted else "")
            safe_plotted = True

    # Mark ego start position
    plt.scatter(ego_start.location.x, ego_start.location.y, color='black', marker='o', s=120, label="Ego Start")

    # Add simulation parameters text in the top-right corner
    param_text = (f"NUM_SIMULATIONS: {NUM_SIMULATIONS}\n"
                  f"EGO_SPEED_RANGE: {EGO_SPEED_RANGE} m/s\n"
                  f"TIME_STEP: {TIME_STEP} s\n"
                  f"SIM_DURATION: {SIM_DURATION} s\n"
                  f"COLLISION_DISTANCE: {COLLISION_DISTANCE} m\n"
                  f"BG Speeds: {bg_speeds} m/s")

    plt.text(0.05, 0.95, param_text, transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='top', horizontalalignment='left',
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))


    # Move collision rate to the bottom-left corner
    plt.text(0.05, 0.05, f"Collision Rate: {collision_rate:.2%}", transform=plt.gca().transAxes,
             fontsize=14, color='red', bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

    plt.legend()
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Monte Carlo Trajectory Visualization")
    plt.grid()

    # Save the figure
    plt.savefig("collision_simulation.png", dpi=300)
    plt.show()



# Connect to CARLA and execute
# client = carla.Client('localhost', 2000)
# client.set_timeout(10.0)
# world = client.get_world()
# ego_vehicle = world.get_actors().filter('vehicle.*')[0]  # Assume first vehicle is ego
# background_vehicles = [v for v in world.get_actors().filter('vehicle.*') if v.id != ego_vehicle.id]
