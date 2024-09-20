from flask import Flask, render_template, request
import googlemaps
import numpy as np
import random
from time import time

app = Flask(__name__)
gmaps = googlemaps.Client(key='AIzaSyCJD3zGTWBY-_QwgCL3w0eKClNmo4gqnTg')

distance_weight = 1.0
weather_weight = 0.5
road_condition_weight = 0.5
traffic_weight = 1.0
accident_rate_weight = 1.0

route_cache = {}

def get_route_details(origin, destination):
    key = (origin, destination)
    if key in route_cache:
        return route_cache[key]
    
    try:
        directions = gmaps.directions(origin, destination, departure_time='now')
        route = directions[0]['legs'][0]
        distance = route['distance']['value']
        duration = route['duration']['value']
        traffic = route.get('duration_in_traffic', route['duration'])['value']
        route_cache[key] = (distance, duration, traffic, directions[0])
        return distance, duration, traffic, directions[0]
    except Exception as e:
        print(f"Error fetching route details: {e}")
        return float('inf'), float('inf'), float('inf'), None

def get_weather_data(location):
    return random.randint(0, 2)

def get_road_condition(location):
    return random.randint(0, 2)

def get_accident_rate(location):
    return random.randint(0, 2)

def find_nearby_emergency_services(location, service_type, max_results=3):
    try:
        places_result = gmaps.places_nearby(
            location=location,
            radius=10000,
            type=service_type
        )
        services = sorted(
            places_result['results'], 
            key=lambda place: get_route_details(location, (place['geometry']['location']['lat'], place['geometry']['location']['lng']))[0]
        )[:max_results]
        return [(place['geometry']['location']['lat'], place['geometry']['location']['lng']) for place in services]
    except Exception as e:
        print(f"Error finding emergency services: {e}")
        return []

def get_coordinates(area, city):
    try:
        geocode_result = gmaps.geocode(f"{area}, {city}")
        if geocode_result:
            location = geocode_result[0]['geometry']['location']
            return location['lat'], location['lng']
        else:
            print("Error: No results found for the given area and city.")
            exit()
    except Exception as e:
        print(f"Error fetching coordinates: {e}")
        exit()

num_ants = 10
num_iterations = 50
decay = 0.1
alpha = 1
beta = 2

def aco(start_location, emergency_services):
    locations = [start_location] + emergency_services
    num_locations = len(locations)
    pheromones = np.ones((num_locations, num_locations))

    best_routes = []
    best_costs = []
    best_directions = []

    for iteration in range(num_iterations):
        routes = []
        costs = []
        directions_list = []

        for ant in range(num_ants):
            route = [0]
            visited = set(route)

            while len(visited) < len(locations):
                current_location = route[-1]
                probabilities = []

                for next_location in range(len(locations)):
                    if next_location not in visited:
                        distance, duration, traffic, directions = get_route_details(locations[current_location], locations[next_location])
                        weather = get_weather_data(locations[next_location])
                        road_condition = get_road_condition(locations[next_location])
                        accident_rate = get_accident_rate(locations[next_location])

                        cost = (distance_weight * distance +
                                weather_weight * weather +
                                road_condition_weight * road_condition +
                                traffic_weight * traffic +
                                accident_rate_weight * accident_rate)

                        if cost == 0:
                            cost = 1e-10

                        pheromone = pheromones[current_location][next_location] ** alpha
                        heuristic = (1 / cost) ** beta
                        probabilities.append(pheromone * heuristic)
                    else:
                        probabilities.append(0)

                probabilities = np.array(probabilities)
                probabilities = probabilities / probabilities.sum()
                next_location = np.random.choice(range(len(locations)), p=probabilities)
                route.append(next_location)
                visited.add(next_location)

            routes.append(route)
            directions_for_route = []
            route_distance = 0
            cost = 0
            for i in range(len(route) - 1):
                distance, _, traffic, directions = get_route_details(locations[route[i]], locations[route[i + 1]])
                route_distance += distance
                cost += (distance_weight * distance +
                         weather_weight * get_weather_data(locations[route[i + 1]]) +
                         road_condition_weight * get_road_condition(locations[route[i + 1]]) +
                         traffic_weight * traffic +
                         accident_rate_weight * get_accident_rate(locations[route[i + 1]]))
                directions_for_route.append(directions)
            costs.append(cost)
            directions_list.append(directions_for_route)

            if len(best_costs) < 3:
                best_costs.append(cost)
                best_routes.append(route)
                best_directions.append(directions_for_route)
            elif cost < max(best_costs):
                max_index = best_costs.index(max(best_costs))
                best_costs[max_index] = cost
                best_routes[max_index] = route
                best_directions[max_index] = directions_for_route

        pheromones = (1 - decay) * pheromones
        for i, route in enumerate(routes):
            for j in range(len(route) - 1):
                pheromones[route[j]][route[j + 1]] += 1 / costs[i]

    unique_routes = []
    unique_costs = []
    unique_route_coordinates = []
    unique_total_distances_km = []

    for i, route in enumerate(best_routes):
        if route not in unique_routes:
            unique_routes.append(route)
            unique_costs.append(best_costs[i])
            unique_route_coordinates.append([locations[loc] for loc in route])
            total_distance = 0
            for j in range(len(route) - 1):
                total_distance += get_route_details(locations[route[j]], locations[route[j + 1]])[0]
            unique_total_distances_km.append(total_distance / 1000)

    return unique_routes, unique_costs, unique_route_coordinates, unique_total_distances_km, best_directions

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        area = request.form['area']
        city = request.form['city']
        emergency_choice = request.form['emergency_choice'].strip().lower()

        service_types = {
            'hospital': 'hospital',
            'fire_station': 'fire_station',
            'police': 'police'
        }

        if emergency_choice not in service_types:
            return "Invalid choice. Please select from hospital, fire_station, police."

        start_lat, start_lng = get_coordinates(area, city)
        start_location = (start_lat, start_lng)
        emergency_services = find_nearby_emergency_services(start_location, service_types[emergency_choice])

        if not emergency_services:
            return f"No nearby {emergency_choice} found."

        start_time = time()
        best_routes, best_costs, best_route_coordinates, total_distances, best_directions = aco(start_location, emergency_services)
        end_time = time()

        while len(best_route_coordinates) < 3:
            best_route_coordinates.append(best_route_coordinates[-1])
            total_distances.append(total_distances[-1])
            best_directions.append(best_directions[-1])

        waypoints_list = [[{'location': {'lat': coord[0], 'lng': coord[1]}} for coord in route[1:-1]] for route in best_route_coordinates]

        return render_template('route_map.html', start_lat=start_lat, start_lng=start_lng, emergency_choice=emergency_choice, waypoints_list=waypoints_list, total_distances=total_distances)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
