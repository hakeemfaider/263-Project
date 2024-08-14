import random

import pandas as pd
import numpy as np
from math import ceil
from pulp import *
import matplotlib.pyplot as plt
import matplotlib
from random import randint
from datetime import datetime
import seaborn as sns
import copy
import statsmodels.stats.api as sms


def create_data():
    df = pd.read_csv('WoolworthsDemand2024.csv')

    df_transposed = df.set_index('Store').transpose().reset_index()
    df_transposed.columns.name = None  # Remove the name of the columns

    # Weekdays:
    df_transposed_weekdays1 = df_transposed.iloc[0:5, :]
    df_transposed_weekdays2 = df_transposed.iloc[7:12, :]
    df_transposed_weekdays3 = df_transposed.iloc[14:19, :]
    df_transposed_weekdays4 = df_transposed.iloc[21:26, :]

    df_weekdays = pd.concat(
        [df_transposed_weekdays1, df_transposed_weekdays2, df_transposed_weekdays3, df_transposed_weekdays4])
    numeric_df_weekdays = df_weekdays.drop(columns=['index'])
    numeric_df_weekdays = numeric_df_weekdays.apply(pd.to_numeric, errors='coerce')
    weekdays = np.ceil(numeric_df_weekdays.quantile(0.75)).to_frame().reset_index()
    weekdays.columns = ["Store", "Upper Quartile Weekday Demand"]

    # Saturdays:
    df_transposed_saturdays1 = df_transposed.iloc[5, :].to_frame().transpose()
    df_transposed_saturdays2 = df_transposed.iloc[12, :].to_frame().transpose()
    df_transposed_saturdays3 = df_transposed.iloc[19, :].to_frame().transpose()
    df_transposed_saturdays4 = df_transposed.iloc[26, :].to_frame().transpose()
    df_saturdays = pd.concat(
        [df_transposed_saturdays1, df_transposed_saturdays2, df_transposed_saturdays3, df_transposed_saturdays4])
    numeric_df_saturdays = df_saturdays.drop(columns=['index'])
    numeric_df_saturdays = numeric_df_saturdays.apply(pd.to_numeric)
    saturdays = np.ceil(numeric_df_saturdays.quantile(0.75)).to_frame().reset_index()
    saturdays.columns = ["Store", "Upper Quartile Saturday Demand"]

    final_df = weekdays.merge(saturdays)

    weekdays_list_demand = weekdays["Upper Quartile Weekday Demand"].to_list()
    weekdays_list_demand.append(0)
    saturdays_list_demand = saturdays["Upper Quartile Saturday Demand"].to_list()
    saturdays_list_demand.append(0)

    return weekdays_list_demand, saturdays_list_demand
def search_nodes(starting_node, map_data, max_time, pallet_data, day):
    possible_nodes = []
    index = map_data.columns.get_loc(starting_node) - 1
    for i in range(len(map_data)):
        if map_data.iloc[index].iloc[i + 1] <= max_time and map_data.iloc[index].iloc[i + 1] != 0:
            if map_data.iloc[i].iloc[0] != "Distribution Centre Auckland" and map_data.iloc[i].iloc[0] != starting_node:
                if pallet_data[day][map_data.iloc[i].iloc[0]] != 0:
                    possible_nodes.append(map_data.iloc[i].iloc[0])
    return possible_nodes


def delete_short(minimum):
    global total_possible_paths
    non_short = 0
    while non_short != len(total_possible_paths):
        if len(total_possible_paths[non_short]) <= minimum:
            total_possible_paths.pop(non_short)
        else:
            non_short += 1
    if len(total_possible_paths[-1]) <= minimum:
        total_possible_paths = total_possible_paths[0:-1]

def possible_node_loop(nodes, pallets, max_pallets, max_time, map_data,
                       count, min_nodes, p_data, day):
    global total_possible_paths
    global current_route
    # If our current node can't see any viable nodes, return to the previous node with updated values
    if len(nodes) == 0:
        if current_route != ['Distribution Centre Auckland']:
            total_possible_paths.append(current_route.copy())
    else:
        check_node_counter = 0
        for j in range(len(nodes)):
            # If the node sees a viable not it hasn't come from
            if nodes[j] not in current_route:
                temp_pallets = pallets + p_data[day][nodes[j]]
                # If the new number of demand pallets is within the trucks limits
                if temp_pallets <= max_pallets:
                    # and we haven't previously visited this route
                    current_route.append(nodes[j])
                    if current_route in total_possible_paths:
                        check_node_counter += 1
                        if current_route != ['Distribution Centre Auckland']:
                            current_route = current_route[0:-1]
                    elif current_route not in total_possible_paths:
                        check_node_counter += 1
                        count = count + 1
                        temp_possible_nodes = search_nodes(nodes[j], map_data, max_time, p_data, day)
                        # Continue the search with the new node
                        possible_node_loop(temp_possible_nodes, temp_pallets, max_pallets, max_time, map_data,
                                           count, min_nodes, p_data, day)
                        if current_route != ['Distribution Centre Auckland']:
                            current_route = current_route[0:-1]
                    # If we've previously visited the node we have to loop back and check the next node
                    else:
                        check_node_counter += 1
                        if current_route != ['Distribution Centre Auckland']:
                            current_route = current_route[0:-1]
                else:
                    check_node_counter += 1
            else:
                check_node_counter += 1
        if check_node_counter == len(nodes):
            if current_route not in total_possible_paths:
                total_possible_paths.append(current_route.copy())


def calc_times():
    global total_possible_paths
    global all_stores
    global map_data
    global final_data
    global day_type
    path_times = {}
    for i in range(len(total_possible_paths)):
        current_path_time_sum = 0
        for j in range(len(total_possible_paths[i]) - 1):
            index_one = all_stores.index(total_possible_paths[i][j])
            index_two = all_stores.index(total_possible_paths[i][j + 1]) + 1
            time_taken = map_data.iloc[index_one].iloc[index_two]
            current_path_time_sum += time_taken
        for store in total_possible_paths[i]:
            current_path_time_sum += final_data[day_type][store] * 60 * 10
        # All times are in hours and cost even if you need 5 mins into next hour, given in seconds currently
        current_path_time_sum = ceil(current_path_time_sum/3600)
        route_name = str(i)
        path_times.update({route_name: current_path_time_sum})

    return path_times


def calc_updated_times(missed_stores, loop_routes):
    global all_stores
    global map_data
    global final_data
    global day_type
    path_times = {}
    for i in range(len(loop_routes)):
        current_path_time_sum = 0
        for missed in missed_stores:
            if missed in loop_routes[i]:
                loop_routes[i].remove(missed)
        for j in range(len(loop_routes[i]) - 1):
            index_one = all_stores.index(final_routes[i][j])
            index_two = all_stores.index(final_routes[i][j + 1]) + 1
            time_taken = map_data.iloc[index_one].iloc[index_two]
            rng_time = np.random.normal(time_taken, time_taken/2)
            while rng_time <= 0:
                rng_time = np.random.normal(time_taken, time_taken/2)
            current_path_time_sum += rng_time
        for store in loop_routes[i]:
            current_path_time_sum += final_data[day_type][store] * 60 * 10
        # All times are in hours and cost even if you need 5 mins into next hour, given in seconds currently
        current_path_time_sum = ceil(current_path_time_sum/3600)
        route_name = str(i)
        path_times.update({route_name: current_path_time_sum})
    return path_times


map_data = pd.read_csv('WoolworthsDurations.csv')
data = pd.read_csv('WoolworthsDemand2024.csv')
store_data = pd.read_csv('WoolworthsLocations.csv')
all_stores = []
for i in range(len(data)):
    all_stores.append(data.iloc[i].iloc[0])
all_stores.append('Distribution Centre Auckland')

# Create the Upper Quartile Data for each store as decided
weekdays, weekend = create_data()
final_data = pd.DataFrame([weekdays, weekend], index=["Weekdays", "Saturday"], columns=all_stores)
final_data = final_data.transpose()

# Min nodes : the amount of shops, other than Distribution centre a truck MUST visit before returning
# Max_time : the max time a truck is allowed to take between those shops

min_nodes = 1
max_time = 1000
used_nodes = []
max_pallets = 20
total_possible_paths = []
trucks = 24
shifts = 2

# Change type of day here, Weekdays or Saturday

day_type = 'Weekdays'
# day_type = 'Saturday'

# This code picks a node a truck could drive to to start, and then searches for nodes around it. Only go to it if it
# has demand
for stores in all_stores:
    if final_data[day_type][stores] > 0:
        total_node_counter = 1
        # Set up the 'route' aka the set of connected nodes
        current_route = ["Distribution Centre Auckland", stores]
        # Set up how many pallets are required at that first store
        current_pallets = final_data[day_type][stores]
        starting_node = stores
        # Search for nodes that satisfy the time criteria around our current node
        possible_nodes = search_nodes(starting_node, map_data, max_time, final_data, day_type)
        # Repeat the same loop of searching for nodes and going to them, until we reach a dead end, then add that route
        # to the possible paths
        possible_node_loop(possible_nodes, current_pallets, max_pallets, max_time, map_data,
                                            total_node_counter, min_nodes, final_data, day_type)



# Function deletes the paths we deem too short as per our node requirements
delete_short(min_nodes)

# Add distribution centre on to the end of each route to complete the route
for i in range(len(total_possible_paths)):
    total_possible_paths[i].append('Distribution Centre Auckland')

# Show our possible paths

'''for i in range(len(total_possible_paths)):
    print(total_possible_paths[i])'''

# Create a list with all the nodes that we can possibly visit using our routes
see_all_companies = []
for i in range(len(total_possible_paths)):
    for j in range(len(total_possible_paths[i])):
        if total_possible_paths[i][j] not in see_all_companies:
            see_all_companies.append(total_possible_paths[i][j])

all_viable_stores = []
for stores in all_stores:
    if final_data[day_type][stores] > 0:
        all_viable_stores.append(stores)
print("Our routes can cover " + str(len(see_all_companies)) + " nodes, including Distribution Centre")
if len(see_all_companies) != len(all_viable_stores) + 1:
    print(set(all_viable_stores) - set(see_all_companies))
    print("were/was missed out")
else:
    print("All nodes can be visited in some way from these routes. Additional routes may be needed if routes overlap.")

# --------------------------------------------------------------------------------------------------

# Start LP Solving

# Cost = 250 * Each truck hours + 325 * any extra hour per truck that goes over + 2300 * Mainfreight 4 hour time

""" Notes : If number of routes > 48, we must use Mainfreight trucks
            If any route > 4 hours, + 325 * hours extra.
            Each location can only be visited once. """

# Function creates a dictionary from the routes calculated with the respective route and time the truck would take
route_times = calc_times()

nodes = {}
# Will use for visiting each node exactly once. Remove the first and last items means that it just has shops
for i in range(len(total_possible_paths)):
    total_possible_paths[i] = total_possible_paths[i][1:-1]
    nodes.update({list(route_times.keys())[i]: total_possible_paths[i]})

too_long_routes = []
max_hours = 4

for routes in route_times:
    if route_times[routes] > max_hours:
        too_long_routes.append(routes)
for i in too_long_routes:
    route_times.pop(i)
    nodes.pop(i)

under_price = 250
over_price = 325
route_times_check = {}
for i in route_times:
    if route_times[i] > max_hours:
        route_times_check.update({i: 1})
    else:
        route_times_check.update({i: 0})

overlapping = {}
for i in all_viable_stores[0:-1]:
    temp_array = []
    for j in list(nodes.keys()):
        if i in nodes[j]:
            temp_array.append(j)
    overlapping.update({i: temp_array})


prob = LpProblem("RoutePlanning", LpMinimize)

route_vars = LpVariable.dicts("Route", list(route_times.keys()), 0, 1, cat=LpBinary)


prob += (lpSum([route_vars[i] * ((under_price * (route_times[i] - (route_times[i] - max_hours) * route_times_check[i])
                                  + over_price * (route_times[i] - max_hours) * route_times_check[i])) for i
                in route_vars])
         , "Objective Loss Function")

prob += lpSum([route_vars[i] for i in route_vars]) <= trucks * shifts, "Trucks before extra needed"
# Must visit each node exactly once
for i in overlapping:
    prob += lpSum([route_vars[j] for j in overlapping[i]]) == 1


prob.writeLP('Routes.lp')
prob.solve()


# The status of the solution is printed to the screen
print("Status:", LpStatus[prob.status])

# The optimised objective function valof Ingredients pue is printed to the screen
print("Total Cost = $", value(prob.objective))

route_count = 0
# Each of the variables is printed with its resolved optimum value
final_routes = []
check_location_array = []
for v in prob.variables():
    if v.varValue == 1:
        route_count += 1
        demand_req = 0
        print(v.name, "=", v.varValue)
        index = v.name.strip("Route_")
        print(f"Hours needed is {route_times[index]}")
        nodes[index].insert(0, "Distribution Centre Auckland")
        nodes[index].append("Distribution Centre Auckland")
        final_routes.append(nodes[index])
        print(f"Route is {nodes[index]}")
        for i in nodes[index]:
            demand_req += final_data[day_type][i]
        print(f"Demand Required was {demand_req} pallets")
        '''
        uncomment to plot map of straight routes
        x_values = []
        y_values = []
        for visits in nodes[index]:
            for i in store_data.index:
                if store_data.loc[i]['Store'] == visits:
                    x_values.append(store_data.loc[i]['Long'])
                    y_values.append(store_data.loc[i]['Lat'])
        plt.plot(x_values, y_values)
        for j in range(len(x_values)):
            plt.scatter(x_values[j], y_values[j], color='black')
        '''
        for locations in nodes[index]:
            check_location_array.append(locations)
# plt.show() uncomment to plot straight routes

for locations in check_location_array:
    if locations not in all_stores:
        print("This store has not been visited with these conditions: ")
        print(locations)

print(f"We need {route_count} routes for this solution")


# ---------------------------------------------------------------------------
runs = 100
mainfreight_count = 0
max_mainfreight = 0
max_cost = 0
run = 0
data_points = []
while run < runs:
    temp_finished_routes = copy.deepcopy(final_routes)
    random_data = copy.deepcopy(final_data)
    for shop in list(random_data.index):
        random_demand = np.random.normal(final_data[day_type][shop], final_data[day_type][shop]/2)
        while random_demand < 0 or random_demand > 20:
            random_demand = np.random.normal(final_data[day_type][shop], final_data[day_type][shop] / 2)
        random_data.loc[shop, day_type] = round(random_demand)
    '''
    Section can be activated to test randomness in routes due to traffic (its normally distributed? May change)
    '''
    cost = 0
    unvisited_stores = []
    for routes in temp_finished_routes:
        demand_req = 0
        demand_og = 0
        for store in routes:
            demand_req += random_data[day_type][store]
            demand_og += final_data[day_type][store]
            if demand_req > 20 and store != "Distribution Centre Auckland":
                unvisited_stores.append(store)
    for i in range(len(final_routes)):
        if "Distribution Centre Auckland" not in temp_finished_routes[i]:
            final_routes[i].append("Distribution Centre Auckland")
            final_routes[i].insert(0, "Distribution Centre Auckland")
    new_route_times = calc_updated_times(unvisited_stores, temp_finished_routes)

    '''
    Truck visits store, if it has enough pallets, and is under the 4 hour barrier, then go to that store
    '''
    additional_route_times = {}
    current_time = 0
    current_pallets = 0
    additional_routes = []
    now_visited_stores = []
    index_one = all_stores.index('Distribution Centre Auckland')
    for store in unvisited_stores:
        current_route = []
        if store not in now_visited_stores:
            current_route.append(store)
            now_visited_stores.append(store)
            current_pallets += random_data[day_type][store]
            index_two = all_stores.index(store)
            random_time = np.random.normal(map_data.iloc[index_one].iloc[index_two],
                                           map_data.iloc[index_one].iloc[index_two]/2)
            while random_time < 0:
                random_time = np.random.normal(map_data.iloc[index_one].iloc[index_two],
                                               map_data.iloc[index_one].iloc[index_two] / 2)
            current_time += random_time + (60 * 10 + current_pallets)
            for extra_store in unvisited_stores:
                index_three = all_stores.index(extra_store)
                random_time_two = np.random.normal(map_data.iloc[index_two].iloc[index_three],
                                                   map_data.iloc[index_two].iloc[index_three]/2)
                while random_time_two < 0:
                    random_time_two = np.random.normal(map_data.iloc[index_two].iloc[index_three],
                                                       map_data.iloc[index_two].iloc[index_three] / 2)
                random_time_three = np.random.normal(map_data.iloc[index_three].iloc[index_one],
                                                     map_data.iloc[index_three].iloc[index_one]/2)
                while random_time_three < 0:
                    random_time_three = np.random.normal(map_data.iloc[index_three].iloc[index_one],
                                                         map_data.iloc[index_three].iloc[index_one] / 2)
                would_be_time = (current_time + random_time_two + 60 * 10 *
                                 random_data[day_type][extra_store] + random_time_three)
                if (current_pallets + random_data[day_type][extra_store]) <= 20 and would_be_time <= 60 * 60 * 4:
                    current_route.append(extra_store)
                    now_visited_stores.append(extra_store)
                    additional_routes.append(current_route)
                    additional_route_times.update({current_route[0]: round(would_be_time/ 3600)})
                    break
                elif extra_store == unvisited_stores[-1]:
                    additional_routes.append(current_route)
                    additional_route_times.update({current_route[0]: round(current_time / 3600)})
    '''
    Hard code additional trucks after our original routes were set out. Weekdays we have 2 trucks left, Saturday is 13
    '''
    additional_trucks = 2
    if day_type == 'Saturday':
        additional_trucks = 13
    count = 0
    mainfreight_cost = 2300
    under_price = 250
    over_price = 325
    route_times_check = {}
    for i in new_route_times:
        if new_route_times[i] > max_hours:
            route_times_check.update({i: 1})
        else:
            route_times_check.update({i: 0})
    for routes in additional_route_times:
        count += 1
        if count <= additional_trucks:
            if additional_route_times[routes] > 4:
                cost += 4 * under_price + (additional_route_times[routes] - 4) * over_price
            else:
                cost += additional_route_times[routes] * under_price
        else:
            cost += round(additional_route_times[routes] / 4) * mainfreight_cost
    if count > additional_trucks:
        mainfreight_count += (count - additional_trucks)
        if (count - additional_trucks) > max_mainfreight:
            max_mainfreight = count - additional_trucks
    for routes in new_route_times:
        cost += ((under_price * (new_route_times[routes] - (new_route_times[routes] - max_hours) *
                                 route_times_check[routes]) + over_price * (new_route_times[routes] - max_hours) *
                  route_times_check[routes]))
    data_points.append(cost)
    if cost > max_cost:
        max_cost = cost
    run += 1

print(f'After {runs} runs, we required {mainfreight_count} Mainfreight trucks, an average of '
      f'{mainfreight_count/runs} per run')
print(f'The maximum number of Mainfreight trucks required during one shift was {max_mainfreight}')
plt.hist(data_points, bins = 50)
plt.show()
average_cost = 0
for costs in data_points:
    average_cost += (costs / runs)
print(f'The average cost was ${round(average_cost)} compared to our original cost of ${value(prob.objective)}')
print(f'The max cost was ${round(max_cost)}')
clt_normal = []
for i in range(1000):
    temp_random = []
    for j in range(int(round(len(data_points) / 2))):
        random_temp = int(random.randint(0, len(data_points) - 1))
        temp_random.append(data_points[random_temp])
    clt_normal.append(np.mean(temp_random))
print("The 95% confidence interval for the mean of our outputs is: ")
print(sms.DescrStatsW(clt_normal).tconfint_mean())



























