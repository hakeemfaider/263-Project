import pandas as pd
import numpy as np
from random import *

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
def search_nodes(starting_node, map_data, max_time):
    possible_nodes = []
    index = map_data.columns.get_loc(starting_node) - 1
    for i in range(len(map_data)):
        if map_data.iloc[index].iloc[i + 1] <= max_time and map_data.iloc[index].iloc[i + 1] != 0:
            if map_data.iloc[i].iloc[0] != "Distribution Centre Auckland" and map_data.iloc[i].iloc[0] != starting_node:
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
                       count, min_nodes, p_data):
    global total_possible_paths
    global current_route
    # If our current node can't see any viable nodes, return to the previous node with updated values
    if len(nodes) == 0:
        if current_route != ['Distribution Centre Auckland']:
            total_duplicate_counter = 0
            for i in range(len(total_possible_paths)):
                duplicate_counter = 0
                for j in range(len(current_route)):
                    if current_route[j] in total_possible_paths[i]:
                        duplicate_counter += 1
                if duplicate_counter == len(current_route):
                    total_duplicate_counter += 1
            if total_duplicate_counter == 0:
                total_possible_paths.append(current_route.copy())
    else:
        check_node_counter = 0
        for j in range(len(nodes)):
            # If the node sees a viable not it hasn't come from
            if nodes[j] not in current_route:
                temp_pallets = pallets + p_data['Weekdays'][nodes[j]]
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
                        temp_possible_nodes = search_nodes(nodes[j], map_data, max_time)
                        # Continue the search with the new node
                        possible_node_loop(temp_possible_nodes, temp_pallets, max_pallets, max_time, map_data,
                                           count, min_nodes, p_data)
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
                total_duplicate_counter = 0
                for i in range(len(total_possible_paths)):
                    duplicate_counter = 0
                    for j in range(len(current_route)):
                        if current_route[j] in total_possible_paths[i]:
                            duplicate_counter += 1
                    if duplicate_counter == len(current_route):
                        total_duplicate_counter += 1
                if total_duplicate_counter == 0:
                    total_possible_paths.append(current_route.copy())


map_data = pd.read_csv('WoolworthsDurations.csv')
data = pd.read_csv('WoolworthsDemand2024.csv')
store_data = pd.read_csv('WoolworthsLocations.csv')
all_stores = []
for i in range(len(data)):
    all_stores.append(data.iloc[i].iloc[0])
all_stores.append('Distribution Centre Auckland')

test_data1 = pd.DataFrame([5] * 64 + [0], index=all_stores, columns=["Weekdays"])
test_data2 = pd.DataFrame([3] * 64 + [0], index=all_stores,columns=["Saturday"])
test_data = pd.concat([test_data1, test_data2], axis=1)

weekdays, weekend = create_data()
final_data = pd.DataFrame([weekdays, weekend], index=["Weekdays", "Saturday"], columns=all_stores)
final_data = final_data.transpose()

min_nodes = 2
max_time = 619
used_nodes = []
max_pallets = 20
total_possible_paths = []

for stores in all_stores:
    total_node_counter = 1
    current_route = ["Distribution Centre Auckland", stores]
    current_pallets = final_data['Weekdays'][stores]
    starting_node = stores
    possible_nodes = search_nodes(starting_node, map_data, max_time)
    possible_node_loop(possible_nodes, current_pallets, max_pallets, max_time, map_data,
                                        total_node_counter, min_nodes, final_data)


if min_nodes == 1:
    for store in all_stores[0:-1]:
        total_possible_paths.append(['Distribution Centre Auckland', store])

for i in range(len(total_possible_paths)):
    total_possible_paths[i].append('Distribution Centre Auckland')

delete_short(min_nodes)
for i in range(len(total_possible_paths)):
    print(total_possible_paths[i])

see_all_companies = []
for i in range(len(total_possible_paths)):
    for j in range(len(total_possible_paths[i])):
        if total_possible_paths[i][j] not in see_all_companies:
            see_all_companies.append(total_possible_paths[i][j])

print("Our routes can cover " + str(len(see_all_companies)) + " nodes")
if len(see_all_companies) != len(all_stores):
    print(set(all_stores) - set(see_all_companies))
    print("were/was missed out")
else:
    print("All nodes can be visited in some way from these routes. Additional routes may be needed if routes overlap.")