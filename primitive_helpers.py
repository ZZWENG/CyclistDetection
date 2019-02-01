import numpy as np
import itertools

def bike_human_nums(object_names):
    '''Returns: (int) categorical index, where
            0 --> more people than bikes
            1 --> more bikes than people
            2 --> same number of bikes and people
    '''
    names = object_names
    num_person = 0
    num_bicycles = 0
    for i in range(np.shape(names)[0]):
        name = names[i]
        if ('person' in name) or ('man' in name) or ('woman' in name) or ('girl' in name) or ('boy' in name) or ('people' in name):
            num_person = num_person+1
        if ('cycle' in name) or ('bike' in name) or ('bicycle' in name):
            num_bicycles = num_bicycles+1
    
    if (num_bicycles == 0) or (num_person == 0):
        return 0
    if num_person == num_bicycles:
        return 2
    elif num_person <= num_bicycles:
        return 0
    else:
        return 1


def bike_human_position(object_names, object_x, object_y):
    person_position = np.array([[0,0],[0,0]])
    bicycle_position = np.array([[0,0],[0,0]])
    for i in range(np.shape(object_names)[0]):
        name = object_names[i]
        if ('person' in name) or ('man' in name) or ('woman' in name) or ('girl' in name) or ('boy' in name) or ('people' in name):
            person_position = np.concatenate((person_position, np.array([[object_x[i],object_y[i]]])))
        if ('cycle' in name) or ('bike' in name) or ('bicycle' in name):
            bicycle_position = np.concatenate((bicycle_position, np.array([[object_x[i],object_y[i]]])))
    
    person_position = person_position[2:,:]
    bicycle_position = bicycle_position[2:,:]
    
    if (np.shape(bicycle_position)[0] == 0) or (np.shape(person_position)[0] == 0):
        return -1
    
    # generate all combinations of people/bikes coordinates
    if len(bicycle_position) >= len(person_position):
        list1 = [list(coord) for coord in bicycle_position]
        list2 = [list(coord) for coord in person_position]
    else:
        list2 = [list(coord) for coord in bicycle_position]
        list1 = [list(coord) for coord in person_position]
        
    coord_comb = [list1, list2]
    person_bike_pairs = itertools.product(*coord_comb)
    positions = []
    # get pairwise positions between people and bikes
    for pair in person_bike_pairs:
        coord_person = pair[0]
        coord_bike = pair[1]
        if (coord_bike[1] - coord_person[1] >= 5  and abs(coord_person[0] - coord_bike[0]) < 50) :
            # if a person is on top of a bike
            return 1
    return -1

def bike_human_distance(names, object_x, object_y):
    '''Returns: (np.float) distance between closest person/bike '''

    # get coordinates (positions) of bikes/people
    person_position = np.array([[0,0],[0,0]])
    bicycle_position = np.array([[0,0],[0,0]])
    for i in range(np.shape(names)[0]):
        name = names[i]
        if ('person' in name) or ('man' in name) or ('woman' in name) or ('girl' in name) or ('boy' in name) or ('people' in name):
            person_position = np.concatenate((person_position, np.array([[object_x[i],object_y[i]]])))
        if ('cycle' in name) or ('bike' in name) or ('bicycle' in name):
            bicycle_position = np.concatenate((bicycle_position, np.array([[object_x[i],object_y[i]]])))
    
    person_position = person_position[2:,:]
    bicycle_position = bicycle_position[2:,:]
    
    if (np.shape(bicycle_position)[0] == 0) or (np.shape(person_position)[0] == 0):
        return -1
    
    # generate all combinations of people/bikes coordinates
    if len(bicycle_position) >= len(person_position):
        list1 = [list(coord) for coord in bicycle_position]
        list2 = [list(coord) for coord in person_position]
    else:
        list2 = [list(coord) for coord in bicycle_position]
        list1 = [list(coord) for coord in person_position]
        
    coord_comb = [list1, list2]
    person_bike_pairs = itertools.product(*coord_comb)
    
    # compute pairwise distances between people and bikes
    dists = []
    for pair in person_bike_pairs:
        coord1 = np.array(pair[0])
        coord2 = np.array(pair[1])
        dists.append(np.linalg.norm(coord1-coord2))     
    
    return np.min(dists)


def bike_human_size(object_names, object_area):
    '''Returns: (int) pixelwise area difference between humans/bikes '''

    # names = object_names.split(' , ')[1:]
    names = object_names
    person_area = np.array([0])
    bicycle_area = np.array([0])
    
    # get 'area' for people and bikes
    for i in range(np.shape(names)[0]):
        name = names[i]
        if ('person' in name) or ('man' in name) or ('woman' in name) or ('girl' in name) or ('boy' in name) or ('people' in name):
            person_area = np.concatenate((person_area, [object_area[i]]))
        if ('cycle' in name) or ('bike' in name) or ('bicycle' in name):
            bicycle_area = np.concatenate((bicycle_area, [object_area[i]]))
    
    person_area = person_area[1:]
    bicycle_area = bicycle_area[1:]
    

    if (np.shape(bicycle_area)[0] == 0) or (np.shape(person_area)[0] == 0):
        area_diff = -1
    
    area_diff = -1
    for i in range(np.shape(bicycle_area)[0]):
        try:
            area_diff_temp = np.max((np.abs(bicycle_area[i]-person_area[:])))
            area_diff = np.max(area_diff_temp, area_diff)
        except:
            continue
    
    return area_diff

