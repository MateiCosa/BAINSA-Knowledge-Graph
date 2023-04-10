import numpy as np
import time
import heapq
import linecache

def dijkstra(WG, src_node, dest_node, const_weights = True, return_path = True, max_steps = None, debug = False):
    """
    Parameters
    ----------
    WG : WikiGraph
        Instance of WikiGraph used for Dijkstra's algorithm.
    src_node : string
        Source node of the edge: either id or title;
    dest_node : string
        Destination node of the edge.
    return_path : bool
        True if path should be returned; False otherwise.
    debug : bool
        True if debug mode is activated; False otherwise.
    Takes two string: id or title of source and destination nodes: either id or title;
    Returns
    -------
    cost : float
        A float representing the cost of the path from src_node to dest_node
        It is either a finite positive integer or -1 if there is no path between the 2 nodes.
        A list of strings representing the path is also returned if return_path is True.
    """
    if not isinstance(debug, bool):
        raise Exception("Debug must be of type bool!")
    if debug:
        t0 = time.time()
    # if not isinstance(WG, WikiGraph):
    #     raise Exception("WG must be of type WikiGraph!")
    if not isinstance(src_node, str) or not isinstance(dest_node, str):
        raise Exception("The two nodes must be both strings!")
    if not WG.__contains__(src_node) or not WG.__contains__(dest_node):
        raise Exception("The two nodes must belong to the graph!")
        
    if src_node == dest_node:
        if debug:
            t1 = time.time()
            print(f"Sucesfully completed task in {t1-t0} seconds.")
            print("Vistited 0 nodes.")
        if return_path:
            return (0., [src_node])
        else:
            return 0.
    
    if WG.get_outward_weight(src_node, dest_node) < np.inf:
        if debug:
            t1 = time.time()
            print(f"Sucesfully completed task in {t1-t0} seconds.")
            print("Vistited 1 node.")
        if return_path:
            return (0., [src_node, dest_node])
        else:
            return 1.    
    
    if src_node in WG.article_id:
        src_node = WG.article_id[src_node]
    if dest_node in WG.article_id:
        dest_node = WG.article_id[dest_node]
    
    #distances = {vertex: np.inf for vertex in WG.title}
    distances = dict()
    distances[src_node] = 0
    
    pred = {src_node: None}
    num_visited_nodes = 0
    
    pq = [(0, src_node)]
    while len(pq) > 0:
    
        current_distance, current_vertex = heapq.heappop(pq)
        num_visited_nodes += 1
        
        if max_steps is not None and num_visited_nodes > max_steps:
            return 4.0
        
        if current_vertex not in distances:
            distances[current_vertex] = np.inf
        
        if current_distance > distances[current_vertex]:
            continue
        
        #neigh = WG.__getitem__(current_vertex)
        neigh = WG.get_outward_neighbours(current_vertex)
        if not const_weights:
            neigh_set = set(neigh)
        
        for neighbour in neigh:
            
            if neighbour == '':
                continue
            
            if const_weights:
                weight = 1.0
            else:
                weight =  WG.get_outward_weight(current_vertex, neighbour, neigh_set, check_input = False, input_ids = True)
            distance = current_distance + weight
            
            if neighbour not in distances:
                distances[neighbour] = np.inf
                
            if distance < distances[neighbour]:
                distances[neighbour] = distance
                heapq.heappush(pq, (distance,neighbour))
                pred[neighbour] = current_vertex

            if dest_node == neighbour:
                
                if debug:
                    t1 = time.time()
                    print(f"Sucesfully completed task in {t1-t0} seconds.")
                    print(f"Vistited {num_visited_nodes} nodes.")
                
                path = [WG.get_title(dest_node)]
                node = dest_node
                while pred[node] is not None:
                    node = pred[node]
                    path.append(WG.get_title(node))
                path = path[::-1]
                
                if return_path:
                    return (distance, path)
                else:
                    return distance
    
    if debug:
        t2 = time.time()
        print(f"No path found. Task completed in {t2-t0} seconds.")
        print("Vistited {num_visited_nodes} nodes.")
    if return_path:
        return (-1, [])
    else:
        return -1        
        
def bi_dijkstra(WG, src_node, dest_node, const_weights = True, debug = True):
    """
    Parameters
    ----------
    WG : WikiGraph
        Instance of WikiGraph used for Dijkstra's algorithm.
    src_node : string
        Source node of the edge: either id or title;
    dest_node : string
        Destination node of the edge.
    debug : bool
        True if debug mode is activated; False otherwise.
    Takes two string: id or title of source and destination nodes: either id or title;
    Returns
    -------
    weight : float
        A float representing the weight of the path from src_node to dest_node
        It is either a finite positive integer or -1 if there is no path between the 2 nodes.
    """
    t0 = time.time()
    
    num_visited_nodes = 0
    
    if not WG.__contains__(src_node) or not WG.__contains__(dest_node):
        raise Exception("Either source {src_node} or target {dest_node} is not in WG")
    
    if src_node in WG.article_id:
        src_node = WG.article_id[src_node]
    if dest_node in WG.article_id:
        dest_node = WG.article_id[dest_node]

    if src_node == dest_node:
        if debug:
            t1 = time.time()
            print(f"Visited {num_visited_nodes}. Path found in {t1 - t0} seconds.")
        return (0, [src_node])
    
    if WG.get_outward_weight(src_node, dest_node) < np.inf:
        if debug:
            t1 = time.time()
            num_visited_nodes += 1
            print(f"Visited {num_visited_nodes}. Path found in {t1 - t0} seconds.")
        return (1, [src_node, dest_node])

    push = heapq.heappush
    pop = heapq.heappop
    # Init:  [Forward, Backward]
    dists = [{}, {}]  # dictionary of final distances
    paths = [{src_node: [src_node]}, {dest_node: [dest_node]}]  # dictionary of paths
    fringe = [[], []]  # heap of (distance, node) for choosing node to expand
    seen = [{src_node: 0}, {dest_node: 0}]  # dict of distances to seen nodes
    # initialize fringe heap
    push(fringe[0], (0, src_node))
    push(fringe[1], (0, dest_node))
    # variables to hold shortest discovered path
    finaldist = np.inf
    finalpath = []
    direction = 1
    while fringe[0] and fringe[1]:
        #print(fringe)
        # choose direction
        # direction == 0 is forward direction and direrction == 1 is back
        direction = 1 - direction
        # extract closest to expand
        (dist, v) = pop(fringe[direction])
        
        num_visited_nodes += 1 #keep track of visited nodes
        
        if v in dists[direction]:
            # Shortest path to v has already been found
            continue
        # update distance
        dists[direction][v] = dist  # equal to seen[dir][v]
        if v in dists[1 - direction]:
            # if we have scanned v in both directions we are done
            # we have now discovered the shortest path
            return (finaldist, finalpath)

        if direction == 0:
            neighs = WG.get_outward_neighbours(v)
        else:
            neighs = WG.get_inward_neighbours(v)
        
        if not const_weights:
            neighs_set = set(neighs)
        
        for w in neighs:
            if w == '':
                continue
            # weight(v, w) for forward and weight(w, v) for back direction
            if const_weights:
                weight = 1.0
            else:
                if direction == 0:
                    weight = WG.get_outward_weight(v, w, src_neighs = neighs_set, check_input = False, input_ids = True) 
                else:
                    weight = WG.get_inward_weight(w, v, src_neighs = neighs_set, check_input = False, input_ids = True)
            
            if weight == np.inf:
                print(v, w, w in neighs_set, direction)
                return -1
            
            vwLength = dists[direction][v] + weight
            if w in dists[direction]:
                if vwLength < dists[direction][w]:
                    raise ValueError("Contradictory paths found: negative weights?")
            elif w not in seen[direction] or vwLength < seen[direction][w]:
                # relaxing
                seen[direction][w] = vwLength
                push(fringe[direction], (vwLength, w))
                paths[direction][w] = paths[direction][v] + [w]
                if w in seen[0] and w in seen[1]:
                    # see if this path is better than the already
                    # discovered shortest path
                    totaldist = seen[0][w] + seen[1][w]
                    if finalpath == [] or finaldist > totaldist:
                        finaldist = totaldist
                        revpath = paths[1][w][:]
                        revpath.reverse()
                        finalpath = paths[0][w] + revpath[1:]
        #if num_visited_nodes > 50:
            #print(len(seen[0]), len(seen[1]))
            #return -1
    raise Exception(f"No path between {src_node} and {dest_node}.")