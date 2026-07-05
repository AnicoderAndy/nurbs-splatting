"""
Adapted from https://github.com/colormotor/calligraph
TSP optimization uses OR-tools
"""

import numpy as np
import scipy.spatial
import scipy
import pdb

"""
Voronoi weight sampling
"""


def rasterize(V):
    n = len(V)
    X, Y = V[:, 0], V[:, 1]
    ymin = int(np.ceil(Y.min()))
    ymax = int(np.floor(Y.max()))
    P = []
    for y in range(ymin, ymax + 1):
        segments = []
        for i in range(n):
            index1, index2 = (i - 1) % n, i
            y1, y2 = Y[index1], Y[index2]
            x1, x2 = X[index1], X[index2]
            if y1 > y2:
                y1, y2 = y2, y1
                x1, x2 = x2, x1
            elif y1 == y2:
                continue
            if (y1 <= y < y2) or (y == ymax and y1 < y <= y2):
                segments.append((y - y1) * (x2 - x1) / (y2 - y1) + x1)

        segments.sort()
        for i in range(0, (2 * (len(segments) // 2)), 2):
            x1 = int(np.ceil(segments[i]))
            x2 = int(np.floor(segments[i + 1]))
            P.extend([[x, y] for x in range(x1, x2 + 1)])
    if not len(P):
        return V
    return np.array(P)


def rasterize_outline(V):
    n = len(V)
    X, Y = V[:, 0], V[:, 1]
    ymin = int(np.ceil(Y.min()))
    ymax = int(np.floor(Y.max()))
    points = np.zeros((2 + (ymax - ymin) * 2, 3), dtype=int)
    index = 0
    for y in range(ymin, ymax + 1):
        segments = []
        for i in range(n):
            index1, index2 = (i - 1) % n, i
            y1, y2 = Y[index1], Y[index2]
            x1, x2 = X[index1], X[index2]
            if y1 > y2:
                y1, y2 = y2, y1
                x1, x2 = x2, x1
            elif y1 == y2:
                continue
            if (y1 <= y < y2) or (y == ymax and y1 < y <= y2):
                segments.append((y - y1) * (x2 - x1) / (y2 - y1) + x1)
        segments.sort()
        for i in range(0, (2 * (len(segments) // 2)), 2):
            x1 = int(np.ceil(segments[i]))
            x2 = int(np.ceil(segments[i + 1]))
            points[index] = x1, x2, y
            index += 1
    return points[:index]


def weighted_centroid_outline(V, P, Q):
    O = rasterize_outline(V)
    X1, X2, Y = O[:, 0], O[:, 1], O[:, 2]

    Y = np.minimum(Y, P.shape[0] - 1)
    X1 = np.minimum(X1, P.shape[1] - 1)
    X2 = np.minimum(X2, P.shape[1] - 1)

    d = (P[Y, X2] - P[Y, X1]).sum()
    x = ((X2 * P[Y, X2] - Q[Y, X2]) - (X1 * P[Y, X1] - Q[Y, X1])).sum()
    y = (Y * (P[Y, X2] - P[Y, X1])).sum()
    if d:
        return [x / d, y / d]
    return [x, y]


def uniform_centroid(V):
    A = 0
    Cx = 0
    Cy = 0
    for i in range(len(V) - 1):
        s = V[i, 0] * V[i + 1, 1] - V[i + 1, 0] * V[i, 1]
        A += s
        Cx += (V[i, 0] + V[i + 1, 0]) * s
        Cy += (V[i, 1] + V[i + 1, 1]) * s
    Cx /= 3 * A
    Cy /= 3 * A
    return [Cx, Cy]


def weighted_centroid(V, D):
    P = rasterize(V)
    Pi = P.astype(int)
    Pi[:, 0] = np.minimum(Pi[:, 0], D.shape[1] - 1)
    Pi[:, 1] = np.minimum(Pi[:, 1], D.shape[0] - 1)
    D = D[Pi[:, 1], Pi[:, 0]].reshape(len(Pi), 1)
    return ((P * D)).sum(axis=0) / D.sum()


def in_box(points, bbox):
    return np.logical_and(
        np.logical_and(bbox[0] <= points[:, 0], points[:, 0] <= bbox[1]),
        np.logical_and(bbox[2] <= points[:, 1], points[:, 1] <= bbox[3]),
    )


def voronoi(points, bbox):
    i = in_box(points, bbox)

    # Mirror points
    points_center = points[i, :]
    points_left = np.copy(points_center)
    points_left[:, 0] = bbox[0] - (points_left[:, 0] - bbox[0])
    points_right = np.copy(points_center)
    points_right[:, 0] = bbox[1] + (bbox[1] - points_right[:, 0])
    points_down = np.copy(points_center)
    points_down[:, 1] = bbox[2] - (points_down[:, 1] - bbox[2])
    points_up = np.copy(points_center)
    points_up[:, 1] = bbox[3] + (bbox[3] - points_up[:, 1])
    points = np.append(
        points_center,
        np.append(
            np.append(points_left, points_right, axis=0),
            np.append(points_down, points_up, axis=0),
            axis=0,
        ),
        axis=0,
    )
    # Compute Voronoi
    vor = scipy.spatial.Voronoi(points)

    # Filter regions
    epsilon = 0.1
    regions = []
    for region in vor.regions:
        flag = True
        for index in region:
            if index == -1:
                flag = False
                break
            else:
                x = vor.vertices[index, 0]
                y = vor.vertices[index, 1]
                if not (
                    bbox[0] - epsilon <= x <= bbox[1] + epsilon
                    and bbox[2] - epsilon <= y <= bbox[3] + epsilon
                ):
                    flag = False
                    break
        if region != [] and flag:
            regions.append(region)
    vor.filtered_points = points_center
    vor.filtered_regions = regions
    return vor


def voronoi_centroids(points, density, density_P=None, density_Q=None):
    """
    Given a set of point and a density array, return the set of weighted
    centroids.
    """

    X, Y = points[:, 0], points[:, 1]
    # assert  0 < X.min() < X.max() < density.shape[0]
    # assert  0 < Y.min() < Y.max() < density.shape[1]

    xmin, xmax = 0, density.shape[1]
    ymin, ymax = 0, density.shape[0]
    bbox = np.array([xmin, xmax, ymin, ymax])
    vor = voronoi(points, bbox)
    regions = vor.filtered_regions
    centroids = []
    region_vertices = []
    for region in regions:
        vertices = vor.vertices[region + [region[0]], :]
        centroid = weighted_centroid_outline(vertices, density_P, density_Q)

        centroids.append(centroid)
        region_vertices.append(vertices)
    return regions, np.array(centroids), region_vertices


def initialization(n, data, subd=10):
    points = []
    # print('datashape', data.shape)
    while len(points) < n:
        X = np.random.uniform(0, data.shape[1], subd * n)
        Y = np.random.uniform(0, data.shape[0], subd * n)
        P = np.random.uniform(0, 1, subd * n)
        idx = 0
        while idx < len(X) and len(points) < n:
            x, y = X[idx], Y[idx]
            x_, y_ = int(np.floor(x)), int(np.floor(y))
            if P[idx] < data[y_, x_]:
                points.append([x, y])
            idx += 1
    return np.array(points)


def weighted_voronoi_sampling(
    density_map,
    n_points,
    points_per_cell=100,
    nb_iter=10,
    thresh=1.0,
    get_regions=False,
):
    scaling = (n_points * points_per_cell) / (
        density_map.shape[0] * density_map.shape[1]
    )
    scaling = int(round(np.sqrt(scaling)))
    # print(points_per_cell)
    # density_map = scipy.ndimage.zoom(density_map, scaling, order=0)

    density_map = np.minimum(density_map, thresh)
    # density_map = density_map**2
    density_P = density_map.cumsum(axis=1)
    density_Q = density_P.cumsum(axis=1)

    points = initialization(n_points, density_map, points_per_cell)

    for i in range(nb_iter):
        regions, points, region_vertices = voronoi_centroids(
            points, density_map, density_P, density_Q
        )
    if get_regions:
        return points, region_vertices

    return points


def make_coverage_penalty(im, lw=1):
    from . import imaging, geom

    if type(im) != np.ndarray:
        im = np.array(im.convert("L")) / 255
    rect = geom.make_rect(0, 0, *im.shape[::-1])
    rast = imaging.ShapeRasterizer(rect, im.shape[1])

    def cost(a, b):
        rast.clear()
        rast.line(a, b, lw=lw)

        img = rast.get_image() / 255
        return np.sum(img * im)

    return cost


def heuristic_solve(
    points,
    time_limit_minutes=1 / 30,
    penalty_func=None,
    penalty_weight=0.1,
    logging=False,
    verbose=False,
    cycle=True,
    end_to_end=False,
    **kwargs,
):
    """Solve the TSP problem using ortools"""
    # See https://developers.google.com/optimization/routing/tsp
    from ortools.constraint_solver import routing_enums_pb2
    from ortools.constraint_solver import pywrapcp

    distance_matrix = scipy.spatial.distance.cdist(points, points, **kwargs)
    num_points = len(points)

    if penalty_weight > 0 and penalty_func is not None:
        for i in range(num_points):
            for j in range(i + 1, num_points):
                cost = penalty_func(points[i], points[j])
                # if cost > 10:
                #     print('ccc', i, j, cost)
                distance_matrix[i, j] += cost * penalty_weight
                distance_matrix[j, i] += cost * penalty_weight

    distance_matrix = distance_matrix.round().astype(int)
    if end_to_end:
        manager = pywrapcp.RoutingIndexManager(num_points, 1, [0], [len(points) - 1])
    else:
        if not cycle:
            # Force 0 cost to return to the end
            # results in an open ended TSP
            distance_matrix[:, 0] = 0
        manager = pywrapcp.RoutingIndexManager(num_points, 1, 0)

    routing_parameters = pywrapcp.DefaultRoutingModelParameters()
    if logging and verbose:
        routing_parameters.solver_parameters.trace_propagation = True
        routing_parameters.solver_parameters.trace_search = True

    routing = pywrapcp.RoutingModel(manager, routing_parameters)

    def distance_callback(from_idx, to_idx):
        from_node = manager.IndexToNode(from_idx)
        to_node = manager.IndexToNode(to_idx)
        return distance_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    if time_limit_minutes:
        search_parameters.time_limit.seconds = int(round(time_limit_minutes * 60))

    if logging:
        search_parameters.log_search = True

    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        index = routing.Start(0)
        tour = [manager.IndexToNode(index)]

        while not routing.IsEnd(index):
            index = solution.Value(routing.NextVar(index))
            tour.append(manager.IndexToNode(index))

        if not end_to_end and not cycle:
            return tour[:-1]
        return tour
    else:
        return None
