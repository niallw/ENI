import math
import numpy as np
import geompreds # Robust orientation and incircle tests

EPSILON = 0.0001
TWO_PI = 6.28318530717958647692

def EPSILON_COMPARE(val1, val2, epsilon=EPSILON):
    """Check if the given values are close to each other.
    """
    return abs(val1 - val2) <= epsilon

def EPSILON_CHECK(val, epsilon=EPSILON):
    """Check if the given value is close to zero.
    """
    return abs(val) <= epsilon

class Vertex:
    """A 2D vertex class.

    Attributes:
        x: x-coordinate of the vertex.
        y: y-coordinate of the vertex.
        data: The numpy array of the vertex's coordinates.
        incoming_edge: The edge that "points" to this vertex. The edge for which this vertex is the
                       second vertex of the edge.
        outgoing_edge: The edge that "points" away from this vertex. The edge for which this vertex
                       is the first vertex of the edge.
        prev_vert: The vertex preceding this vertex in the CCW list of polygon vertices.
        next_vert: The vertex succeeding this vertex in the CCW list of polygon vertices.
    """

    def __init__(self, px, py):
        """Constructor for the Vertex class.

        Args:
            px: The x-coordinate of the vertex.
            py: The y-coordinate of the vertex.
        """
        self.x = px
        self.y = py
        self.data = np.array([self.x, self.y])
        self.incoming_edge = None
        self.outgoing_edge = None
        self.prev_vert = None
        self.next_vert = None

    def __eq__(self, other):
        if other == None:
            return False
        return self.x == other.x and self.y == other.y

class Edge:
    """An edge class.

    Attributes:
        v1: The first vertex of the edge.
        v2: The second vertex of the edge.
    """

    def __init__(self, _v1, _v2):
        """Constructor for the Edge class.

        Args:
            _v1: The first vertex of the edge.
            _v2: The second vertex of the edge.
        """
        self.v1 = _v1
        self.v2 = _v2

    def __eq__(self, other):
        if other == None:
            return False
        return self.v1 == other.v1 and self.v2 == other.v2

def orient_non_robust(p1, p2, p3):
    """Performs an orientation test to determine which direction a triple of points turns in.
    That is, this function is used to determine if a polyline of 2 segments turns to the left, to
    the right, or lies in a straight line.
    Args:
        p1: The first point.
        p2: The second point.
        p3: The third point.
    Returns:
        0 if the points are colinear.
        -1 if the points are clockwise (right turn).
        1 if the points are counter-clockwise (left turn).
    Return type:
        int
    """
    o = (p2[0] - p1[0])*(p3[1] - p1[1]) - (p2[1] - p1[1])*(p3[0] - p1[0])
    if EPSILON_COMPARE(0, o): return 0 # Epsilon check (close to 0)
    if o < 0: return -1
    if o > 0: return 1

def orient(p1, p2, p3):
    """Performs an orientation test to determine which direction a triple of points turns in.
    That is, this function is used to determine if a polyline of 2 segments turns to the left, to
    the right, or lies in a straight line.

    Args:
        p1: The first point.
        p2: The second point.
        p3: The third point.

    Returns:
        0 if the points are colinear.
        -1 if the points are clockwise (right turn).
        1 if the points are counter-clockwise (left turn).

    Return type:
        int
    """
    val = geompreds.orient2d(p1, p2, p3)
    if val > 0: return 1
    if val < 0: return -1
    return 0

def line_point_distance(s1, s2, p):
    """Computes the minimum distance between a line segment and a point.

    Args:
        s1: The first vertex of the line segment.
        s1: The second vertex of the line segment.
        p: The point.

    Returns:
        The length of the projection of the point onto the line segment.

    Return type:
        float
    """
    assert np.linalg.norm(s2 - s1) != 0, 'Trying to compute length of a single point, not a vector! s1 = [{}, {}] | s2 = [{}, {}]'.format(s1[0], s1[1], s2[0], s2[1])

    t = np.dot(p - s1, s2 - s1) / (np.linalg.norm(s2 - s1) * np.linalg.norm(s2 - s1))
    t = min(max(t, 0), 1)
    closest_point = s1 + ((s2 - s1) * t)

    return np.linalg.norm(closest_point - p)

def line_line_intersection(s1_1, s1_2, s2_1, s2_2):
    """Compute the point at which two line segments intersect.
    This function does not work properly if the lines are colinear.

    Args:
        s1_1: The first vertex of the first line segment.
        s1_2: The second vertex of the first line segment.
        s2_1: The first vertex of the second line segment.
        s2_2: The second vertex of the second line segment.

    Returns:
        The point of intersection of the two line segments.
        If the two lines do not intersect, it returns the point (\infty, \infty).

    Return type:
        2D point (numpy array)
    """
    a1 = s1_2[1] - s1_1[1]
    b1 = s1_1[0] - s1_2[0]
    c1 = (a1 * s1_1[0]) + (b1 * s1_1[1])

    a2 = s2_2[1] - s2_1[1]
    b2 = s2_1[0] - s2_2[0]
    c2 = (a2 * s2_1[0]) + (b2 * s2_1[1])

    det = a1 * b2 - a2 * b1

    # Lines are parallel
    if det == 0:
        return np.array([float('inf'), float('inf')])
    else:
        return np.array([(b2 * c1 - b1 * c2) / det, (a1 * c2 - a2 * c1) / det])

def ray_line_intersect(ray_origin, ray_dir, s1, s2):
    """Compute the intersection between a ray and a line segment.
    Via https://stackoverflow.com/questions/53893292/how-to-calculate-ray-line-segment-intersection-preferably-in-opencv-and-get-its.

    Args:
        ray_origin: The point from which the ray originates.
        ray_dir: The direction in which the ray travels (radians).
        s1: The first point of the line segment.
        s2: The second point of the line segment.

    Returns:
        The length along the ray at which point the ray intersects with the line segment.
        If the ray and line segment do not intersect, it returns -1.0.

    Return type:
        float
    """
    p1 = ray_origin - s1
    p2 = s2 - s1
    p3 = np.array([-math.sin(ray_dir), math.cos(ray_dir)])

    d = np.dot(p2, p3)
    if EPSILON_CHECK(d):
        return -1.0

    t1 = np.cross(p2, p1) / d
    t2 = np.dot(p1, p3) / d

    if t1 >= 0.0 and (t2 >= 0.0 and t2 <= 1.0):
        return t1

    return -1.0

def point_inside_polygon(p, polygon_verts):
    """Check if the point p is inside the polygon. Returns true if it is, false otherwise.
    Via https://www.ics.uci.edu/~eppstein/161/960307.html.

    Args:
        p: The point which we are checking if it is inside the polygon.
        polygon_verts: A list of the polygon's vertices, in CCW order.

    Returns:
        1 if p is inside the polygon.
        -1 if p is outside the polygon.
        0 if is on the polygon boundary.

    Return type:
        int
    """
    crossings = 0

    for i in range(len(polygon_verts)):
        cur = polygon_verts[i]
        prev = polygon_verts[-1] if (i == 0) else polygon_verts[i-1]
        next_pt = polygon_verts[(i + 1) % len(polygon_verts)]

        if orient(cur, p, next_pt) == 0 or orient_non_robust(cur, p, next_pt) == 0: return 0 # On boundary

        if ((cur[0] < p[0] and p[0] < next_pt[0]) or (cur[0] > p[0] and p[0] > next_pt[0])):
            t = (p[0] - next_pt[0]) / (cur[0] - next_pt[0])
            cy = t * cur[1] + (1 - t) * next_pt[1]
            if (p[1] == cy): return 0 # On boundary
            elif (p[1] > cy): crossings += 1

        if (cur[0] == p[0] and cur[1] <= p[1]):
            if (cur[1] == p[1]): return 0 # On boundary
            if (next_pt[0] == p[0]):
                if (cur[1] <= p[1] and p[1] <= next_pt[1] or
                    cur[1] >= p[1] and p[1] >= next_pt[1]):
                    return 0 # On boundary

            elif (next_pt[0] > p[0]): crossings += 1
            if (prev[0] > p[0]): crossings += 1

    if (crossings % 2 == 1): return 1 # Inside
    else: return -1 # Outside

def ray_point_intersection(ray_origin, ray_dir, p):
    """Determine if a given point lies along a ray.

    Args:
        ray_origin: The point from which the ray emanates.
        ray_dir: The direction in which the ray shoots.
        p: The point whose presence on the ray we are checking.

    Returns:
        True if the point p is on the ray.
        False if the point p is not on the ray.

    Return type:
        bool
    """
    # Force ray_dir to be positive so that we can compare it to angle later
    ray_dir = math.fmod(ray_dir + math.pi + math.pi, math.pi + math.pi)
    p = p - ray_origin
    angle = math.atan2(p[1], p[0])
    if angle < 0:
        angle += math.pi + math.pi

    return EPSILON_COMPARE(ray_dir, angle)

def project_point(v, p):
    """Project a point onto a vector. It is required that the vector and point are
    centered at the origin (i.e. the vector originates from the point (0, 0)).

    Args:
        v: The vector to project onto.
        p: The point to be projected onto v.

    Returns:
        The point p after it has been projected onto v.

    Return type:
        numpy array
    """
    return (np.dot(p, v) / np.dot(v, v)) * v

def angle(p1, p2, p3):
    """Compute the angle at p2 formed by the triplet (p1, p2, p3). 
    This is just an application of the law of cosines.

    Args:
        p1: The first point of the triplet of points.
        p2: The second point of the triplet of points. The angle we are measuring
            is formed at this point by connecting p2 to p1 and p3.
        p3: The third point of the triplet of points.
    
    Returns:
        The angle formed at point p2 by the line segments p1p2 and p2p3. Angle is
        in radians and is constrained to be in the range [0, 2pi).

    Return type:
        float
    """
    a = np.linalg.norm(p2 - p3)
    b = np.linalg.norm(p2 - p1)
    c = np.linalg.norm(p1 - p3)
    return math.acos((c**2 - a**2 - b**2) / (-2 * a * b))

def angle2(p):
    """Get the angle formed by a vector p, in the range [0, 2pi].

    Args:
        p: The point defining the vector whose angle we wish to compute.

    Returns:
        The angle formed by the vector p, in the range [0, 2pi].

    Return type:
        float
    """
    a = math.atan2(p[1], p[0])
    if a < 0:
        a += TWO_PI
    return a

def sort_by_angle(pts):
    """Sort a list of points by polar angle.

    Args:
        pts: The points to be sorted.

    Returns:
        A copy of the list of points, sorted by polar angle.

    Return type:
        List of points.
    """
    sorted_pts = sorted(pts, key=angle2)
    return sorted_pts

def get_first_polygon_intersection(polygon_pts, ray_origin, ray_dir):
    """Get the first intersection with a polygon boundary starting from a 
    given position and in a specified direction.

    Args:
        polygon_pts: The vertices denoting the boundary of the polygon.
        ray_origin: The position from which we want to check for an intersection.
        ray_dir: The direction in which to seek the first intersection with the polygon,
                 emanating from ray_origin.

    Returns:
        A point on the polygon boundary denoting the first intersection in the direction
        ray_dir emanating ray_origin, or None if there is no intersection.

    Return type:
        2D numpy array
    """
    its_dist = float('inf')
    its_pt = None

    for i in range(len(polygon_pts)):
        p1 = polygon_pts[i]
        p2 = polygon_pts[(i+1) % len(polygon_pts)]

        # Line segment is colinear with the ray, get the closer of the two line 
        # segment points.
        if ray_point_intersection(ray_origin, ray_dir, p1) and \
            ray_point_intersection(ray_origin, ray_dir, p2):
            p1_dist = np.linalg.norm(ray_origin - p1)
            p2_dist = np.linalg.norm(ray_origin - p2)

            if p1_dist < p2_dist:
                its_dist = np.linalg.norm(ray_origin - p1)
                its_pt = p1
            else:
                its_dist = np.linalg.norm(ray_origin - p2)
                its_pt = p2
        # Ray intersects with the first point of the line segment.
        elif ray_point_intersection(ray_origin, ray_dir, p1) and \
            np.linalg.norm(ray_origin - p1) < its_dist:
            its_dist = np.linalg.norm(ray_origin - p1)
            its_pt = p1
        # Ray intersects with the second point of the line segment.
        elif ray_point_intersection(ray_origin, ray_dir, p2) and \
            np.linalg.norm(ray_origin - p2) < its_dist:
            its_dist = np.linalg.norm(ray_origin-  p2)
            its_pt = p2
        # Ray intersects with some point of the line segment other than the endpoints.
        else:
            t = ray_line_intersect(ray_origin, ray_dir, p1, p2)
            if t != -1 and t < its_dist:
                its_dist = t
                its_pt = ray_origin + (np.array([math.cos(ray_dir), math.sin(ray_dir)]) * t)

    return its_pt

def line_segments_intersect(s1_1, s1_2, s2_1, s2_2):
    """Determine if two line segments intersect.

    Args:
        s1_1: The first endpoint of the first line segment.
        s1_2: The second endpoint of the first line segment.
        s2_1: The first endpoint of the second line segment.
        s2_1: The second endpoint of the second line segment.

    Returns:
        True if the line segments intersect, False otherwise.

    Return type:
        boolean
    """
    return (orient(s1_1, s1_2, s2_1) != orient(s1_1, s1_2, s2_2)) and (orient(s2_1, s2_2, s1_1) != orient(s2_1, s2_2, s1_2))

def polyline_edge_intersection(polyline, edge):
    """Compute the intersection points between an edge and a polyline.

    Args:
        polyline: A chain of connected line segments.
        edge: The edge whose intersection(s) with the polyline we wish to compute.

    Returns:
        A list of intersection points between the polyline and edge.

    Return type:
        A list of numpy arrays.
    """
    its_pts = []
    e_p1 = edge[0]
    e_p2 = edge[1]
    for i in range(len(polyline) - 1):
        polyline_p1 = polyline[i]
        polyline_p2 = polyline[i+1]

        if line_segments_intersect(e_p1, e_p2, polyline_p1, polyline_p2):
            its_pts.append(line_line_intersection(e_p1, e_p2, polyline_p1, polyline_p2))
    return its_pts
