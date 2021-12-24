import numpy as np
import math
import geometry
from scipy.optimize import minimize_scalar
from scipy import integrate
from scipy.optimize import brute, basinhopping, differential_evolution
from scipy.optimize import rosen
import sys
from shapely.geometry import Polygon

class MinimizeStopper(object):
    def __init__(self, f, poly1, poly2, tau=0.1):
        self.fun = f                     # set the objective function
        self.best_x = None
        self.best_func = np.inf
        self.threshold = tau                   # set the user-desired threshold
        self.poly1 = poly1
        self.poly2 = poly2

    def __call__(self, xk, convergence=None, *args, **kwds):
        # print('convergence: {}'.format(convergence))
        rotated_poly2 = self.poly2.rotate(xk)
        loss = self.poly1.compare_symmetric_diff(rotated_poly2)
        return geometry.EPSILON_CHECK(loss, epsilon=self.threshold)

class StarPolygon:
    """A polygon class.

    The polygon must be defined by a kernel and a list of vertices in CCW order.
    In this particular program, we only use this class to define star polygons (visibility
    polygons). A star polygon is a special type of polygon that has at least one point that
    can be connected to every other point in the polygon by a straight line without intersecting
    with any edge of the polygon. That is, every point of the polygon can be "seen" by an
    observer who stands at this special point.

    Attributes:
        kernel: The point from which every other point of the polygon can be seen.
        pts: A list of 2D points (numpy vectors) denoting the vertices of the polygon.
        verts: A list of Vertex objects denoting the vertices of the polygon.
        edges: A list of Edge objects denoting the edges of the polygon.
        area: The area of the polygon.
        perimeter: The perimeter of the polygon.
        min_diameter: The shortest line that connects two edges of the polygon and passes
                      through the kernel.
        avg_radius: The average value of the distance from the kernel to the polygon boundary.
    """

    def __init__(self, pts, center=None, compute_area=False):
        """Constructor for the Polygon class.

        Args:
            pts: The vertices that define the boundary of the polygon, in CCW order.
            center: The kernel of the polygon. This is only used if the polygon we are creating
                    is a visibility polygon. If we just want a regular polygon (not a visibility
                    polygon), simply do not pass in any second argument into your constructor call.
        """
        self.kernel = center
        self.pts = self.add_theta_0_point(pts)
        self.verts = []
        self.edges = []
        self.compute_connections()
        if not compute_area:
            self.area = 0
        else:
            self.area = self.compute_area()
        self.perimeter = 0
        # self.perimeter = self.compute_perimeter()
        if not center is None:
            self.min_diameter = 0
            self.min_radius = 0

            self.compute_parameterizations()

    def add_theta_0_point(self, pts):
        """Find the intersection point of the ray theta=0 from the kernel to the boundary, and
        add it to the polygon's list of vertices. This is done so that we have a standard starting
        point for all polygons for when we compute integrals. If the intersection point is already
        part of the input vertices, no additional point is added to the vertex list.

        Args:
            pts: The vertices of the polygon that were initially passed in.

        Returns:
            The list of vertices, with the intersection point for the ray theta=0 added to 
            the list.

        Return type:
            List of numpy points.
        """
        if self.kernel is None:
            return pts
        
        for i in range(len(pts)):
            p1 = pts[i]
            p2 = pts[(i+1) % len(pts)]
            if geometry.ray_point_intersection(self.kernel, 0, p1) or \
               geometry.ray_point_intersection(self.kernel, 0, p2):
                break
            
            t = geometry.ray_line_intersect(self.kernel, 0, p1, p2)
            if t != -1.0:
                its_pt = self.kernel + (np.array([math.cos(0), math.sin(0)]) * t)
                pts.insert((i+1) % len(pts), its_pt)

        pts = self.shift_vertices(pts)
        return np.array(self.remove_duplicates(pts))

    def shift_vertices(self, pts):
        for i in range(len(pts)):
            if geometry.EPSILON_COMPARE(math.atan2(pts[i][1] - self.kernel[1], pts[i][0] - self.kernel[0]), 0):
                return pts[i:] + pts[:i]
        self.error_log('Failed to find a point at theta=0 to mark the start of the list of polygon vertices!')
        assert False, 'Failed to find a point at theta=0 to mark the start of the list of polygon vertices!'

    def compute_connections(self):
        """Computes the pointers between vertices and edges. That is, this function
        creates vertex and edge objects, and initializes them to have the correct pointers
        to the next and previous vertices and edges. This is done so that we are able to
        "walk" along the boundary of the polygon by simply updating pointers.

        Returns:
            None
        """
        for i in range(len(self.pts)):
            p1 = self.pts[i]
            p2 = self.pts[(i+1) % len(self.pts)]

            v1 = geometry.Vertex(p1[0], p1[1])
            # On the last vertex (next vertex was already created in the first iteration)
            if np.array_equal(p2, self.pts[0]):
                v2 = self.verts[0]
            else:
                v2 = geometry.Vertex(p2[0], p2[1])
            self.verts.append(v1)

        for i in range(len(self.verts)):
            v1 = self.verts[i]
            v2 = self.verts[(i+1) % len(self.verts)]
            v3 = self.verts[(i+2) % len(self.verts)]

            v1.next_vert = v2
            v2.prev_vert = v1

        for i in range(len(self.verts)):
            v1 = self.verts[i]
            v2 = self.verts[(i+1) % len(self.verts)]
            self.edges.append(geometry.Edge(v1, v2))

        for i in range(len(self.verts)):
            v1 = self.verts[i]
            v2 = self.verts[(i+1) % len(self.verts)]

            v1.outgoing_edge = self.edges[i]
            v2.incoming_edge = self.edges[i]

    def compute_area(self):
        """Computes the area of the polygon using the shoelace formula.
        https://en.wikipedia.org/wiki/Shoelace_formula

        Returns:
            The area of the polygon.

        Return type:
            float
        """
        area_sum = 0
        for i in range(len(self.pts)):
            p1 = self.pts[i]
            p2 = self.pts[(i+1) % len(self.pts)]
            area_sum += p1[0] * p2[1]
            area_sum -= p2[0] * p1[1]
        return abs(area_sum) * 0.5

    def compute_perimeter(self):
        """Computes the perimeter of the polygon.

        Returns:
            The perimeter of the polygon.

        Return type:
            float
        """
        perim = 0.0

        for i in range(len(self.pts)):
            p1 = self.pts[i]
            p2 = self.pts[(i + 1) % len(self.pts)]
            perim += np.linalg.norm(p1 - p2)

        return perim

    def get_polyline_list(self, start, end):
        """Get the list of vertices of the polygon that are between two points on the polygon's
        boundary, in CCW order. This function starts at one point on the polygon boundary and
        "walks" along the boundary until it reaches the specified end point also on the boundary.
        The polyline is the list of all the vertices encountered along this walk, including the
        start and end points.

        Args:
            start: The point on the polygon boundary from which we start the walk.
            end: The point on the polygon boundary from which we end the walk.

        Returns:
            A list of the polygon vertices that lie between the start and end points of a walk
            along the polygon boundaries. The vertices are in CCW order.

        Return type:
            list of vertices
        """
        closest_edge_to_start = self.closest_edge_to_point(start)
        closest_edge_to_end = self.closest_edge_to_point(end)
        if closest_edge_to_start == closest_edge_to_end:
            return [start, end]

        polyline = [start]
        found_end_of_polyline = False
        cur_pt = closest_edge_to_start.v2
        while not found_end_of_polyline:
            polyline.append(cur_pt.data)
            if cur_pt.outgoing_edge == closest_edge_to_end:
                found_end_of_polyline = True
            cur_pt = cur_pt.next_vert
        if not np.allclose(polyline[-1], end):
            polyline.append(end)

        return polyline

    def closest_edge_to_point(self, pt):
        """Finds the edge of the polygon that is closest to a given point.

        Args:
            pt: The point from which we want to find the closest polygon edge.

        Returns:
            The Edge object that is closest to pt.

        Return type:
            Edge
        """
        closest_dist = float('inf')
        closest_edge = None

        for edge in self.edges:
            e_p1 = edge.v1.data
            e_p2 = edge.v2.data
            d = geometry.line_point_distance(e_p1, e_p2, pt)
            if d < closest_dist:
                closest_edge = edge
                closest_dist = d

        return closest_edge

    def get_opposite_edge(self, vertex):
        """Get the edge of the polygon opposite to a vertex. Here, "opposite edge" is defined
        as the first edge of the poylgon that is intersected by the ray originating from a
        specified point and passing through the polygon's kernel. This method also returns the
        intersection point on the opposite edge.

        Args:
            vertex: A vertex of the polygon.

        Returns:
            The edge opposite to the specified vertex, as well as the intersection point on
            the opposite edge.

        Return type:
            (edge, point) tuple
        """
        centered_vert = vertex.data - self.kernel
        theta_to_vertex = math.atan2(centered_vert[1], centered_vert[0])
        theta = theta_to_vertex + math.pi

        return self.get_intersecting_edge(theta)

    def get_intersecting_edge(self, theta):
        """Get the edge that intersects the ray originating from the kernel and going in
        the direction of theta. This method also returns the point intersection along the
        intersected edge.

        Args:
            theta: The direction of the ray emanating from the kernel (radians).

        Returns:
            The edge intersected by the ray emanating from the kernel, and the point of
            intersection along this edge.

        Return type:
            (edge, point) tuple
        """
        for i in range(len(self.pts)):
            p1 = self.pts[i]
            p2 = self.pts[(i+1) % len(self.pts)]

            # Check if the ray intersects with any of the vertices of the polygon. If it does,
            # just return that intersected vertex.
            if geometry.ray_point_intersection(self.kernel, theta, p1):
                return self.verts[i].incoming_edge, p1
            if geometry.ray_point_intersection(self.kernel, theta, p2):
                return self.verts[(i+1) % len(self.verts)].incoming_edge, p2

            t = geometry.ray_line_intersect(self.kernel, theta, p1, p2)
            if t != -1.0:
                intersect_pt = self.kernel + (np.array([math.cos(theta), math.sin(theta)]) * t)
                intersect_edge = None
                if np.allclose(p1, intersect_pt):
                    intersect_pt = p1
                    intersect_edge = self.verts[i].outgoing_edge
                elif np.allclose(p2, intersect_pt):
                    intersect_pt = p2
                    intersect_edge = self.verts[(i+1) % len(self.verts)].outgoing_edge
                else:
                    intersect_edge = self.verts[i].outgoing_edge

                return intersect_edge, intersect_pt
        assert False, 'No intersecting edge found! Your polygon is somehow not closed! ' + str(self)

    def connecting_line_length(self, alpha, s1_1, s1_2, s2_1, s2_2):
        """Computes the length of the line that connects two line segments and passes through
        the kernel. This connecting line can be thought of as the ray that originates from some
        point along one line segment, passes through the kernel, and intersects the other line
        segment. This is the function that we optimize in order to determine the shortest line
        that connects two line segments and passes through the kernel (the minimum diameter).

        Note that the endpoints of the current polyline segment are passed as s1. This is because
        we know that any ray originating from a point along a polyline segment and passing through
        the kernel will intersect with s2, by definition (the reverse of this is how the polyline
        was constructed in get_polyline_list()). By passing in the arguments in this way, we never
        have to worry about the ray never intersecting s2 during optimization.

        Args:
            alpha: How far along the first line segment the ray origin is. Must be in the range
                   [0, 1]. When alpha == 0, the ray origin is the first point of the line segment.
                   When alpha == 1, the ray is the second point of the line segment. When alpha == 0.5,
                   the ray origin at the midpoint of the line segment.
            s1_1: The first point of the first line segment.
            s1_2: The second point of the first line segment.
            s2_1: The first point of the second line segment.
            s2_2: The second point of the second line segment.

        Returns:
            The length of the line connecting the two line segments and passing through the kernel.

        Return type:
            float
        """
        u = (s1_2 - s1_1) / np.linalg.norm(s1_2 - s1_1) # direction vector of s1
        p1 = s1_1 + (alpha * u * np.linalg.norm(s1_2 - s1_1)) # point along s1
        centered_kernel = self.kernel - p1
        theta = math.atan2(centered_kernel[1], centered_kernel[0])
        ans = geometry.ray_line_intersect(p1, theta, s2_1, s2_2)
        if ans == -1:
            ans = sys.maxsize
        return ans

    def connecting_line_length_hyponuse_version(self, alpha, s1_1, s1_2, s2_1, s2_2):
        """Computes the length of the line that connects two line segments and passes through
        the kernel. This connecting line can be thought of as the ray that originates from some
        point along one line segment, passes through the kernel, and intersects the other line
        segment.

        This does not actually get used anywhere since it is more expensive to compute, but I
        implemented it because it is the way the optimization is described in the paper. I think this
        version is easier to interpret geometrically.

        Args:
            alpha: How far along the first line segment the ray origin is. Must be in the range
                   [0, 1]. When alpha == 0, the ray origin is the first point of the line segment.
                   When alpha == 1, the ray is the second point of the line segment. When alpha == 0.5,
                   the ray origin at the midpoint of the line segment.
            s1_1: The first point of the first line segment.
            s1_2: The second point of the first line segment.
            s2_1: The first point of the second line segment.
            s2_2: The second point of the second line segment.

        Returns:
            The length of the line connecting the two line segments and passing through the kernel.

        Return type:
            float
        """
        s1_vector = (s1_2 - s1_1) / np.linalg.norm(s1_2 - s1_1)
        x1 = geometry.project_point(s1_vector, self.kernel - s1_1) + s1_1
        h1 = np.linalg.norm(x1 - self.kernel)
        p1 = s1_1 + (alpha * s1_vector * np.linalg.norm(s1_2 - s1_1))
        b1 = np.linalg.norm(p1 - x1)
        l1 = math.sqrt(h1**2 + b1**2)

        s2_vector = (s2_2 - s2_1) / np.linalg.norm(s2_2 - s2_1)
        x2 = geometry.project_point(s2_vector, self.kernel - s2_1) + s2_1
        h2 = np.linalg.norm(x2 - self.kernel)
        ray_dir = self.kernel - p1
        ray_dir = math.atan2(ray_dir[1], ray_dir[0])
        p2 = p1 + (geometry.ray_line_intersect(p1, ray_dir, s2_1, s2_2) * np.array([math.cos(ray_dir), math.sin(ray_dir)]))
        b2 = np.linalg.norm(p2 - x2)
        l2 = math.sqrt(h2**2 + b2**2)

        return l1 + l2

    def compute_minimum_radius(self):
        """Compute the length of the shortest ray from the kernel to any boundary segment.

        Returns:
            The length of the shortest ray from the kernel to the boundary.

        Return type:
            float
        """
        shortest_radius = float('inf')
        for i in range(len(self.pts)):
            p1 = self.pts[i]
            p2 = self.pts[(i+1) % len(self.pts)]
            d = geometry.line_point_distance(p1, p2, self.kernel)
            if d < shortest_radius:
                shortest_radius = d
        return shortest_radius

    def compute_parameterizations(self):
        """Wrapper function to compute various boundary parameterizations and ray
        integrals of this polygon.
        """
        # Boundary parameterizations
        self.theta_boundary_parameterization = ([[0.0]], [[0.0]])
        # self.theta_boundary_parameterization = self.parameterize_boundary_by_theta(self.pts)
        # self.hypotenuse_boundary_parameterization = self.parameterize_boundary_by_hypotenuse(self.pts)

    def parameterize_boundary_by_hypotenuse(self, boundary_pts):
        """Parameterize the polygon boundary by the length of the lines connecting points
        along the boundary to the kernel. The lengths of these connecting lines are computed
        using the hypotenuse method.

        Args:
            boundary_pts: The vertices defining the polygon boundary.

        Returns:
            A tuple of two lists of lists. The first list contains the distances of the 
            points forming the hypotenuses for each segment of the boundary. The second 
            list contains the lengths of these hypotenuses.

        Return type:
            A tuple of lists of lists of floats: ([[float, ...], ...], [[float, ...], ...])
        """
        pieces_x = []
        pieces_y = []

        for i in range(len(boundary_pts)):
            p1 = boundary_pts[i]
            p2 = boundary_pts[(i+1) % len(boundary_pts)]

            # Compute the projection of the kernel onto the line formed by the line segment.
            # This projected point is the origin of the x-axis (the x-axis is the line formed
            # by the line segment under consideration).
            t_p2 = p2 - p1
            t_kernel = self.kernel - p1
            projected_kernel = geometry.project_point(t_p2, t_kernel) + p1
            h = np.linalg.norm(projected_kernel - self.kernel)

            # Account for which side of the x-axis origin the current segment endpoints appear on.
            if geometry.orient(self.kernel, p1, p2) != 0:
                x_min = np.linalg.norm(projected_kernel - p1)
                x_min *= geometry.orient(self.kernel, projected_kernel, p1)
                x_max = np.linalg.norm(projected_kernel - p2)
                x_max *= geometry.orient(self.kernel, projected_kernel, p2)
            # Colinear points. Just find the distance from the kernel to all the points along the
            # current boundary segment.
            else:
                x_min = np.linalg.norm(p1 - self.kernel)
                x_max = np.linalg.norm(p2 - self.kernel)

            x_vals = np.linspace(x_min, x_max, 100)
            y_vals = [math.sqrt(h**2 + x**2) for x in x_vals]
            if len(pieces_x) > 0:
                prev_x_lim = max(pieces_x[-1])
                length_along_boundary = np.linalg.norm(p1 - p2)
                offset_x_vals = list(np.linspace(0, length_along_boundary, 100))
                pieces_x.append([prev_x_lim + x for x in offset_x_vals])
            else:
                pieces_x.append(x_vals)
            pieces_y.append(y_vals)

        return (pieces_x, pieces_y)

    def parameterize_boundary_by_theta(self, boundary_pts):
        """Parameterize the polygon boundary by the length of the lines connecting points 
        along the boundary to the kernel. The lengths of these connecting lines are computed 
        using the ray-shooting method.

        Args:
            boundary_pts: The vertices defining the polygon boundary.

        Returns:
            A tuple of two lists of lists. The first list contains the distances of the 
            points forming the rays for each segment of the boundary. The second 
            list contains the lengths of these rays.

        Return type:
            A tuple of lists of lists of floats: ([[float, ...], ...], [[float, ...], ...])
        """
        pieces_x = []
        pieces_y = []

        for i in range(len(boundary_pts)):
            p1 = boundary_pts[i]
            p2 = boundary_pts[(i+1) % len(boundary_pts)]

            x_min, x_max = self.get_theta_domain(boundary_pts[0], p1, p2)
            if i == len(boundary_pts) - 1:
                x_max = math.pi + math.pi
            x_vals, y_vals = self.get_distance_values(boundary_pts[0], p1, p2, x_min, x_max)
            pieces_x.append(x_vals)
            pieces_y.append(y_vals)

        return (pieces_x, pieces_y)

    def get_distance_values(self, start_point, s1, s2, x_min, x_max):
        # Here, we have converted the "ray line intersection" function into a single equation,
        # since the only free variable is the angle (ray direction). Thus, we can come up with
        # a formula for a given line segment, where the constants of the formula are defined by
        # the kernel and the line segment end points. This function computes the equation constants.
        # is larger.
        p1 = self.kernel - s1
        p2 = s2 - s1

        # Points are not colinear. Just compute the distance from the kernel to the line segment
        # for all values of theta.
        if geometry.orient(self.kernel, s1, s2) != 0:
            x_vals = np.linspace(x_min, x_max, num=100)
            y_vals = [(p2[0]*p1[1] - p2[1]*p1[0]) / (p2[0]*-math.sin(x) + p2[1]*math.cos(x)) for x in x_vals]
        # Points are colinear. Just compute the distance from the kernel to all points along the
        # line segment (in the direction of s1 to s2).
        else:
            s1_dist = np.linalg.norm(self.kernel - s1)
            s2_dist = np.linalg.norm(self.kernel - s2)
            x_vals = np.linspace(0, np.linalg.norm(s1 - s2), 100)
            if s1_dist > s2_dist:
                y_vals = [s1_dist - x for x in x_vals]
            else:
                y_vals = [s1_dist + x for x in x_vals]
            x_vals = [x_min] * 100

        return x_vals, y_vals

    def get_theta_domain(self, start_point, s1, s2):
        x_min = math.atan2(s1[1] - self.kernel[1], s1[0] - self.kernel[0])
        x_max = math.atan2(s2[1] - self.kernel[1], s2[0] - self.kernel[0])
        if np.allclose(x_min, 0):
            x_min = 0
        if x_min < 0:
            x_min = math.pi + math.pi + x_min
        x_min = math.fmod(x_min, math.pi + math.pi)
        if x_max < 0:
            x_max = math.pi + math.pi + x_max
        return x_min, x_max

    def polar_area(self, x, kernel, s1, s2):
        # We return the ray length squared it's an integral of a polar curve.
        ray_length = geometry.ray_line_intersect(kernel, x, s1, s2)
        if ray_length < 0:
            if geometry.ray_point_intersection(kernel, x, s1) and \
                geometry.ray_point_intersection(kernel, x, s2):
                ray_length = min(np.linalg.norm(kernel - s1), np.linalg.norm(kernel - s2))
            elif geometry.ray_point_intersection(kernel, x, s1):
                ray_length= np.linalg.norm(kernel - s1)
            elif geometry.ray_point_intersection(kernel, x, s2):
                ray_length = np.linalg.norm(kernel - s2)
        if ray_length < 0:
            test = 4
        assert ray_length >= 0, 'Polar area is somehow negative!'
        return ray_length**2

    def compare_best_virt_non_overlap(self, phys_poly):
        optim = differential_evolution(self.best_alignment_helper, callback=MinimizeStopper(self.best_virt_alignment_helper, self, phys_poly), bounds=[(0, geometry.TWO_PI)], disp=True, args=(self, phys_poly))
        return optim['fun']

    def minus(self, other_poly):
        """Returns the area of `self` that does not overlap with `other_poly`.
        """
        shapely_self_poly = Polygon([[p[0]-self.kernel[0], p[1]-self.kernel[1]] for p in self.pts])
        shapely_other_poly = Polygon([[p[0]-other_poly.kernel[0], p[1]-other_poly.kernel[1]] for p in other_poly.pts])
        self_minus_other = shapely_self_poly.difference(shapely_other_poly)
        return self_minus_other.area

    def best_virt_alignment_helper(self, theta, virt_poly, phys_poly):
        rotated_virt_poly = virt_poly.rotate(theta)
        comparison = rotated_virt_poly.minus(phys_poly)
        return comparison    

    def compare_min_diameter(self, poly2):
        return abs(self.min_diameter - poly2.min_diameter)
    
    def compare_symmetric_diff(self, poly2):
        poly1 = self
        shapely_poly1 = Polygon([[p[0]-poly1.kernel[0], p[1]-poly1.kernel[1]] for p in poly1.pts])
        shapely_poly2 = Polygon([[p[0]-poly2.kernel[0], p[1]-poly2.kernel[1]] for p in poly2.pts])
        poly1_diff_poly2 = shapely_poly1.difference(shapely_poly2)
        poly2_diff_poly1 = shapely_poly2.difference(shapely_poly1)
        if poly1_diff_poly2.area == 0:
            if poly2_diff_poly1.area == 0:
                return 0
            else:
                return float('inf')
        # return poly2_diff_poly1.area - poly1_diff_poly2.area
        # return poly2_diff_poly1.area / poly1_diff_poly2.area
        return poly1_diff_poly2.area + poly2_diff_poly1.area

    def compute_polyline_integral(self, polyline, polygon):
        integral = 0
        kernel = polygon.kernel
        for i in range(len(polyline) - 1):
            p1 = polyline[i]
            p2 = polyline[i+1]

            # This edge of the polyline is colinear with the kernel, so it doesn't contribute
            # anything to the polyline integral. The endpoints of this edge will be accounted 
            # for in the preceeding and following edges' integrals.
            if geometry.orient_non_robust(kernel, p1, p2) == 0 or \
                geometry.EPSILON_CHECK(np.linalg.norm(p1 - p2)):
                continue

            # Compute the edge integral normally.
            p1_theta = math.atan2(p1[1] - kernel[1], p1[0] - kernel[0])
            p1_theta = geometry.angle2(p1 - kernel)
            if np.allclose(p1, polygon.pts[0]) and i == 0:
                p1_theta = 0
            p2_theta = math.atan2(p2[1] - kernel[1], p2[0] - kernel[0])
            p2_theta = geometry.angle2(p2 - kernel)
            if (np.allclose(p2, polygon.pts[0]) or geometry.EPSILON_CHECK(p2_theta)) and i + 1 == len(polyline) - 1:
                p2_theta = geometry.TWO_PI

            integral += integrate.quad(self.polar_area, a=p1_theta, b=p2_theta, args=(kernel, p1, p2))[0] * 0.5
        return integral

    def compute_critical_points(self, poly1, poly2):
        """Compute the theta values of two polygons' vertices, relative to each polygon's kernel.
        This method returns a list of points corresponding to all vertices of both polygons, where the
        list elements have been sorted by the order in which they appear during a simultaneous angular
        sweep along the border of two polygons. This list represents the domains for which we compute
        the ray length integrals for both polygons.

        Args:
            poly1: The first polygon to sweep.
            poly2: The second polygon to sweep.

        Returns:
            A list containing tuples that correspond to a vertex of either polygon. These tuples contain
            information about the vertex's angle relative to the respective kernel, the vertex's coordinates,
            and a pointer to the polygon object to which this vertex belongs. This list is sorted in
            ascending order by the angle relative to the kernel.
        
        Return type:
            A list of tuples.
        """
        poly1_thetas = poly1.convert_points_to_theta()
        poly2_thetas = poly2.convert_points_to_theta()
        crit_pts = []
        empty = (float('inf'), np.array([float('inf'), float('inf')]), poly1)

        while len(poly1_thetas) > 0 or len(poly2_thetas) > 0:
            poly1_theta = poly1_thetas[0] if len(poly1_thetas) > 0 else empty
            next_poly1_theta = poly1_thetas[1] if len(poly1_thetas) > 1 else empty
            poly2_theta = poly2_thetas[0] if len(poly2_thetas) > 0 else empty
            next_poly2_theta = poly2_thetas[1] if len(poly2_thetas) > 1 else empty
            
            if np.allclose(poly2_theta[0], poly1_theta[0]):
                crit_pts.append(poly2_theta)
                poly2_thetas.pop(0)
                crit_pts.append(poly1_theta)
                poly1_thetas.pop(0)
                if np.allclose(poly2_theta[0], next_poly2_theta[0]):
                    crit_pts.append(next_poly2_theta)
                    poly2_thetas.pop(0)
                if np.allclose(poly1_theta[0], next_poly1_theta[0]):
                    crit_pts.append(next_poly1_theta)
                    poly1_thetas.pop(0)
            elif poly2_theta[0] < poly1_theta[0]:
                crit_pts.append(poly2_theta)
                poly2_thetas.pop(0)
                if np.allclose(next_poly2_theta[0], poly2_theta[0]):
                    crit_pts.append(next_poly2_theta)
                    poly2_thetas.pop(0)
            elif poly1_theta[0] < poly2_theta[0]:
                crit_pts.append(poly1_theta)
                poly1_thetas.pop(0)
                if np.allclose(next_poly1_theta[0], poly1_theta[0]):
                    crit_pts.append(next_poly1_theta)
                    poly1_thetas.pop(0)

        return crit_pts

    def convert_points_to_theta(self):
        thetas_and_points = []
        for i in range(len(self.pts)):
            centered_pt = self.pts[i] - self.kernel
            theta = math.atan2(centered_pt[1], centered_pt[0])
            if i == 0 and (np.allclose(theta, math.pi+math.pi) or np.allclose(theta, 0)):
                theta = 0
            if theta < 0 and not np.allclose(theta, 0):
                theta += math.pi + math.pi
            thetas_and_points.append((theta, self.pts[i], self))
        return thetas_and_points

    def project_point_onto_boundary(self, point_to_project):
        """Note that point_to_project has already been centered around the origin (0, 0)!
        """
        closest_its_dist = float('inf')
        closest_its = None

        point_to_project += self.kernel

        # centered_self_pts = [pt - self.kernel for pt in self.pts]
        centered_self_pts = self.pts

        # On the boundary
        if geometry.point_inside_polygon(point_to_project, centered_self_pts) == 0:
            return point_to_project
        # Inside the polygon
        elif geometry.point_inside_polygon(point_to_project, centered_self_pts) == 1:
            ray_dir = math.atan2(point_to_project[1]-self.kernel[1], point_to_project[0]-self.kernel[0])
            ray_origin = np.array([0, 0])
            ray_origin = self.kernel

            return geometry.get_first_polygon_intersection(centered_self_pts, ray_origin, ray_dir)
        # Outside the polygon
        else:
            s1 = np.array([0, 0])
            s1 = self.kernel
            s2 = point_to_project

            best_its_pt = None
            min_dist = float('inf')
            for i in range(len(centered_self_pts)):
                p1 = centered_self_pts[i]
                p2 = centered_self_pts[(i+1) % len(centered_self_pts)]

                if geometry.line_segments_intersect(p1, p2, s1, s2):
                    its_pt = geometry.line_line_intersection(p1, p2, s1, s2)
                    if not np.allclose(its_pt, np.array([float('inf'), float('inf')])) and \
                        np.linalg.norm(its_pt - s1) < min_dist:
                        best_its_pt = its_pt
                        min_dist = np.linalg.norm(its_pt - s1)
            # best_its_pt += self.kernel
            return best_its_pt

    def rotate(self, theta):
        centered_pts = [v - self.kernel for v in self.pts]
        rotated_pts = [np.array([v[0]*math.cos(theta) - v[1]*math.sin(theta), 
                                v[1]*math.cos(theta) + v[0]*math.sin(theta)]) + self.kernel 
                                for v in centered_pts]
        return StarPolygon(rotated_pts, self.kernel)

    def find_best_alignment(self, other_poly):
        # optim = minimize_scalar(self.best_alignment_helper, bounds=(0, geometry.TWO_PI), method='bounded', options={'disp':3}, args=(self, other_poly))
        optim = differential_evolution(self.best_alignment_helper, callback=MinimizeStopper(self.best_alignment_helper, self, other_poly), bounds=[(0, geometry.TWO_PI)], disp=False, args=(self, other_poly))
        # optim = basinhopping(self.best_alignment_helper, x0=0, disp=True, minimizer_kwargs={'args':(self, other_poly)})
        # optim = brute(self.best_alignment_helper, ranges=(0, geometry.TWO_PI), disp=True, args=(self, other_poly))
        return optim['fun'], optim['x']

    def best_alignment_helper(self, theta, poly1, poly2):
        rotated_poly2 = poly2.rotate(theta)
        comparison = poly1.compare_symmetric_diff(rotated_poly2)
        return comparison

    def scale(self, scale_factor):
        self.kernel[0] /= scale_factor
        self.kernel[1] /= scale_factor
        for pt in self.pts:
            pt[0] /= scale_factor
            pt[1] /= scale_factor

    ####################################################################################################
    ######################################## UTILITY FUNCTIONS #########################################
    ####################################################################################################

    def point_inside(self, p):
        """Determines if a given point is inside the polygon.

        Args:
            p: The point whose insided-ness we wish to determine.

        Returns:
            1 if p is inside the polygon.
            -1 if p is outside the polygon.
            0 if is on the polygon boundary.

        Return type:
            int
        """
        return geometry.point_inside_polygon(p, self.pts)

    def remove_duplicates(self, pts):
        """Remove duplicate points from a list of points.

        Args:
            pts: The list of points from which duplicates should be removed.

        Returns:
            A list of points, with no duplicate points.

        Return type:
            List of points
        """
        new_list = []

        prev_pt = None
        for pt in pts:
            if prev_pt is None:
                new_list.append([pt[0], pt[1]])
                prev_pt = pt
                continue
            if not np.array_equal(pt, prev_pt) and not np.allclose(pt, prev_pt):
                new_list.append([pt[0], pt[1]])
                prev_pt = pt

        return new_list

    def show_data(self):
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        print(self)
        print('Area:', self.area)
        print('Minimum diameter:', self.min_diameter)
        print('Minimum radius:', self.min_radius)
        print('Average radius (hypotenuse method):', self.hypotenus_avg_radius)
        print('Ray integral (hypotenuse method):', self.hypotenuse_integral)

        # Graph the two boundary parameterizations
        fig, ax = plt.subplots(2)
        theta_pieces_x, theta_pieces_y = self.theta_boundary_parameterization
        hypo_pieces_x, hypo_pieces_y = self.hypotenuse_boundary_parameterization

        for i in range(len(hypo_pieces_x)):
            ax[0].plot(hypo_pieces_x[i], hypo_pieces_y[i], c='r')
        ax[0].set_title('Hypotenuse Boundary Parameterization')
        ax[0].set(xlabel='Distance along perimeter', ylabel='Distance from kernel to boundary')

        for i in range(len(theta_pieces_x)):
            ax[1].plot(theta_pieces_x[i], theta_pieces_y[i], c='b')
        ax[1].set_title('Theta Boundary Parameterization')
        ax[1].set(xlabel='Theta', ylabel='Distance from kernel to boundary')
        plt.show()

    def error_log(self, log_message=None):
        if log_message:
            print(log_message)
        print(str(self))

    def __str__(self):
        ret_str = '\n****\nKernel:\n   ' + str(self.kernel) + '\nVertices:\n'
        for p in self.pts:
            ret_str += '   [{}, {}],\n'.format(p[0], p[1])
        ret_str += '****'
        return ret_str

def graph_two_polygons(poly1, poly2):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(1)
    poly1_pieces_x, poly1_pieces_y = poly1.theta_boundary_parameterization
    poly2_pieces_x, poly2_pieces_y = poly2.theta_boundary_parameterization

    for i in range(len(poly1_pieces_x)):
        plt.plot(poly1_pieces_x[i], poly1_pieces_y[i], c='r')

    for i in range(len(poly2_pieces_x)):
        plt.plot(poly2_pieces_x[i], poly2_pieces_y[i], c='b')
        
    plt.title('Polygon Boundary Parameterization')
    # plt.set(xlabel='Theta', ylabel='Distance from kernel to boundary')

    # ax[1].set_title('Polygon 1 Boundary Parameterization')
    # ax[1].set(xlabel='Theta', ylabel='Distance from kernel to boundary')
    plt.show()        

if __name__ == '__main__':
    plain_square = StarPolygon([np.array([1.0, 1.0]), np.array([-1.0, 1.0]), np.array([-1.0, -1.0]), np.array([1.0, -1.0])], np.array([0, 0]))

    rotated_square = StarPolygon([np.array([1.00003154, 0.]), np.array([0.99996846, 0.00794235]), np.array([0.99202611, 1.00791081]), np.array([-1.00791081, 0.99202611]), np.array([-0.99202611, -1.00791081]), np.array([1.00791081, -0.99202611])], np.array([0, 0]))

    print(plain_square.compare_symmetric_diff(rotated_square))