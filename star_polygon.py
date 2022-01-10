import numpy as np
import math
import geometry
from scipy.optimize import minimize_scalar
from scipy import integrate
from scipy.optimize import brute, basinhopping, differential_evolution
from scipy.optimize import rosen
import sys
from shapely.geometry import Polygon

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
    """

    def __init__(self, pts, center=None, compute_area=False):
        """Constructor for the Polygon class.

        Args:
            pts: The vertices that define the boundary of the polygon, in CCW order.
            center: The kernel of the polygon. This is only used if the polygon we are creating
                    is a visibility polygon. If we just want a regular polygon (not a visibility
                    polygon), simply do not pass in any second argument into your constructor call.

                    Note that this is a little bit of a hack/abuse of "notation" since a star polygon
                    requires a kernel, but we are allowing the creation of a StarPolygon object without
                    a kernel. This was done to make the code simpler, so that I didn't have to make two 
                    classes (generic Polygon class and StarPolygon class). I just "recycle" this class
                    when creating polygon borders for obstacles and environment borders in environment.py
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

    def add_theta_0_point(self, pts):
        """Find the intersection point of the ray theta=0 from the kernel to the boundary, and
        add it to the polygon's list of vertices. This is done so that we have a standard starting
        point for all polygons for when we compute integrals. If the intersection point is already
        part of the input vertices, no additional point is added to the vertex list.

        Args:
            pts: The vertices of the polygon that were initially passed in.

        Returns:
            The list of vertices, with the intersection point for the ray theta=0 added to the list.

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
        """Cyclicly shift the order of vertices of the polygon such that the first vertex in the
        list is the vertex on the boundary at angle theta=0 relative to the polygon kernel.

        Args:
            pts: The vertices of the polygon that we wish to shift.

        Returns:
            The list of vertices, with the first vertex being located at angle theta=0 relative to
            the polygon kernel.

        Return type:
            List of numpy points.
        """
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

    def minus(self, other_poly):
        """Returns the area of `self` that does not overlap with `other_poly`.

        Args:
            other_polygon: The other polygon that will be "subtracted" from this polygon.

        Returns:
            The area of this polygon that does not overlap with other_poly.

        Return type:
            float
        """
        shapely_self_poly = Polygon([[p[0]-self.kernel[0], p[1]-self.kernel[1]] for p in self.pts])
        shapely_other_poly = Polygon([[p[0]-other_poly.kernel[0], p[1]-other_poly.kernel[1]] for p in other_poly.pts])
        self_minus_other = shapely_self_poly.difference(shapely_other_poly)
        return self_minus_other.area

    def rotate(self, theta):
        """Rotate the polygon vertices around the kernel by theta degrees (anti-clockwise).

        Args:
            theta: The amount to rotate each vertex of the polygon around kernel.

        Returns:
            A copy of the polygon, with the vertices rotated theta degrees around the kernel.

        Return type:
            A StarPolygon object.
        """
        centered_pts = [v - self.kernel for v in self.pts]
        rotated_pts = [np.array([v[0]*math.cos(theta) - v[1]*math.sin(theta),
                                v[1]*math.cos(theta) + v[0]*math.sin(theta)]) + self.kernel
                                for v in centered_pts]
        return StarPolygon(rotated_pts, self.kernel)

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

    def error_log(self, log_message=None):
        """Print the polygon and optionally an error message to the terminal, to help with
        debugging.

        Args:
            log_message: The optional error message to be printed.

        Returns:
            None
        """
        if log_message:
            print(log_message)
        print(str(self))

    def __str__(self):
        """Convert the polygon object to a string descriptor so that it can be
        easily printed to the terminal.
        """
        ret_str = '\n****\nKernel:\n   ' + str(self.kernel) + '\nVertices:\n'
        for p in self.pts:
            ret_str += '   [{}, {}],\n'.format(p[0], p[1])
        ret_str += '****'
        return ret_str

if __name__ == '__main__':
    plain_square = StarPolygon([np.array([1.0, 1.0]), np.array([-1.0, 1.0]), np.array([-1.0, -1.0]), np.array([1.0, -1.0])], np.array([0, 0]))
    rotated_square = StarPolygon([np.array([1.00003154, 0.]), np.array([0.99996846, 0.00794235]), np.array([0.99202611, 1.00791081]), np.array([-1.00791081, 0.99202611]), np.array([-0.99202611, -1.00791081]), np.array([1.00791081, -0.99202611])], np.array([0, 0]))

    print(plain_square.minus(rotated_square))