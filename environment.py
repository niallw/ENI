import os
import itertools
from itertools import product
import numpy as np
import math
import xml.etree.ElementTree as ET
import star_polygon
import triangle as tr
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib import collections  as mc
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import PercentFormatter
import matplotlib.tri as tri
import matplotlib.cm as cm 
import random
import pathlib
from tqdm import tqdm
from datetime import datetime
from skgeom import *
import statistics
import geometry
import operator

class Environment():
    """An environment class.

    This class stores data related to a physical or virtual environment, which is defined in an
    XML file. This class is useful for sampling points and paths in the environment.

    Attributes:
        name: The name of the environment.
        border: The polygon that defines the boundary of the environment.
        center: The center of the environment. This is not always the centroid.
        area: The area of the walkable space in the environment.
        obstacles: A list of obstacles present in the environment. Does not include the boundary.
        triangulation_verts: The vertices defining the triangulation of the walkable space in 
                              the environment.
        triangulation_segs: The line segments defining the triangulation of the walkable space in
                             the environment.
        triangulation_holes: The holes in the environment, for which we should not triangulate.
                              These holes are used to identify the obstacles of the environment, so
                              they must always be inside an obstacle. The holes are represented as
                              2D points (that lie inside the obstacle).
        triangle_areas: The areas of the triangles in the triangulation of the environment. This is
                         used when uniformly sampling points in the environment.
        normalized_tringle_areas: The normalized areas of the triangles in the triangulation of the
                                   environment. This is used when uniformly sampling points in the
                                   environment.
        triangle_distribution: A histogram of the triangles in the triangulation of the environment. 
                                This list is the indices of triangles (freq. of index scaled by 
                                triangle's relative area). This is used when uniformly sampling points 
                                in the environment.
        triangulated_env: The triangulated environment (from the triangle library).
        visibility_segments: A representation of the environment boundary and obstacles as a list of
                              2D line segments. This is used for visibility computations.
        num_samples: The number of samples taken in the environment.
        sampled_points: A list of the points that were sampled from the environment.
        sampled_visibility_polygons: A list of the visibility polygons computed at each of the sampled
                                      points.
        sampled_min_diameters: A list of the minimum diameters of the visibility polygons that were
                                computed at the sampled points.
    """

    def __init__(self, env_file, triangle_area=None):
        self.file = env_file
        self.name = None
        self.border = None # Polygon object
        self.center = None
        self.width = None
        self.height = None
        self.area = None
        self.obstacles = [] # List of polygon objects
        self.triangulation_verts = None
        self.triangulation_segs = None
        self.triangulation_holes = None
        self.triangle_areas = []
        self.normalized_tringle_areas = []
        self.triangle_distribution = []
        self.triangulated_env = None
        self.num_samples = None
        self.sampled_points = []
        self.triangulation_visibility_polygons = []
        self.random_sampled_visibility_polygons = []
        self.sampled_min_diameters = []
        self.visibility_arrangement = arrangement.Arrangement()
        self.tri_expansion = None

        self.parse_env_file(env_file)
        self.compute_area()
        self.compute_extents()
        if triangle_area == None:
            # self.triangulate(area=0.01)
            # self.triangulate(area=.5)
            # self.triangulate(area=1)
            # self.triangulate(area=(200 / self.area))
            self.triangulate(area=(self.area / 500))
        else:
            self.triangulate(area=triangle_area)
        self.build_visibility_polygons()

    def compute_area(self):
        obs_areas = [o.area for o in self.obstacles]
        obs_area = sum(obs_areas)
        self.area = self.border.area - obs_area

    def compute_extents(self):
        min_x = min(p[0] for p in self.border.pts)
        max_x = max(p[0] for p in self.border.pts)
        min_y = min(p[1] for p in self.border.pts)
        max_y = max(p[1] for p in self.border.pts)
        for o in self.obstacles:
            min_x = min(min_x, min([p[0] for p in o.pts]))
            max_x = max(max_x, max([p[0] for p in o.pts]))
            min_y = min(min_y, min([p[1] for p in o.pts]))
            max_y = max(max_y, max([p[1] for p in o.pts]))
        self.extents_x = (max_x - min_x) / 2
        self.extents_y = (max_y - min_y) / 2
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y

        sum_x = sum([p[0] for p in self.border.pts])
        sum_x /= len(self.border.pts)
        sum_y = sum([p[1] for p in self.border.pts])
        sum_y /= len(self.border.pts)
        self.centroid = np.array([sum_x, sum_y])

    def parse_env_file(self, env_file):
        tree = ET.parse(env_file)
        root = tree.getroot()
        self.name = root.attrib['name']
        print('*' * len('* Parsing environment: {} *'.format(self.name)))
        print('* Parsing environment: {} *'.format(self.name))
        print('*' * len('* Parsing environment: {} *'.format(self.name)))
        
        for child in root:
            if child.tag == 'border':
                self.parse_border(child)

            elif child.tag == 'obstacle':
                self.parse_obs(child)
        
        # Convert environmnet to segments, for visibility computations
        for i in range(len(self.border.pts)):
            p1 = self.border.pts[i]
            p2 = self.border.pts[(i+1) % len(self.border.pts)]
            self.visibility_arrangement.insert(Segment2(Point2(p1[0], p1[1]), Point2(p2[0], p2[1])))
        for o in self.obstacles:
            for i in range(len(o.pts)):
                p1 = o.pts[i]
                p2 = o.pts[(i+1) % len(o.pts)]
                self.visibility_arrangement.insert(Segment2(Point2(p1[0], p1[1]), Point2(p2[0], p2[1])))

        self.width = abs(min([p[0] for p in self.border.pts]) - max([p[0] for p in self.border.pts]))
        self.height = abs(min([p[1] for p in self.border.pts]) - max([p[1] for p in self.border.pts]))
        print('--- Finished environment parsing ---')

    def parse_border(self, elem):
        print('Parsing {}...'.format(elem.attrib['name']))
        center = None
        verts = []

        for child in elem:
            # Border of the environment
            if len(child.attrib) == 0 and child.tag == 'vertices':
                for vert in child:
                    v = np.array([float(vert.text.split(',')[0]), float(vert.text.split(',')[1])])
                    verts.append(v)
            # Center of the environment (usually [0.0, 0.0])
            elif child.attrib['type'] == 'center':
                center = np.array([float(child.text.split(',')[0]), float(child.text.split(',')[1])])

        # Add a random Steiner point to break the first wall into 2 segments. This is done to 
        # ensure that we compute different Delaunay triangulations if we are given a pair of 
        # identical environments.
        p1 = verts[0]
        p2 = verts[1]
        t = random.uniform(0, 1)
        steiner_pt = (t * p1) + ((1-t) * p2)
        verts.insert(1, steiner_pt)
        self.border = star_polygon.StarPolygon(verts, compute_area=True)
        self.center = center

    def parse_obs(self, elem):
        print('Parsing {}...'.format(elem.attrib['name']))
        verts = []

        for child in elem:
            for vert in child:
                v = np.array([float(vert.text.split(',')[0]), float(vert.text.split(',')[1])])
                verts.append(v)

        self.obstacles.append(star_polygon.StarPolygon(verts, compute_area=True))

    def triangulate(self, area=0.1):
        print('--- Triangulating environment ---')
        print('CDT area:', area)
        # Extract the segment data from the environment border and obstacles
        walls = [[v[0], v[1]] for v in self.border.pts]
        obs = []
        for o in self.obstacles:
            temp = [[v[0], v[1]] for v in o.pts]
            obs.append(temp)

        # Format the data so it can be used by the triangle library
        self.triangulation_verts, self.triangulation_segs, self.triangulation_holes = self.get_triangulation_data(walls, obs)

        # Let there be triangles!
        if len(self.triangulation_holes) == 0:
            A = dict(vertices=self.triangulation_verts, segments=self.triangulation_segs)
        else:
            A = dict(vertices=self.triangulation_verts, segments=self.triangulation_segs, holes=self.triangulation_holes)
        # self.triangulated_env = tr.triangulate(A, 'pa0.1q')
        print('crash1')
        self.triangulated_env = tr.triangulate(A, 'pa{}'.format(area))
        # self.triangulated_env = tr.triangulate(A, 'p')
        print('crash2')

        # Compute the areas of the triangles. This is used to unformly sample the triangulation.
        for tri in self.triangulated_env['triangles']:
            p1 = self.triangulated_env['vertices'][tri[0]]
            p2 = self.triangulated_env['vertices'][tri[1]]
            p3 = self.triangulated_env['vertices'][tri[2]]
            self.triangle_areas.append(self.triangle_area(p1, p2, p3))
        self.area = sum(self.triangle_areas)
        self.normalized_tringle_areas = [int(round(x / min(self.triangle_areas))) for x in self.triangle_areas]
        for i in range(len(self.normalized_tringle_areas)):
            self.triangle_distribution.extend([i] * self.normalized_tringle_areas[i])
        print('--- Finished triangulation ---')

    def triangle_area(self, p1, p2, p3):
        return abs(((p1[0]*(p2[1] - p3[1])) + (p2[0]*(p3[1] - p1[1])) + (p3[0]*(p1[1] - p2[1]))) * 0.5)

    def get_triangulation_data(self, walls, obs):
        verts = []
        segments = []
        holes = []

        idx = 0
        verts_offset = 0
        for w in walls:
            verts.append(w)
        for w in walls:
            segments.append((idx, (idx+1) % len(walls)))
            idx += 1

        verts_offset = len(verts)
        for obstacle in obs:
            for o in obstacle:
                verts.append(o)
        for obstacle in obs:
            idx = 0
            for o in obstacle:
                segments.append((idx+verts_offset, ((idx+1) % len(obstacle))+verts_offset))
                idx += 1
            verts_offset += len(obstacle)
            holes.append(self.get_point_inside(obstacle))

        return verts, segments, holes

    def build_visibility_polygons(self):
        # One-time precomputation to aid in visibility queries
        self.tri_expansion = TriangularExpansionVisibility(self.visibility_arrangement) 
        tri_x = [v[0] for v in self.triangulated_env['vertices']]
        tri_y = [v[1] for v in self.triangulated_env['vertices']]
        print('Building visibility polygons for triangulation vertices.')
        for i in tqdm(range(len(tri_x))):
            vis_poly = self.build_vis_poly_helper(tri_x[i], tri_y[i])
            if vis_poly:
                self.triangulation_visibility_polygons.append(vis_poly)

        # # Build vis polygons that are centered at points randomly sampled across the environment.
        # # These polygons are NOT centered on the vertices of the triangulation.
        # print('Building visibility polygons for randomly sampled points.')
        # self.num_samples = int(self.SAMPLES_PER_SQ_METER * self.area) + 1
        # for i in tqdm(range(self.num_samples)):
        #     pt = self.sample_environment()
        #     vis_poly = self.build_vis_poly_helper(pt[0], pt[1])
        #     if vis_poly:
        #         self.sampled_points.append(pt)
        #         self.random_sampled_visibility_polygons.append(vis_poly)
        self.random_sampled_visibility_polygons = []

        print('num DT vis polys:', len(self.triangulation_visibility_polygons))
        print('num random sample vis polys:', len(self.random_sampled_visibility_polygons))
        # # Delete this because it prevents us from being able to do multithreading
        # self.tri_expansion = None
        # self.visibility_arrangement = None

    def build_vis_poly_helper(self, tri_x, tri_y):
        p = [tri_x, tri_y]
        if not self.point_is_legal(p):
            return None
        else:
            query_point = Point2(tri_x, tri_y)
            face = self.visibility_arrangement.find(query_point)
            vis = self.tri_expansion.compute_visibility(query_point, face)
            vis = [np.array([float(v.point().x()), float(v.point().y())]) for v in vis.vertices]
            return star_polygon.StarPolygon(vis, np.array([tri_x, tri_y]))
            
    def get_point_inside(self, verts):
        segs = [(i, (i+1) % len(verts)) for i in range(len(verts))]
        A = dict(vertices=verts, segments=segs)
        B = tr.triangulate(A, 'p')
        idx_p1 = B['triangles'].tolist()[0][0]
        idx_p2 = B['triangles'].tolist()[0][1]
        idx_p3 = B['triangles'].tolist()[0][2]
        p1 = B['vertices'].tolist()[idx_p1]
        p2 = B['vertices'].tolist()[idx_p2]
        p3 = B['vertices'].tolist()[idx_p3]

        return [(p1[0] + p2[0] + p3[0]) / 3, (p1[1] + p2[1] + p3[1]) / 3]

    def sample_environment(self):
        triangle_index = random.choice(self.triangle_distribution)
        tri = self.triangulated_env['triangles'][triangle_index]
        p1 = self.triangulated_env['vertices'][tri[0]]
        p2 = self.triangulated_env['vertices'][tri[1]]
        p3 = self.triangulated_env['vertices'][tri[2]]
        sampled_point = self.sample_triangle(p1, p2, p3)
        while not self.point_is_legal(sampled_point):
            sampled_point = self.sample_triangle(p1, p2, p3)
        assert self.point_is_legal(sampled_point), 'Sampled point is not valid!'
        return sampled_point

    def sample_triangle(self, p1, p2, p3):
        p1 = np.array(p1)
        p2 = np.array(p2)
        p3 = np.array(p3)
        r1 = random.uniform(0, 1)
        r2 = random.uniform(0, 1)

        sampled_pt = ((1 - math.sqrt(r1)) * p1) + (math.sqrt(r1) * (1 - r2) * p2) + (math.sqrt(r1) * r2 * p3)
        return [sampled_pt[0], sampled_pt[1]]

    def point_is_legal(self, p):
        # Point is not inside the border of the environment
        if self.border.point_inside(p) != 1:
            return False

        # Check if point is inside any obstacle
        for o in self.obstacles:
            if o.point_inside(p) != -1:
                return False

        return True

    def get_distance_in_direction(self, pos, direction):
        closest_dist = float('inf')
        for i in range(len(self.border.pts)):
            p1 = self.border.pts[i]
            p2 = self.border.pts[(i+1) % len(self.border.pts)]
            its = geometry.ray_line_intersect(pos, direction, p1, p2)
            if its >= 0 and its < closest_dist:
                closest_dist = its

        for obs in self.obstacles:
            for i in range(len(obs.pts)):
                p1 = obs.pts[i]
                p2 = obs.pts[(i+1) % len(obs.pts)]
                its = geometry.ray_line_intersect(pos, direction, p1, p2)
                if its >= 0 and its < closest_dist:
                    closest_dist = its

        assert closest_dist != float('inf'), 'Could not find a closest object from position ({}, {}) in direction {} for environment {}!'.format(pos[0], pos[1], direction, self.name)
        return closest_dist

    def optimal_comparison_helper(self, virt_poly, phys_poly_list):
        best_val = float('inf')
        for phys_poly in phys_poly_list:
            best_virt_poly = self.find_best_match(phys_poly)
            cur_val = best_virt_poly.compare_non_overlap(phys_poly)
            if cur_val < best_val:
                best_val = cur_val
        return best_val

    def get_triangulation_vis_poly(self, kernel):
        listcopy = list.copy(self.triangulation_visibility_polygons)
        listcopy.sort(key=lambda x: np.linalg.norm(x.kernel - kernel))
        return listcopy[0]
        ttt = 3
        # TODO: FIXME: use a hashmap?
        for poly in self.triangulation_visibility_polygons:
            # if np.allclose(poly.kernel, kernel):
            if np.linalg.norm(poly.kernel - kernel) <= 0.01:
                return poly
        print('Failed to find a visibility polygon for the point: {}'.format(kernel))

    def compare_optimal(self, other_env, extra_title=''):
        self.similarities = []
        scalar_field_values = []
        print('Comparing environments {} and {}...'.format(self.name, other_env.name))
        # self_polygons = list.copy(self.triangulation_visibility_polygons)
        virt_polygons = list.copy(self.random_sampled_visibility_polygons)
        # other_polygons = list.copy(other_env.random_sampled_visibility_polygons)
        other_polygons = list.copy(other_env.triangulation_visibility_polygons)

        # for i in tqdm(range(len(self_polygons))):
        
        tri_x = [v[0] for v in other_env.triangulated_env['vertices']]
        tri_y = [v[1] for v in other_env.triangulated_env['vertices']]
        for i in tqdm(range(len(tri_x))):
            p = [tri_x[i], tri_y[i]]
            vals = []
            if not other_env.point_is_legal(p):
                # Tuple format is (extra other poly area, phys pos, best match virt pos)
                vals.append((0, p, p))
            else:
                cur_phys_poly = other_env.get_triangulation_vis_poly(np.array(p))
                # Tuple format is (extra other poly area, phys pos, best match virt pos)
                temp_vals = [(virt_poly.minus(cur_phys_poly), cur_phys_poly.kernel, virt_poly.kernel) for virt_poly in virt_polygons]
                temp_vals.sort(key=lambda x: x[0])
                vals.append(temp_vals[0])
            self.similarities.append(vals[0][0])
            scalar_field_values.append(vals[0])

        # for i in tqdm(range(len(other_polygons))):
        #     cur_phys_poly = other_polygons[i]
        #     # Tuple format is (extra other poly area, phys pos, best match virt pos)
        #     vals = [(virt_poly.minus(cur_phys_poly), cur_phys_poly.kernel, virt_poly.kernel) for virt_poly in virt_polygons]
        #     # vals = [(self_polygons[i].minus(cur_poly), self_polygons[i].kernel, cur_poly.kernel) for cur_poly in other_polygons]
        #     vals.sort(key=lambda x: x[0])
        #     self.similarities.append(vals[0][0])
        #     scalar_field_values.append(vals[0])

        # Draw the line connection graphs
        current_time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S-")
        env_width = abs(min([p[0] for p in self.border.pts]) - max([p[0] for p in self.border.pts]))
        env_height = abs(min([p[1] for p in self.border.pts]) - max([p[1] for p in self.border.pts]))
        virt_offset = np.array([env_width+20, env_height+20])
        phys_pos_x = [p[1][0] for p in scalar_field_values]
        phys_pos_y = [p[1][1] for p in scalar_field_values]
        virt_pos_x = [p[2][0] + virt_offset[0] for p in scalar_field_values]
        virt_pos_y = [p[2][1] + virt_offset[1] for p in scalar_field_values]
        fig, ax = plt.subplots(1)
        # Draw phys env
        num_pts = len(other_env.border.pts)
        phys_border = [(other_env.border.pts[i], other_env.border.pts[(i+1) % num_pts]) for i in range(num_pts)]
        phys_lc = mc.LineCollection(phys_border, colors=np.array([(0, 0, 0, 1)]), linewidths=1)
        ax.add_collection(phys_lc)
        # Environment obstacles
        for o in other_env.obstacles:
            x = [pt[0] for pt in o.pts]
            y = [pt[1] for pt in o.pts]
            plt.fill(x, y, 'black')

        # Draw virt env
        num_pts = len(self.border.pts)
        virt_border = [(self.border.pts[i]+virt_offset, self.border.pts[(i+1) % num_pts]+virt_offset) for i in range(num_pts)]
        virt_lc = mc.LineCollection(virt_border, colors=np.array([(0, 0, 0, 1)]), linewidths=1)
        ax.add_collection(virt_lc)
        # Environment obstacles
        for o in self.obstacles:
            x = [pt[0]+virt_offset[0] for pt in o.pts]
            y = [pt[1]+virt_offset[1] for pt in o.pts]
            plt.fill(x, y, 'black')

        # Draw lines connecting the points
        connection_lines = []
        for i in range(len(phys_pos_x)):
            if other_env.point_is_legal([phys_pos_x[i], phys_pos_y[i]]):
                connection_lines.append((np.array([phys_pos_x[i], phys_pos_y[i]]), np.array([virt_pos_x[i], virt_pos_y[i]])))
        # connection_lines = [(np.array([phys_pos_x[i], phys_pos_y[i]]), np.array([virt_pos_x[i], virt_pos_y[i]])) for i in range(len(phys_pos_x))]
        connection_lines_lc = mc.LineCollection(connection_lines, colors=np.array([(1, 0, 0, 0.5)]), linewidths=1)
        ax.add_collection(connection_lines_lc) 
        # ax.set_aspect('equal')
        # ax.set_xlim(-10, 30)
        # ax.set_ylim(-10, 30)   
        ax.autoscale(True)
        plt.title('{} best match connections\n{}'.format(other_env.name, extra_title))
        plt.savefig(os.path.join('img', '{}{}_and_{}_best_match_connections.png'.format(current_time, other_env.name, other_env.name).replace(' ', '_')), dpi=1000)
        # plt.show()
        # exit()

        # Draw scalar field
        tri_x = [v[0] for v in other_env.triangulated_env['vertices']]
        tri_y = [v[1] for v in other_env.triangulated_env['vertices']]
        for i in range(len(tri_x)):
            p = [tri_x[i], tri_y[i]]
            # print(p)
            if not other_env.point_is_legal(p):
                scalar_field_values.append((0, p, p))

        # tri_x = [v[0] for v in other_env.triangulated_env['vertices']]
        # tri_y = [v[1] for v in other_env.triangulated_env['vertices']]
        # for i in range(len(tri_x)):
        #     p = [tri_x[i], tri_y[i]]
        #     # print(p)
        #     if not other_env.point_is_legal(p):
        #         scalar_field_values.append((0, p, p))


        # Draw scalar field visualizing the value of the best-case match for each point checked
        fig, ax = plt.subplots(1)
        # Environment borders
        num_pts = len(other_env.border.pts)
        border = [(other_env.border.pts[i], other_env.border.pts[(i+1) % num_pts]) for i in range(num_pts)]
        lc = mc.LineCollection(border, colors=np.array([(0, 0, 0, 1)]), linewidths=1)
        ax.add_collection(lc)
        # Environment obstacles
        for o in other_env.obstacles:
            x = [pt[0] for pt in o.pts]
            y = [pt[1] for pt in o.pts]
            plt.fill(x, y, 'black')
        # Draw scalar field
        tri_x = [v[0] for v in other_env.triangulated_env['vertices']]
        tri_y = [v[1] for v in other_env.triangulated_env['vertices']]
        for i in range(len(tri_x)):
            p = [tri_x[i], tri_y[i]]
            # print(p)
            if not other_env.point_is_legal(p):
                scalar_field_values.append((0, p, p))

        tri_list = [list(t) for t in other_env.triangulated_env['triangles']]
        pt_vals = [val[0] for val in scalar_field_values]
        pt_x = [val[1][0] for val in scalar_field_values]
        pt_y = [val[1][1] for val in scalar_field_values]
        print('len pt_x:', len(pt_x))
        print('len pt_y:', len(pt_y))
        print('len tri_x:', len(tri_x))
        print('len tri_y:', len(tri_y))
        print('len tri_list:', len(tri_list))
        print('len pt_vals:', len(pt_vals))
        plot_triangulation = tri.Triangulation(pt_x, pt_y, tri_list)
        # plt.scatter(pt_x, pt_y, s=5, c=pt_vals)
        plt.tricontourf(plot_triangulation, pt_vals)
        verts = np.array(other_env.triangulated_env['vertices'])
        ax.triplot(verts[:, 0], verts[:, 1], other_env.triangulated_env['triangles'], 'k,-', linewidth=.5)
        ax.set_aspect('equal')
        # ax.margins(-2.5,)
        # plt.xlim(-2.5, 0)
        # plt.ylim(0, 2.5)
        plt.title('{} scalar field\n{}'.format(other_env.name, extra_title))
        plt.savefig(os.path.join('img', '{}{}_and_{}_optimal_comparison_scalar_field.png'.format(current_time, other_env.name, other_env.name).replace(' ', '_')), dpi=1000)
        # plt.show()

        # Create the figure of the two environments we just compared
        current_time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S-")
        fig = plt.figure(constrained_layout=True)
        gs = GridSpec(2, 2, figure=fig)
        ax1 = fig.add_subplot(gs[:, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 1], sharex=ax2, sharey=ax2)
        # The histogram of the data
        ax1.hist(self.similarities, 50, weights=np.ones(len(self.similarities)) / len(self.similarities), facecolor='g', alpha=0.75)
        ax1.set_title('{} and {}\nOptimal comparison\n{}'.format(self.name, other_env.name, extra_title))
        ax1.yaxis.set_major_formatter(PercentFormatter(1))
        ax1.grid(True)
        # The environments that were compared
        ax2 = self.update_axis(ax2)
        ax3 = other_env.update_axis(ax3)
        # Save the results
        plt.savefig(os.path.join('img', '{}{}_and_{}_optimal_comparison.png'.format(current_time, self.name, other_env.name).replace(' ', '_')), dpi=1000)
        with open(os.path.join('img', '{}{}_and_{}_optimal_comparison_values.txt'.format(current_time, self.name, other_env.name).replace(' ', '_')), 'w') as f:
            f.write('min value: {} | max value: {} | std. dev.: {} | mean: {} | median: {}\n'.format(min(self.similarities), max(self.similarities), np.std(self.similarities), statistics.mean(self.similarities), statistics.median(self.similarities)))
            for item in self.similarities:
                f.write("%s, " % item)

    def find_best_match(self, self_poly):
        best_match = None
        best_val = float('inf')
        for i in range(len(self.random_sampled_visibility_polygons)):
            # print('Comparing against virtual polygon #{}/{}'.format(i, len(self.triangulation_visibility_polygons)))
            # virt_pt = self.sample_environment()
            # virt_poly = self.triangulation_visibility_polygons[i]
            # virt_poly = self.build_vis_poly_helper(virt_pt[0], virt_pt[1])
            virt_poly = self.random_sampled_visibility_polygons[i]
            # diff = virt_poly.compare_best_virt_non_overlap(self_poly) # rotations
            diff = virt_poly.compare_non_overlap(self_poly) # no rotations

            if diff < best_val:
                best_val = diff
                best_match = virt_poly
            
        return best_val

    def compare(self, other_env, extra_title=''):
        print('Comparing environments {} and {}...'.format(self.name, other_env.name))
        e1_polygons = list.copy(self.triangulation_visibility_polygons)
        e2_polygons = list.copy(other_env.sampled_visibility_polygons)
        polygon_pairs = []
        for i in range(max(len(e1_polygons), len(e2_polygons))):
            polygon_pairs.append((random.choice(e1_polygons), random.choice(e2_polygons)))
        self.similarities = []

        print('Env. 1 has {} polygons.'.format(len(e1_polygons)))
        print('Env. 2 has {} polygons.'.format(len(e2_polygons)))
        for i in range(len(polygon_pairs)):
            pair = polygon_pairs[i]
            if i % 100 == 0:
                print('Comparison #{}/{}'.format(i, len(polygon_pairs)))
                
            val = pair[0].compare_symmetric_diff(pair[1])
            # val = self.compare_vis_polys_helper(pair[0], pair[1])
            # val = pair[0].find_best_alignment(pair[1])[1]
            if val == float('inf'):
                continue
            self.similarities.append(val)

            # theta = 0
            # num_rotations = 72 # 5 degrees
            # amount_to_rotate = geometry.TWO_PI / num_rotations 
            # increment = (math.pi) / 20
            # for i in range(num_rotations):
            #     theta = amount_to_rotate * (i)
            #     rotated_poly2 = pair[1].rotate(theta)

            #     val = self.compare_vis_polys_helper(pair[0], rotated_poly2)
            #     # val = pair[0].find_best_alignment(pair[1])[1]
            #     if val == float('inf'):
            #         continue
            #     self.similarities.append(val)

        # Create the figure of the two environments we just compared
        current_time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S-")
        fig = plt.figure(constrained_layout=True)
        gs = GridSpec(2, 2, figure=fig)
        ax1 = fig.add_subplot(gs[:, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 1], sharex=ax2, sharey=ax2)
        # The histogram of the data
        # Fixing random state for reproducibility
        ax1.hist(self.similarities, 50, weights=np.ones(len(self.similarities)) / len(self.similarities), facecolor='g', alpha=0.75)
        ax1.set_title('{} and {}\n{}'.format(self.name, other_env.name, extra_title))
        ax1.yaxis.set_major_formatter(PercentFormatter(1))
        # ax1.set_xlim()
        ax1.grid(True)
        # The environments that were compared
        ax2 = self.update_axis(ax2)
        ax3 = other_env.update_axis(ax3)
        plt.savefig(os.path.join('img', '{}{}_and_{}_comparison.png'.format(current_time, self.name, other_env.name).replace(' ', '_')), dpi=1000)

        with open(os.path.join('img', '{}{}_and_{}_comparison_values.txt'.format(current_time, self.name, other_env.name).replace(' ', '_')), 'w') as f:
            f.write('min value: {} | max value: {} | std. dev.: {} | mean: {} | mode: {}\n'.format(min(self.similarities), max(self.similarities), np.std(self.similarities), statistics.mean(self.similarities), statistics.mode(self.similarities)))
            for item in self.similarities:
                f.write("%s, " % item)

        # plt.show()

    def update_axis(self, ax):
        num_pts = len(self.border.pts)
        border = [(self.border.pts[i], self.border.pts[(i+1) % num_pts]) for i in range(num_pts)]
        lc = mc.LineCollection(border, colors=np.array([(0, 0, 0, 1)]), linewidths=1)
        ax.add_collection(lc)

        x = [-0.1, 0.1]
        y = [-0.1, 0.1]
        ax.fill(x, y, 'white')
        
        # Environment obstacles
        for o in self.obstacles:
            x = [pt[0] for pt in o.pts]
            y = [pt[1] for pt in o.pts]
            ax.fill(x, y, 'black')

        ax.set_aspect('equal')
        ax.set_title(self.name)
        return ax

    def compare_vis_polys_helper(self, poly1, poly2):
        return poly1.compare(poly2)
    
    def draw(self, paths=[]):
        fig, ax = plt.subplots(1)

        # Environment borders
        num_pts = len(self.border.pts)
        border = [(self.border.pts[i], self.border.pts[(i+1) % num_pts]) for i in range(num_pts)]
        lc = mc.LineCollection(border, colors=np.array([(0, 0, 0, 1)]), linewidths=1)
        ax.add_collection(lc)
        
        # Environment obstacles
        for o in self.obstacles:
            x = [pt[0] for pt in o.pts]
            y = [pt[1] for pt in o.pts]
            plt.fill(x, y, 'black')

        # Optional paths passed as an argument
        for p in paths:
            x = [pt.position[0] for pt in p.path]
            y = [pt.position[1] for pt in p.path]
            plt.plot(x, y)

        ax.set_aspect('equal')
        plt.title(self.name)
        plt.show()
        return plt

    def draw_sampled_points(self):
        fig, ax = plt.subplots(1)

        # Environment borders
        num_pts = len(self.border.pts)
        border = [(self.border.pts[i], self.border.pts[(i+1) % num_pts]) for i in range(num_pts)]
        lc = mc.LineCollection(border, colors=np.array([(0, 0, 0, 1)]), linewidths=1)
        ax.add_collection(lc)
        
        # Environment obstacles
        for o in self.obstacles:
            x = [pt[0] for pt in o.pts]
            y = [pt[1] for pt in o.pts]
            plt.fill(x, y, 'black')

        sampled_pts_x = [v[0] for v in self.sampled_points]
        sampled_pts_y = [v[1] for v in self.sampled_points]
        plt.scatter(sampled_pts_x, sampled_pts_y, s=1)
        ax.set_aspect('equal')
        plt.title(self.name)
        plt.show()        

    def draw_scalar_field(self, draw_triangulation=False):
        print('=== Drawing scalar field ===')
        fig, ax = plt.subplots(1)

        # Environment borders
        num_pts = len(self.border.pts)
        border = [(self.border.pts[i], self.border.pts[(i+1) % num_pts]) for i in range(num_pts)]
        lc = mc.LineCollection(border, colors=np.array([(0, 0, 0, 1)]), linewidths=1)
        ax.add_collection(lc)
        
        # Environment obstacles
        for o in self.obstacles:
            x = [pt[0] for pt in o.pts]
            y = [pt[1] for pt in o.pts]
            plt.fill(x, y, 'black')

        # Compute scalar field values at each point in the triangulation
        tri_x = [v[0] for v in self.triangulated_env['vertices']]
        tri_y = [v[1] for v in self.triangulated_env['vertices']]
        tri_list = [list(t) for t in self.triangulated_env['triangles']]

        print('Triangulation has {} points.'.format(len(tri_x)))
        args = list(zip(tri_x, tri_y))
        with Pool() as pool:
            # Run jobs in parallel and returns when all are complete
            scalar_values = pool.starmap(self.scalar_field_helper, tqdm.tqdm(args, total=len(tri_x)))
        print('=== Scalar field computation complete ===')

        # Draw scalar field
        plot_triangulation = tri.Triangulation(tri_x, tri_y, tri_list)
        plt.tricontourf(plot_triangulation, scalar_values, 150)
        if draw_triangulation:
            verts = np.array(self.triangulated_env['vertices'])
            ax.triplot(verts[:, 0], verts[:, 1], self.triangulated_env['triangles'], 'k,-', linewidth=.5)
        ax.set_aspect('equal')
        # ax.margins(-2.5,)
        plt.xlim(-2.5, 0)
        plt.ylim(0, 2.5)
        plt.title(self.name + ' scalar field')
        plt.show()

    def scalar_field_helper(self, tri_x, tri_y):
        p = [tri_x, tri_y]

        if not self.point_is_legal(p):
            return 0
        else:
            vis = visibility.compute(p, self.polygons_converted_to_segs)
            return star_polygon.StarPolygon(vis, p).min_diameter

    def draw_triangulation(self):
        fig = plt.figure(constrained_layout=True)
        gs = GridSpec(1, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])

        # Environment borders
        num_pts = len(self.border.pts)
        border = [(self.border.pts[i], self.border.pts[(i+1) % num_pts]) for i in range(num_pts)]
        lc1 = mc.LineCollection(border, colors=np.array([(0, 0, 0, 1)]), linewidths=1)
        lc2 = mc.LineCollection(border, colors=np.array([(0, 0, 0, 1)]), linewidths=1)
        ax1.add_collection(lc1)
        ax2.add_collection(lc2)
        
        # Environment obstacles
        for o in self.obstacles:
            x = [pt[0] for pt in o.pts]
            y = [pt[1] for pt in o.pts]
            # plt.fill(x, y, 'black')
            ax1.fill(x, y, 'black')
            ax2.fill(x, y, 'black')

        # Get all triangle lines and draw them 
        verts = np.array(self.triangulated_env['vertices'])
        verts = [v for v in verts if self.point_is_legal(v)]
        verts_x = [v[0] for v in verts]
        verts_y = [v[1] for v in verts]
        verts = np.array(self.triangulated_env['vertices'])

        import matplotlib.font_manager
        fonts = [i for i in matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf') if 'times' in i.lower()]

        ax1.triplot(verts[:, 0], verts[:, 1], self.triangulated_env['triangles'], 'k-', linewidth=1.25, color='green')
        ax1.set_axis_off()
        ax1.set_aspect('equal')
        ax2.scatter(verts_x, verts_y, c='green', s=8)
        ax2.set_axis_off()
        ax2.set_aspect('equal')
        # ax2.set_title('Environment Sampled Points', fontsize=20, fontname='Times New Roman')
        # plt.title(self.name + ' triangulation')
        plt.show()
        save_path = os.path.join(pathlib.Path(__file__).parent.absolute(), 'paper materials', 'triangulation.pdf')
        # fig.savefig(save_path, dpi=10000)

def compare_envs(env1, env2, with_rotations=True, title_data=''):
    """For each evenly spaced position (i.e. triangulation vertex) in env1, find the 
    best matching position in env2 (from random samples of the environment).
    In practice, env1 is the virtual environment and env2 is the physical environment.
    """
    import time
    start_time = time.time()
    if with_rotations:
        title_data += ' | WITH ROTATIONS'

    print('Comparing {} and {}...'.format(env1.name, env2.name))
    matches = [] # Format is [(match value, env1 position, env2 position), ...]

    env1_positions_x = [v[0] for v in env1.triangulated_env['vertices']]
    env1_positions_y = [v[1] for v in env1.triangulated_env['vertices']]
    env2_polys = env2.triangulation_visibility_polygons

    env2_polys_rotated = []
    num_rotations = 10
    rotation_amount = geometry.TWO_PI / num_rotations
    for i in range(num_rotations):
        rotated_by_i = [env2_polygon.rotate(i * rotation_amount) for env2_polygon in env2_polys]
        env2_polys_rotated.extend(rotated_by_i)

    for i in tqdm(range(len(env1_positions_x))):
        p = [env1_positions_x[i], env1_positions_y[i]]
        if not env1.point_is_legal(p):
            pass
        else:
            env1_vis_poly = env1.get_triangulation_vis_poly(np.array(p))
            best_match_val = float('inf')
            best_match_tuple = None
            if with_rotations:
                temp_matches = [(env1_vis_poly.minus(rotated_env2_poly), env1_vis_poly.kernel, rotated_env2_poly.kernel) for rotated_env2_poly in env2_polys_rotated]
                temp_matches.sort(key=lambda x: x[0])
                if temp_matches[0][0] < best_match_val:
                    best_match_val = temp_matches[0][0]
                    best_match_tuple = temp_matches[0]
                assert best_match_tuple != None
                matches.append(best_match_tuple)
            else:
                temp_matches = [(env1_vis_poly.minus(env2_vis_poly), env1_vis_poly.kernel, env2_vis_poly.kernel) for env2_vis_poly in env2_polys]
                temp_matches.sort(key=lambda x: x[0])
                matches.append(temp_matches[0])

    end_time = time.time()
    elapsed_time = end_time - start_time
    match_vals = [x[0] for x in matches]
    matches_mean = statistics.mean(match_vals)
    matches_st_dev = statistics.stdev(match_vals)
    title_data += ' | Mean score: {} | Std. Dev.: {} | Num CDT sampled points virtual: {} | Num CDT sampled points virtual: {} | Elapsed time: {}'.format(matches_mean, matches_st_dev, len(env1.triangulation_visibility_polygons), len(env2.triangulation_visibility_polygons), elapsed_time)

    draw_bokeh_plots(matches, env1, env2, title_data)

def compute_phi(env1, env2, p1, p2):
    '''Area of visibility polygon at p1 in env1 that does not overlap with the visibility polygon
    at p2 in env2.
    '''
    vis_poly1 = env1.build_vis_poly_helper(p1[0], p1[1])
    vis_poly2 = env2.build_vis_poly_helper(p2[0], p2[1])

    smallest_overlap = float('inf')

    num_rotations = 10
    rotation_amount = geometry.TWO_PI / num_rotations
    for i in range(num_rotations):
        rotated_vis_poly2 = vis_poly2.rotate(i * rotation_amount)
        diff = vis_poly1.minus(rotated_vis_poly2)
        if diff < smallest_overlap:
            smallest_overlap = diff

    return diff

def draw_bokeh_plots(matches, env1, env2, title_data):
    from bokeh.io import curdoc, show
    from bokeh.layouts import row
    from bokeh.models import CustomJS, ColumnDataSource, LinearColorMapper, Grid, LinearAxis, MultiPolygons, Plot, ColorBar, LogColorMapper
    from bokeh.plotting import figure, output_file, save
    from bokeh.models.widgets import DataTable, DateFormatter, TableColumn
    from bokeh.models import HoverTool
    import matplotlib as mpl
    from bokeh.layouts import gridplot
    from bokeh.models import Range1d

    current_time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S-")
    # Correspondence plot
    # All credit goes to https://stackoverflow.com/questions/34164587/get-selected-data-contained-within-box-select-tool-in-bokeh
    # Data
    env1_pts_x = []
    env1_pts_y = []
    env2_pts_x = []
    env2_pts_y = []
    pts_vals = []
    for match in matches:
        env1_pos = match[1]
        if env1.point_is_legal(env1_pos):
            env2_pos = match[2]
            env1_pts_x.append(env1_pos[0])
            env1_pts_y.append(env1_pos[1])
            env2_pts_x.append(env2_pos[0])
            env2_pts_y.append(env2_pos[1])
            pts_vals.append(match[0])
    pts_vals_normalized = np.array([val / max(pts_vals)+1e-10 for val in pts_vals])
    colors = [
        '#%02x%02x%02x' % (int(r), int(g), int(b)) for r, g, b, _ in 255*mpl.cm.viridis(mpl.colors.Normalize()(pts_vals_normalized))
    ]
    max_extent = max(abs(env1.min_x), abs(env1.max_x), abs(env1.min_y), abs(env1.max_y)) + 5
    # Environment 1
    s1 = ColumnDataSource(data=dict(x=env1_pts_x, y=env1_pts_y, color=colors))
    correspondence_env1_plot = figure(match_aspect=True, tools=['lasso_select', 'wheel_zoom', 'reset', 'save'], title=env1.name)
    correspondence_env1_plot.patch([p[0] for p in env1.border.pts], [p[1] for p in env1.border.pts], alpha=1, line_color='black', fill_color='white', line_width=2)
    for obs in env1.obstacles:
        correspondence_env1_plot.patch([p[0] for p in obs.pts], [p[1] for p in obs.pts], alpha=1, line_color='black', fill_color='black', line_width=2)
    correspondence_env1_plot.xgrid.grid_line_color = None
    correspondence_env1_plot.ygrid.grid_line_color = None
    correspondence_env1_plot.xaxis.major_tick_line_color = None
    correspondence_env1_plot.xaxis.minor_tick_line_color = None
    correspondence_env1_plot.yaxis.major_tick_line_color = None
    correspondence_env1_plot.yaxis.minor_tick_line_color = None
    correspondence_env1_plot.x_range = Range1d(-max_extent, max_extent)
    correspondence_env1_plot.y_range = Range1d(-max_extent, max_extent)
    print("MAX VALUE IN COLOR BAR: " + str(max(pts_vals)))
    color_mapper = LinearColorMapper(palette="Viridis256", low=0, high=max(pts_vals))
    color_bar = ColorBar(color_mapper=color_mapper, label_standoff=12)
    correspondence_env1_plot.add_layout(color_bar, 'left')
    correspondence_env1_plot.circle('x', 'y', source=s1, radius=0.1, fill_color='color', line_color='color')

    # Environment 2
    s2 = ColumnDataSource(data=dict(x=env2_pts_x, y=env2_pts_y))
    s3 = ColumnDataSource(data=dict(x=[], y=[]))
    correspondence_env2_plot = figure(match_aspect=True, tools=['wheel_zoom', 'box_zoom', 'reset', 'save'], title=env2.name)
    correspondence_env2_plot.toolbar.active_drag = None
    correspondence_env2_plot.toolbar.active_scroll = "auto"
    correspondence_env2_plot.patch([p[0] for p in env2.border.pts], [p[1] for p in env2.border.pts], alpha=1, line_color='black', fill_color='white', line_width=2)
    for obs in env2.obstacles:
        correspondence_env2_plot.patch([p[0] for p in obs.pts], [p[1] for p in obs.pts], alpha=1, line_color='black', fill_color='black', line_width=2)
    correspondence_env2_plot.xgrid.grid_line_color = None
    correspondence_env2_plot.ygrid.grid_line_color = None
    correspondence_env2_plot.xaxis.major_tick_line_color = None
    correspondence_env2_plot.xaxis.minor_tick_line_color = None
    correspondence_env2_plot.yaxis.major_tick_line_color = None
    correspondence_env2_plot.yaxis.minor_tick_line_color = None
    correspondence_env2_plot.x_range = Range1d(-max_extent, max_extent)
    correspondence_env2_plot.y_range = Range1d(-max_extent, max_extent)
    correspondence_env2_plot.circle("x", "y", source=s3, radius=0.1, color="firebrick")
    # fancy javascript to link subplots
    # js pushes selected points into ColumnDataSource of 2nd plot
    # inspiration for this from a few sources:
    # credit: https://stackoverflow.com/users/1097752/iolsmit via: https://stackoverflow.com/questions/48982260/bokeh-lasso-select-to-table-update
    # credit: https://stackoverflow.com/users/8412027/joris via: https://stackoverflow.com/questions/34164587/get-selected-data-contained-within-box-select-tool-in-bokeh
    s1.selected.js_on_change(
        "indices",
        CustomJS(
            args=dict(s1=s1, s2=s2, s3=s3),
            code="""
            var inds = cb_obj.indices;
            var d1 = s1.data;
            var d2 = s2.data;
            var d3 = s3.data;
            d3['x'] = []
            d3['y'] = []
            for (var i = 0; i < inds.length; i++) {
                d3['x'].push(d2['x'][inds[i]])
                d3['y'].push(d2['y'][inds[i]])
            }
            s3.change.emit();
        """,
        ),
    )

    # Histogram
    match_values = [val[0] for val in matches]
    env1_positions = [val[1] for val in matches]
    env2_positions = [val[2] for val in matches]
    match_values_normalized = [val / max(match_values) for val in match_values]
    hist, edges = np.histogram(match_values, density=False, bins=50)
    hist_link_dataset = {}
    for i in range(len(edges)-1):
        bucket_lower = edges[i]
        bucket_upper = edges[i+1]
        if i < len(edges)-2:
            vals = [match for match in matches if match[0] >= bucket_lower and match[0] < bucket_upper]
        else:
            vals = [match for match in matches if match[0] >= bucket_lower and match[0] <= bucket_upper]
        formatted_vals = [[[v[1][0], v[1][1]], [v[2][0], v[2][1]]] for v in vals]
        hist_link_dataset[i] = formatted_vals

    hist_data = {}
    hist_data['y'] = hist
    hist_data['left'] = edges[:-1]
    hist_data['right'] = edges[1:]

    # hist_data['matches'] = hist_link_dataset
    histogram_plot = figure(title='Histogram of Inaccessible Virtual Space', x_axis_label='Area of inaccessible virtual space', y_axis_label='Count')
    # Quad glyphs to create a histogram
    hist_data = ColumnDataSource(hist_data)
    hist_rect = histogram_plot.quad(source=hist_data, bottom=0, top='y', left='left', right='right', color='#2ca25f', fill_alpha=0.7, hover_fill_color='#18703e', hover_fill_alpha=1.0, line_color='black')
    
    connection_circles_env1_data = ColumnDataSource(data=dict(x=[], y=[]))
    connection_circles_env1 = correspondence_env1_plot.circle('x', 'y', source=connection_circles_env1_data, radius=0.1, line_color="red", fill_color='#ff8400')

    connection_circles_env2_data = ColumnDataSource(data=dict(x=[], y=[]))
    connection_circles_env2 = correspondence_env2_plot.circle('x', 'y', source=connection_circles_env2_data, radius=0.1, line_color='blue', fill_color='#ff8400')
    hover_code = """
    const link_data = %s  
    const my_index = cb_data.index.indices;

    var d1 = connection_circles_env1_data.data;
    var d2 = connection_circles_env2_data.data;
    if (my_index.length > 0){
        var index = my_index[0];
        var key_list = Object.keys(link_data)
        var data_to_plot = link_data[index];

        d1['x'] = []
        d1['y'] = []
        d2['x'] = []
        d2['y'] = []
        for (var i = 0; i < data_to_plot.length; i++) {
            var env1_pos = data_to_plot[i][0];
            var env2_pos = data_to_plot[i][1];
            d1['x'].push(env1_pos[0])
            d1['y'].push(env1_pos[1])
            d2['x'].push(env2_pos[0])
            d2['y'].push(env2_pos[1])
        }
    }
    else{
        d1['x'] = []
        d1['y'] = []
        d2['x'] = []
        d2['y'] = []
    }
    connection_circles_env1_data.change.emit();
    connection_circles_env2_data.change.emit();
    """ % hist_link_dataset
    callback = CustomJS(args={'hist_rect': hist_rect.data_source, 'connection_circles_env1_data': connection_circles_env1_data, 'connection_circles_env2_data': connection_circles_env2_data}, code=hover_code)
    # Hover tool with vline mode
    hover = HoverTool(callback=callback,
                      mode='vline',
                      tooltips=[('Count', '@y')])
    histogram_plot.add_tools(hover)

    filename = '{}_and_{}_interaction_{}.html'.format(env1.name, env2.name, current_time)
    if not os.path.exists('img'):
        os.mkdir('img')
    output_file(os.path.join('img', filename))

    from bokeh.models import Div
    div = Div(text=title_data, width=200, height=100)
    # save(row(correspondence_env1_plot, correspondence_env2_plot, histogram_plot), row(div))
    save(gridplot([correspondence_env1_plot, correspondence_env2_plot, histogram_plot, div], ncols=3))

    print('Finished bokeh plot generation!')

if __name__ == '__main__':
    random.seed(13443)
    cur_path = os.path.join(pathlib.Path(__file__).parent.absolute(), 'envs')
    
    virt_grand_demo = Environment(os.path.join(cur_path, 'virt', 'big_square.xml'))
    phys_grand_demo = Environment(os.path.join(cur_path, 'phys', 'big_square.xml'))
    compare_envs(virt_grand_demo, phys_grand_demo, True, 'Best phys for each virt pos.\nArea of virt. nonoverlap')
    
    print('Finished ok')