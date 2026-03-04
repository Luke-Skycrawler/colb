import warp as wp 
from viewer import PSViewer
import polyscope as ps
import numpy as np 
thickness = 0.0475
contact_volume = 2048
from quat_util import vec3, vec4, mat33, mat44, scalar
from geometry import Soup

'''
TODO: make sure max_unroll = 0 before importing this module

(see `fix_interference` kernel)
'''

@wp.kernel 
def edge_aabb(x: wp.array(dtype = vec3), edges: wp.array(dtype = int), aabb_lower: wp.array(dtype = wp.vec3), aabb_upper: wp.array(dtype = wp.vec3), thickness: scalar):
    i = wp.tid()
    p0 = x[edges[i * 2]]
    p1 = x[edges[i * 2 + 1]]

    aabb_lower[i] = wp.vec3(wp.min(p0, p1) - vec3(thickness))
    aabb_upper[i] = wp.vec3(wp.max(p0, p1) + vec3(thickness))
    
@wp.kernel
def c_gets_i_mod_2(color: wp.array(dtype = int)):
    i = wp.tid() 
    color[i] = i % 2

# @wp.struct 
# class ContactInfo:
#     a1a2b1b2: wp.vec4i
#     lam: scalar
#     k: scalar
#     cj: scalar

@wp.struct 
class XConstraint: 
    a1a2b1b2: wp.vec4i
    l0: scalar
    alpha: scalar
    lam: scalar

ContactInfo = XConstraint
# @wp.struct
# class HTableEntry: 
#     list_idx: int
#     # updated_stamp: int

@wp.struct 
class Contacts:
    list: wp.array(dtype = ContactInfo)
    cnt: wp.array(dtype = int)
    htable: wp.array(dtype = int)

@wp.func
def _hash(a1: int, b1: int) -> int:
    '''
    Hash function in "Optimized Spatial Hashing for Collision Detection of Deformable Objects"
    '''
    h = wp.bit_xor(a1 * 73856093, b1 * 19349663)
    return h % 8191

@wp.func
def append(contacts: Contacts, a1: int, a2: int, b1: int, b2: int):
    idx = wp.atomic_add(contacts.cnt, 0, 1)
    h = _hash(a1, b1)
    idx = idx % contact_volume
    contacts.list[idx].a1a2b1b2 = wp.vec4i(a1, a2, b1, b2)
    contacts.list[idx].l0 = scalar(thickness * 2.0)
    contacts.list[idx].alpha = scalar(1e-6)
    contacts.htable[h] = idx

@wp.kernel
def edge_edge_collision(bvh: wp.uint64, x: wp.array(dtype = vec3), edges: wp.array(dtype = int), contacts: Contacts, thickness: float):
    i = wp.tid()
    if True:
        # edge exists
        a1 = edges[i * 2]
        a2 = edges[i * 2 + 1]
        p1 = wp.vec3(x[a1])
        p2 = wp.vec3(x[a2])

        low = wp.min(p1, p2) - wp.vec3(thickness)
        high = wp.max(p1, p2) + wp.vec3(thickness)
        query = wp.bvh_query_aabb(bvh, low, high)
        j = int(0) 
        while wp.bvh_query_next(query, j):
            connected = edges[i * 2] == edges[j * 2] or edges[i * 2] == edges[j * 2 + 1] or edges[i * 2 + 1] == edges[j * 2] or edges[i * 2 + 1] == edges[j * 2 + 1]
            
            if i < j and not connected: 
                b1 = edges[j * 2]
                b2 = edges[j * 2 + 1]
                
                q1 = wp.vec3(x[b1])
                q2 = wp.vec3(x[b2])
                std = wp.closest_point_edge_edge(p1, p2, q1, q2, 1e-6)
                dist = std[2]
                if dist < thickness * 2.5:
                    append(contacts, a1, a2, b1, b2)

@wp.func 
def fix_interference(v: wp.vec4i, color: wp.array(dtype = int), dirty: wp.array(dtype = bool)):
    colors = wp.vec4i(color[v.x], color[v.y], color[v.z], color[v.w])
    cm = wp.max(colors)

    for ii in range(1, 4):
        for jj in range(ii):
            if color[v[jj]] == color[v[ii]]:
                color[v[ii]] = cm + 1
                cm += 1
                dirty[v[ii]] = True

@wp.kernel
def color_contacts(contacts: Contacts, color: wp.array(dtype = int), dirty: wp.array(dtype = bool)):
    i = wp.tid() 
    if i < contacts.cnt[0]:
        fix_interference(contacts.list[i].a1a2b1b2, color, dirty)

class ContactSolverBase:
    def __init__(self):
        '''
        Base contact interface to detect the edge-edge contacts. Contains an optional colorization function

        need to have self.soup: Soup defined prior to calling this constructor
        '''
        self.soup: Soup

        n_edges = self.soup.edges.shape[0] // 2
        n_nodes = self.soup.xcs.shape[0]
        
        # bvh
        self.bvh_edges_lower = wp.zeros((n_edges, ), dtype = wp.vec3) 
        self.bvh_edges_upper = wp.zeros((n_edges, ), dtype = wp.vec3)
        self.compute_edge_aabbs()
        self.bvh_edges = wp.Bvh(self.bvh_edges_lower, self.bvh_edges_upper)
        
        # color 
        self.color = wp.zeros((n_nodes, ), dtype = int)
        # self.color_cnt = wp.zeros((1,), dtype = int)
        self.dirty_bit = wp.zeros((n_nodes, ), dtype = bool)


        # contacts 
        self.contacts_list = wp.zeros((contact_volume,), dtype = ContactInfo)
        self.contacts_cnt = wp.zeros((1,), dtype = int)
        self.contacts_htable = wp.zeros((8191,), dtype = int)
        self.contacts = Contacts()

        self.contacts_list_new = wp.zeros((contact_volume,), dtype = ContactInfo)
        self.contacts_cnt_new = wp.zeros((1,), dtype = int)
        self.contacts_htable_new = wp.zeros((8191,), dtype = int)
        self.contacts_new = Contacts()

        self.contacts.list = self.contacts_list
        self.contacts.cnt = self.contacts_cnt
        self.contacts.htable = self.contacts_htable

        self.contacts_new.list = self.contacts_list_new
        self.contacts_new.cnt = self.contacts_cnt_new
        self.contacts_new.htable = self.contacts_htable_new

        self.n_contacts = 0



    def update_bvh(self):
        self.compute_edge_aabbs()
        self.bvh_edges.refit()
    
    def compute_edge_aabbs(self):
        n_edges = self.soup.edges.shape[0] // 2
        wp.launch(edge_aabb, n_edges, inputs = [self.soup.x_transformed, self.soup.edges, self.bvh_edges_lower, self.bvh_edges_upper, thickness * 2.0])

    def colorization(self):
        n_nodes = self.soup.xcs.shape[0]
        wp.launch(c_gets_i_mod_2, n_nodes, inputs = [self.color])
        dirty = True 
        while (dirty):
            self.dirty_bit.zero_()
            wp.launch(color_contacts, (self.n_contacts, ), inputs = [self.contacts, self.color, self.dirty_bit])
            dirty = self.dirty_bit.numpy().any()
            # if dirty:
            #     print("dirty! recoloring...")
        self.color_cnt = np.max(self.color.numpy()) + 1
        # print(f"color cnt = {self.color_cnt}")
        # colornp = self.color.numpy()
        # print(f"color in contact pairs (19, 20, 21, 59, 60, 61): {colornp[19]}, {colornp[20]}, {colornp[21]}, {colornp[59]}, {colornp[60]}, {colornp[61]}")
    
    def compute_V(self, ret = True): 
        return None
        
    def detect_collision(self): 
        self.compute_V(ret = False)
        self.update_bvh()
        self.contacts_new.cnt.zero_()
        self.contacts_new.htable.fill_(-1)
        n_edges = self.soup.edges.shape[0] // 2
        
        wp.launch(edge_edge_collision, n_edges, inputs = [self.bvh_edges.id, self.soup.x_transformed, self.soup.edges, self.contacts_new, thickness])
        wp.synchronize()
        self.n_contacts = self.contacts_new.cnt.numpy()[0]
        # print(f"n contacts = {self.n_contacts}")
        # print(self.contacts.list.numpy()["a1a2b1b2"][:self.n_contacts])
