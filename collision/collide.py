import numpy as np
import numba as nb
from helper.constants import THREADS
from collision.gjk import gjk
from collision.epa import epa
from collision.sat import sat


@nb.njit(fastmath=True, parallel=True)
def collide(pairs, vertices, starts, lengths, pos, sr, isr, mesh_index, normals, rAs, rBs, contacts):
    num_pairs = len(pairs)
    
    if num_pairs == 0:
        return
    
    # batch operations
    batch_size = np.ceil(num_pairs / THREADS)
    
    for thread in nb.prange(THREADS):
        # create buffers for gjk
        index_a = np.empty(3, dtype='int16')
        index_b = np.empty(3, dtype='int16')
        minks = np.empty((3, 2), dtype='float32')
        
        # create buffers for epa
        support_buffer = np.empty((15, 2), dtype='float32')
        normal_buffer = np.empty((15, 2), dtype='float32')
        face_buffer = np.empty((15, 2), dtype='uint8')
        distance_buffer = np.empty(15, dtype='float32')
        set_buffer = np.empty(15, dtype='uint8')
        
        idcs_buffer = np.empty(8, dtype='uint8')
        world_buffer = np.empty((8, 2), dtype='float32')
    
        for i in range(thread * batch_size, min(num_pairs, (thread + 1) * batch_size)):
            # collect the body indices from the pairs
            a = pairs[i, 0]
            b = pairs[i, 1]
            
            # extract mesh data
            mesh_index_a = mesh_index[a]
            mesh_index_b = mesh_index[b]
            
            start_a = starts[mesh_index_a]
            start_b = starts[mesh_index_b]
            
            # collect needed data for gjk
            pos_a, pos_b = pos[a], pos[b]
            sr_a, sr_b = sr[a], sr[b]
            verts_a = vertices[start_a : start_a + lengths[mesh_index_a]]
            verts_b = vertices[start_b : start_b + lengths[mesh_index_b]]
            
            collided = gjk(
                pos_a = pos_a,
                pos_b = pos_b,
                sr_a = sr_a,
                sr_b = sr_b,
                verts_a = verts_a,
                verts_b = verts_b,
                index_a = index_a,
                index_b = index_b,
                minks = minks,
                free = 0
            )
            
            contacts[i] = 0
            
            # early termination when gjk fails
            if not collided:
                continue
            
            # TODO add body coloring for debug
            
            mtv = epa(
                pos_a = pos_a,
                pos_b = pos_b,
                sr_a = sr_a,
                sr_b = sr_b,
                verts_a = verts_a,
                verts_b = verts_b,
                simplex = minks,
                faces = face_buffer,
                sps = support_buffer,
                normals = normal_buffer,
                dists = distance_buffer,
                set = set_buffer
            )
        
            contacts[i] = 2
            
            isr_a, isr_b = isr[a], isr[b]
            rA, rB = rAs[i], rBs[i]
            
            sat(
                pos_a = pos_a,
                pos_b = pos_b,
                rs_a = sr_a,
                rs_b = sr_b,
                irs_a = isr_a,
                irs_b = isr_b,
                verts_a = verts_a,
                verts_b = verts_b,
                mtv = mtv,
                idcs = idcs_buffer,
                world = world_buffer,
                rA = rA,
                rB = rB
            )