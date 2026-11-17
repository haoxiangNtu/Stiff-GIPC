import warp as wp

HASH_SIZE = 1 << 20
HASH_MASK = HASH_SIZE - 1
EMPTY = -1
PENDING = -2

@wp.func
def hash_find_block(
    key: int,
    hash_keys: wp.array(dtype=int),
    hash_vals: wp.array(dtype=int),
) -> int:
    h = key & HASH_MASK
    step = step_hash(key)

    # for _ in range(64):
    while True:
        k = hash_keys[h]
        if k == key:
            v = hash_vals[h]
            # if you use a "pending" marker (see below), handle it:
            if v >= 0: return v
            # v < 0 means not ready or invalid; keep probing / retry
            # but fortunately we will not use this function before initialization is done
        elif k == EMPTY:
            return -1

        h = (h + step) & HASH_MASK

    return -1

@wp.func
def hash_find_or_insert_block(
    key: int,
    block_xyz: wp.vec3i,
    hash_keys: wp.array(dtype=int),
    hash_vals: wp.array(dtype=int),        # stores block_id (>=0), or PENDING, or -1
    # block_key_by_id: wp.array(dtype=int), # stores key per block_id
    block_xyz_by_id: wp.array(dtype=wp.vec3i), # stores xyz per block_id
    block_count: wp.array(dtype=int),      # length-1 counter
    max_blocks: int,
) -> int:

    # 1. Compute the initial hash position (simple linear probing).
    # Note: negative keys need to be handled so the hash stays positive.
    h = key & HASH_MASK
    step = step_hash(key)
    
    # for _ in range(64):
    while True:
        # 1. Try to atomically claim the slot.
        # Use CAS directly: -1 means this thread claimed it; key means another thread did.
        old_key = wp.atomic_cas(hash_keys, h, EMPTY, key)

        if old_key == EMPTY:
            # --- This thread won and is responsible for initializing the block. ---
            # Mark it as PENDING first so other threads do not read the initial -1.
            hash_vals[h] = PENDING

            # Allocate an ID.
            new_id = wp.atomic_add(block_count, 0, 1)
            
            if new_id >= 0 and new_id < max_blocks:
                # Fill metadata.
                # block_key_by_id[new_id] = key
                block_xyz_by_id[new_id] = block_xyz  # Assuming you want to store the coordinates here
                
            # Finally release PENDING and write the real ID.
            hash_vals[h] = new_id
            return new_id

        elif old_key == key:
            # --- Another thread already claimed the key slot. ---
            # Wait until the corresponding hash_vals entry is no longer -1
            # (initial state) and no longer PENDING.
            # Once it is >= 0, the ID has been allocated and block_keys_by_id is ready.

            while True:
                val = hash_vals[h]
                if val >= 0:
                    return val
                # Spin here until the winning thread finishes assigning the value.
        
        # Linear probing: move to the next slot after a collision.
        h = (h + step) & HASH_MASK
        
    return EMPTY # Should be unreachable.

@wp.func
def part1by2(x: int) -> int:
    # expand lower 10 bits so that there are 2 zero bits between each bit
    x &= 0x000003ff                 # ---- ---- ---- ---- ---- --98 7654 3210
    x = (x | (x << 16)) & 0x030000FF
    x = (x | (x << 8))  & 0x0300F00F
    x = (x | (x << 4))  & 0x030C30C3
    x = (x | (x << 2))  & 0x09249249
    return x

@wp.func
def morton3D(bx: int, by: int, bz: int) -> int:
    return ((part1by2(bx) << 2) | (part1by2(by) << 1) | part1by2(bz))

# ----------------------------
# Helper functions (hash + indexing) - improved
# ----------------------------

@wp.func
def fmix32(h: int) -> int:
    # MurmurHash3 fmix32 finalizer (excellent avalanche; good low bits)
    h &= 0xffffffff
    h ^= (h >> 16)
    h *= 0x85ebca6b
    h &= 0xffffffff
    h ^= (h >> 13)
    h *= 0xc2b2ae35
    h &= 0xffffffff
    h ^= (h >> 16)
    return h & 0xffffffff


@wp.func
def make_block_key(bx: int, by: int, bz: int) -> int:
    # return bx << 20 | by << 10 | bz
    # key is stable identifier (use morton), optionally fmix for better bit diffusion
    return fmix32(morton3D(bx, by, bz))

@wp.func
def step_hash(key: int) -> int:
    # secondary hash -> probing step; force odd for power-of-two table
    s = fmix32(key ^ 0x9e3779b9) & HASH_MASK
    return s | 1
