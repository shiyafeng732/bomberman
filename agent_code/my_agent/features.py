"""
Feature extraction for the Bomberman agent.

The whole design goal here is: turn the raw game_state dictionary into a SHORT
binary vector that already contains the information a player needs, so that even
a linear model can act on it.

Layout of the feature vector (FEATURE_DIM = 26).
Directions are always ordered UP, RIGHT, DOWN, LEFT (same order as ACTIONS).

    0 -  3  move_ok[d]        1 if stepping in direction d is possible and not instant death
    4 -  7  coin_dir[d]       one-hot: first step of the shortest path to the nearest coin
    8 - 11  crate_dir[d]      one-hot: first step to the nearest tile from which we can bomb a crate
   12 - 15  escape_dir[d]     one-hot: first step to the nearest safe tile (only set when in danger)
   16 - 19  danger_dir[d]     1 if the neighbouring tile is inside a blast that goes off very soon
   20       in_danger         1 if our own tile is inside the blast radius of some bomb
   21       bomb_available    1 if we are allowed to drop a bomb
   22       bomb_hits_crate   1 if a bomb dropped here would destroy at least one crate
   23       bomb_is_safe      1 if we could still escape a bomb dropped here
   24       bomb_hits_enemy   1 if a bomb dropped here would reach an opponent
   25       bias              always 1
"""

from collections import deque

import numpy as np

import settings as s

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# (dx, dy) for UP, RIGHT, DOWN, LEFT -- note that y grows downwards (image coordinates)
DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]

FEATURE_DIM = 26

SAFE = 99  # "danger" value of a tile that no bomb can reach


def blast_coords(field, pos, power=None):
    """All tiles hit by a bomb at `pos`. The blast is stopped by stone walls only."""
    if power is None:
        power = s.BOMB_POWER
    x, y = pos
    coords = [(x, y)]
    for dx, dy in DIRECTIONS:
        for i in range(1, power + 1):
            nx, ny = x + i * dx, y + i * dy
            if field[nx, ny] == -1:      # stone wall stops the explosion
                break
            coords.append((nx, ny))
    return coords


def build_danger_map(field, bombs, explosion_map):
    """For every tile: in how many steps it becomes deadly (0 = deadly right now, SAFE = never)."""
    danger = np.full(field.shape, SAFE, dtype=int)
    danger[explosion_map > 0] = 0        # a fire that is burning right now
    for pos, timer in bombs:
        for (x, y) in blast_coords(field, pos):
            danger[x, y] = min(danger[x, y], timer)
    return danger


def build_free_map(field, bombs, others):
    """Boolean map of tiles an agent can walk on (ignoring explosions)."""
    free = field == 0
    for pos, _ in bombs:
        free[pos] = False
    for pos in others:
        free[pos] = False
    return free


def bfs_first_step(free, start, targets, danger=None):
    """Breadth-first search from `start` to the closest tile in `targets`.

    Returns (direction_index, distance). direction_index is the index into
    DIRECTIONS of the first step of the shortest path, or -1 if no target can be
    reached. If `danger` is given, we only walk over tiles that are still safe at
    the time we would arrive there (used for escaping bombs).
    """
    targets = set(targets)
    if not targets:
        return -1, -1
    if start in targets:
        return -1, 0

    visited = {start}
    queue = deque()
    for i, (dx, dy) in enumerate(DIRECTIONS):
        queue.append(((start[0] + dx, start[1] + dy), i, 1))

    while queue:
        pos, first, dist = queue.popleft()
        if pos in visited or not free[pos]:
            continue
        if danger is not None and danger[pos] <= dist:
            continue                      # this tile explodes before/when we arrive
        visited.add(pos)
        if pos in targets:
            return first, dist
        for dx, dy in DIRECTIONS:
            queue.append(((pos[0] + dx, pos[1] + dy), first, dist + 1))
    return -1, -1


def crate_targets(field, free):
    """Free tiles that have at least one crate as a neighbour (good places to bomb)."""
    targets = []
    xs, ys = np.where(field == 1)
    for cx, cy in zip(xs, ys):
        for dx, dy in DIRECTIONS:
            nx, ny = cx + dx, cy + dy
            if free[nx, ny]:
                targets.append((nx, ny))
    return targets


def safe_targets(free, danger):
    """All walkable tiles that no bomb can reach."""
    xs, ys = np.where((danger == SAFE) & free)
    return list(zip(xs.tolist(), ys.tolist()))


def can_escape_after_bomb(field, free, danger, pos):
    """Would we survive if we dropped a bomb right here?"""
    virtual = danger.copy()
    for (x, y) in blast_coords(field, pos):
        virtual[x, y] = min(virtual[x, y], s.BOMB_TIMER)
    step, _ = bfs_first_step(free, pos, safe_targets(free, virtual), danger=virtual)
    return step != -1


def parse_state(game_state):
    """Pull the pieces we need out of the game state dictionary."""
    field = game_state['field']
    _, _, bombs_left, (x, y) = game_state['self']
    bombs = game_state['bombs']
    others = [xy for (_, _, _, xy) in game_state['others']]
    coins = game_state['coins']

    danger = build_danger_map(field, bombs, game_state['explosion_map'])
    free = build_free_map(field, bombs, others)
    return field, (x, y), bombs_left, coins, others, danger, free


def coin_distance(game_state):
    """Length of the shortest path to the nearest coin (-1 if unreachable). Used for rewards."""
    if game_state is None:
        return -1
    field, pos, _, coins, others, danger, free = parse_state(game_state)
    return bfs_first_step(free, pos, coins)[1]


def crate_distance(game_state):
    """Length of the shortest path to the nearest bombing spot (-1 if none). Used for rewards."""
    if game_state is None:
        return -1
    field, pos, _, _, _, danger, free = parse_state(game_state)
    return bfs_first_step(free, pos, crate_targets(field, free))[1]


def state_to_features(game_state: dict) -> np.ndarray:
    """Convert the game state into the feature vector described at the top of this file."""
    if game_state is None:
        return None

    field, pos, bombs_left, coins, others, danger, free = parse_state(game_state)
    x, y = pos
    f = np.zeros(FEATURE_DIM, dtype=np.float32)

    # 0-3 / 16-19: what does it look like around us?
    for i, (dx, dy) in enumerate(DIRECTIONS):
        n = (x + dx, y + dy)
        f[i] = 1.0 if (free[n] and danger[n] > 0) else 0.0
        f[16 + i] = 1.0 if danger[n] <= 1 else 0.0

    # 4-7: where is the nearest coin?
    step, _ = bfs_first_step(free, pos, coins)
    if step >= 0:
        f[4 + step] = 1.0

    # 8-11: where is the nearest crate we could blow up?
    step, _ = bfs_first_step(free, pos, crate_targets(field, free))
    if step >= 0:
        f[8 + step] = 1.0

    # 12-15: if we are in danger, where do we run?
    in_danger = danger[x, y] < SAFE
    if in_danger:
        step, _ = bfs_first_step(free, pos, safe_targets(free, danger), danger=danger)
        if step >= 0:
            f[12 + step] = 1.0

    # 20-25: scalar situation flags
    f[20] = 1.0 if in_danger else 0.0
    f[21] = 1.0 if bombs_left else 0.0
    blast = blast_coords(field, pos)
    f[22] = sum(field[c] == 1 for c in blast) / 4.0
    f[23] = 1.0 if can_escape_after_bomb(field, free, danger, pos) else 0.0
    f[24] = 1.0 if any(o in blast for o in others) else 0.0
    f[25] = 1.0

    return f
