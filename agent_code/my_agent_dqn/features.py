"""
Feature extraction for the Bomberman agent

The design goal is a short binary vector that already contains the information a player needs, so that even a linear model can act on it

Layout of the feature vector (FEATURE_DIM = 26)
Directions are ordered UP, RIGHT, DOWN, LEFT, same as ACTIONS

0-3      move_ok[d]        1 if we can walk into neighbour d without dying right away
4-7      coin_dir[d]       one-hot, first step on the path to the nearest coin
8-11     crate_dir[d]      one-hot, first step to a tile where we can bomb a crate
12-15    escape_dir[d]     one-hot, first step towards safety, only when in danger
16-19    danger_dir[d]     1 if the neighbouring tile is inside a blast that goes off very soon
20       in_danger         1 if our own tile is inside the blast radius of some bomb
21       bomb_available    1 if we are allowed to drop a bomb
22       bomb_hits_crate   crates a bomb dropped here would hit, divided by 4
23       bomb_is_safe      1 if we could still escape a bomb dropped here
24       bomb_hits_enemy   1 if a bomb dropped here would reach an opponent
25       bias              always 1
"""




from collections import deque
import numpy as np
import settings as s





ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
FEATURE_DIM = 26
SAFE = 99





def blast_coords(field, pos, power=None):
    if power is None:
        power = s.BOMB_POWER
    x, y = pos
    coords = [(x, y)]
    for dx, dy in DIRECTIONS:
        for i in range(1, power + 1):
            nx, ny = x + i * dx, y + i * dy
            if field[nx, ny] == -1: # stone wall stops the explosion
                break
            coords.append((nx, ny))
    return coords


def build_danger_map(field, bombs, explosion_map):
    # for every tile: in how many steps it becomes deadly (0 = deadly now, SAFE = never)
    danger = np.full(field.shape, SAFE, dtype=int)
    danger[explosion_map > 0] = 0        # a fire that is burning right now
    for pos, timer in bombs:
        for (x, y) in blast_coords(field, pos):
            danger[x, y] = min(danger[x, y], timer)
    return danger


def build_free_map(field, bombs, others):
    # boolean map of tiles an agent can walk on (explosions are ignored here)
    free = field == 0
    for pos, _ in bombs:
        free[pos] = False
    for pos in others:
        free[pos] = False
    return free


def bfs_first_step(free, start, targets, danger=None):
    # BFS to the closest tile in targets, returns (direction index, distance), -1 if unreachable
    # With danger we only use tiles that are still safe when we arrive, needed for escaping bombs
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
        if danger is not None and danger[pos] < dist:
            continue # this tile explodes before/when we arrive
        visited.add(pos)
        if pos in targets:
            return first, dist
        for dx, dy in DIRECTIONS:
            queue.append(((pos[0] + dx, pos[1] + dy), first, dist + 1))
    return -1, -1


def crate_targets(field, free):
    targets = []
    xs, ys = np.where(field == 1)
    for cx, cy in zip(xs, ys):
        for dx, dy in DIRECTIONS:
            nx, ny = cx + dx, cy + dy
            if free[nx, ny]:
                targets.append((nx, ny))
    return targets


def safe_targets(free, danger):
    xs, ys = np.where((danger == SAFE) & free)
    return list(zip(xs.tolist(), ys.tolist()))


def can_escape_after_bomb(field, free, danger, pos):
    virtual = danger.copy()
    for (x, y) in blast_coords(field, pos):
        virtual[x, y] = min(virtual[x, y], s.BOMB_TIMER)
    step, _ = bfs_first_step(free, pos, safe_targets(free, virtual), danger=virtual)
    return step != -1


def parse_state(game_state):
    field = game_state['field']
    _, _, bombs_left, (x, y) = game_state['self']
    bombs = game_state['bombs']
    others = [xy for (_, _, _, xy) in game_state['others']]
    coins = game_state['coins']

    danger = build_danger_map(field, bombs, game_state['explosion_map'])
    free = build_free_map(field, bombs, others)
    return field, (x, y), bombs_left, coins, others, danger, free


def coin_distance(game_state):
    # length of the shortest path to the nearest coin (-1 if unreachable), used for rewards
    if game_state is None:
        return -1
    field, pos, _, coins, others, danger, free = parse_state(game_state)
    return bfs_first_step(free, pos, coins)[1]


def crate_distance(game_state):
    # length of the shortest path to the nearest bombing spot (-1 if none), used for rewards
    if game_state is None:
        return -1
    field, pos, _, _, _, danger, free = parse_state(game_state)
    return bfs_first_step(free, pos, crate_targets(field, free))[1]




def state_to_features(game_state: dict) -> np.ndarray:
    if game_state is None:
        return None

    field, pos, bombs_left, coins, others, danger, free = parse_state(game_state)
    x, y = pos
    f = np.zeros(FEATURE_DIM, dtype=np.float32)

    for i, (dx, dy) in enumerate(DIRECTIONS):
        n = (x + dx, y + dy)
        f[i] = 1.0 if (free[n] and danger[n] > 0) else 0.0
        f[16 + i] = 1.0 if danger[n] <= 1 else 0.0

    step, _ = bfs_first_step(free, pos, coins)
    if step >= 0:
        f[4 + step] = 1.0

    step, _ = bfs_first_step(free, pos, crate_targets(field, free))
    if step >= 0:
        f[8 + step] = 1.0

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