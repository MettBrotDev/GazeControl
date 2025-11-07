"""
Maze Dataset Generator for GazeControl

Generates maze images for:
a) Image reconstruction task
b) RL classification task: Are start (top-left) and exit (bottom-right) connected?

Maze Properties:
- 6×6 grid of blocks
- Each block is 4×4 pixels
- Total image size: 24×24 pixels
- Start: top-left (green marker)
- Exit: bottom-right (red marker)
- White paths, black walls
- Binary label: 1 if connected, 0 if not connected
"""

import os
import numpy as np
import argparse
from PIL import Image
import json
from tqdm import tqdm


class MazeGenerator:
    def __init__(self, grid_size=6, block_size=4, random_endpoints: bool = False):
        """
        Args:
            grid_size: Number of blocks in each dimension (default 6×6)
            block_size: Pixels per block (default 4×4)
            random_endpoints: If True, choose start/end randomly on edges per maze
        """
        self.grid_size = grid_size
        self.block_size = block_size
        self.img_size = grid_size * block_size  # 24×24 pixels
        self.random_endpoints = bool(random_endpoints)
        
    def _random_edge_points(self):
        """Pick two distinct random edge cells (x,y) on the N×N grid."""
        N = self.grid_size
        edges = []
        for x in range(N):
            edges.append((x, 0))
            edges.append((x, N - 1))
        for y in range(N):
            edges.append((0, y))
            edges.append((N - 1, y))
        # Remove duplicates (corners were added twice)
        edges = list(dict.fromkeys(edges))
        i = np.random.randint(len(edges))
        j = np.random.randint(len(edges) - 1)
        if j >= i:
            j += 1
        return edges[i], edges[j]

    def _shortest_path_len(self, maze, start, end):
        """Return shortest path length (#steps) between start and end or None if disconnected."""
        from collections import deque
        sx, sy = start
        tx, ty = end
        if maze[sy, sx] == 0 or maze[ty, tx] == 0:
            return None
        N = self.grid_size
        visited = np.zeros_like(maze, dtype=bool)
        dist = np.full_like(maze, fill_value=-1, dtype=int)
        q = deque([(sx, sy)])
        visited[sy, sx] = True
        dist[sy, sx] = 0
        while q:
            x, y = q.popleft()
            if x == tx and y == ty:
                return int(dist[y, x])
            for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < N and 0 <= ny < N and not visited[ny, nx] and maze[ny, nx] == 1:
                    visited[ny, nx] = True
                    dist[ny, nx] = dist[y, x] + 1
                    q.append((nx, ny))
        return None

    def generate_maze(self, force_disconnected=False, start=None, end=None):
        """
        Generate a random maze with proper walls and paths.
        
        Args:
            force_disconnected: If True, ensure start and exit are NOT connected
            start: Optional (sx,sy) grid start cell (overrides random_endpoints)
            end:   Optional (tx,ty) grid end cell (overrides random_endpoints)
            
        Returns:
            maze: (grid_size, grid_size) binary array. 1=path (white), 0=wall (black)
            connected: bool, whether start and exit are connected
            start: (sx,sy) used for this maze
            end: (tx,ty) used for this maze
        """
        N = self.grid_size
        # Determine endpoints
        if start is None or end is None:
            if self.random_endpoints:
                sxsy, txty = self._random_edge_points()
                start = sxsy
                end = txty
            else:
                start = (0, 0)
                end = (N - 1, N - 1)
        sx, sy = start
        tx, ty = end

        # Helper to shuffle 4-neighborhood
        def shuffled_neighbors(x, y):
            nbrs = []
            if y > 0:
                nbrs.append((x, y - 1))
            if x < N - 1:
                nbrs.append((x + 1, y))
            if y < N - 1:
                nbrs.append((x, y + 1))
            if x > 0:
                nbrs.append((x - 1, y))
            np.random.shuffle(nbrs)
            return nbrs

        # Build a single main path from start to goal via DFS backtracking
        maze = np.zeros((N, N), dtype=np.uint8)
        visited = np.zeros((N, N), dtype=bool)

        path_stack = []
        found = False

        def dfs(x, y):
            nonlocal found
            if found:
                return True
            visited[y, x] = True
            path_stack.append((x, y))
            if x == tx and y == ty:
                found = True
                return True
            for nx, ny in shuffled_neighbors(x, y):
                if not visited[ny, nx]:
                    if dfs(nx, ny):
                        return True
            # backtrack
            path_stack.pop()
            return False

        # Attempt DFS; if it fails (shouldn't), fall back to simple rectilinear path between endpoints
        dfs(sx, sy)
        if not found:
            path_stack = []
            # Horizontal then vertical (or vice versa) from start to end
            x_dir = 1 if tx >= sx else -1
            y_dir = 1 if ty >= sy else -1
            for x in range(sx, tx + x_dir, x_dir):
                path_stack.append((x, sy))
            for y in range(sy + y_dir, ty + y_dir, y_dir):
                path_stack.append((tx, y))

        # Mark the main path cells as path
        for (x, y) in path_stack:
            maze[y, x] = 1

        # Helpers for corridor-like constraints
        def path_neighbors(x, y):
            cnt = 0
            if y > 0 and maze[y - 1, x] == 1:
                cnt += 1
            if x < N - 1 and maze[y, x + 1] == 1:
                cnt += 1
            if y < N - 1 and maze[y + 1, x] == 1:
                cnt += 1
            if x > 0 and maze[y, x - 1] == 1:
                cnt += 1
            return cnt

        def makes_2x2(x, y):
            # Would setting (x,y)=1 create any 2x2 all-ones block?
            # Check up-left squares around (x,y)
            for dy in (-1, 0):
                for dx in (-1, 0):
                    x0, y0 = x + dx, y + dy
                    if 0 <= x0 < N - 1 and 0 <= y0 < N - 1:
                        s = maze[y0, x0] + maze[y0, x0 + 1] + maze[y0 + 1, x0] + maze[y0 + 1, x0 + 1]
                        # include the candidate cell if it's within the square
                        if (dx, dy) == (0, 0):
                            s += 1  # (x,y) is bottom-right of the square
                        elif (dx, dy) == (-1, 0):
                            s += 1  # (x,y) is bottom-left
                        elif (dx, dy) == (0, -1):
                            s += 1  # (x,y) is top-right
                        else:
                            s += 1  # (x,y) is top-left
                        if s >= 4:
                            return True
            return False

        # Add corridor-like branches with directional bias and constraints
        num_branches = max(2, N)  # a bit more branches but constrained
        max_branch_len = 3 if N <= 6 else 4
        dirs = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        for _ in range(num_branches):
            bx, by = path_stack[np.random.randint(len(path_stack))]
            # choose initial direction away from dead-end if possible
            candidates = [(dx, dy) for dx, dy in dirs if 0 <= bx + dx < N and 0 <= by + dy < N and maze[by + dy, bx + dx] == 0]
            if not candidates:
                continue
            dx, dy = candidates[np.random.randint(len(candidates))]
            cx, cy = bx, by
            length = np.random.randint(1, max_branch_len + 1)
            for step in range(length):
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < N and 0 <= ny < N):
                    break
                if maze[ny, nx] == 1:
                    break
                # Degree constraints: avoid creating nodes with high degree frequently
                if path_neighbors(nx, ny) >= 2:
                    # occasionally allow a junction
                    if np.random.rand() < 0.1:
                        pass
                    else:
                        break
                # Avoid making 2x2 open rooms
                if makes_2x2(nx, ny):
                    break
                # Carve
                maze[ny, nx] = 1
                cx, cy = nx, ny
                # Directional persistence with small turn probability
                if np.random.rand() < 0.2:
                    # turn left or right
                    if dx == 0 and dy != 0:
                        dx, dy = (1, 0) if np.random.rand() < 0.5 else (-1, 0)
                    else:
                        dx, dy = (0, 1) if np.random.rand() < 0.5 else (0, -1)

        # Ensure start/end are paths
        maze[sy, sx] = 1
        maze[ty, tx] = 1

        if force_disconnected:
            # Try to sever connectivity by removing one or more cells along the shortest path
            # Compute parents from BFS to get a path; then cut an interior cell
            cut_attempts = 0
            max_cuts = 10
            while self._is_connected(maze, start, end) and cut_attempts < max_cuts:
                # BFS to reconstruct a path from start to goal
                from collections import deque
                parents = { (sx,sy): None }
                q = deque([(sx,sy)])
                seen = set([(sx,sy)])
                reached = False
                while q:
                    x, y = q.popleft()
                    if (x, y) == (tx, ty):
                        reached = True
                        break
                    for nx, ny in shuffled_neighbors(x, y):
                        if maze[ny, nx] == 1 and (nx, ny) not in seen:
                            seen.add((nx, ny))
                            parents[(nx, ny)] = (x, y)
                            q.append((nx, ny))
                if not reached:
                    break
                # Reconstruct path
                path = []
                cur = (tx, ty)
                while cur is not None:
                    path.append(cur)
                    cur = parents.get(cur)
                if len(path) > 2:
                    # pick an interior cell to cut
                    cx, cy = path[np.random.randint(1, len(path)-1)]
                    maze[cy, cx] = 0
                cut_attempts += 1

            # Last resort: insert a barrier row/column
            if self._is_connected(maze, start, end):
                if np.random.rand() < 0.5:
                    r = np.random.randint(1, N-1)
                    maze[r, :] = 0
                else:
                    c = np.random.randint(1, N-1)
                    maze[:, c] = 0
                maze[sy, sx] = 1
                maze[ty, tx] = 1

        connected = self._is_connected(maze, start, end)
        path_len = self._shortest_path_len(maze, start, end) if connected else None
        return maze, connected, start, end, path_len
    
    def _is_connected(self, maze, start, end):
        """Check if given start connects to end using BFS."""
        from collections import deque
        sx, sy = start
        tx, ty = end
        if maze[sy, sx] == 0 or maze[ty, tx] == 0:
            return False
        N = self.grid_size
        visited = np.zeros_like(maze, dtype=bool)
        queue = deque([(sx, sy)])
        visited[sy, sx] = True
        
        while queue:
            x, y = queue.popleft()
            
            # Check if we reached the exit
            if x == tx and y == ty:
                return True
            
            # Explore neighbors (4-connected)
            for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < N and 
                    0 <= ny < N and 
                    not visited[ny, nx] and 
                    maze[ny, nx] == 1):
                    visited[ny, nx] = True
                    queue.append((nx, ny))
        
        return False

    def _is_connected_walls(self, walls):
        """Check connectivity using walls representation."""
        from collections import deque
        N = self.grid_size

        visited = np.zeros((N, N), dtype=bool)
        q = deque([(0, 0)])
        visited[0, 0] = True

        while q:
            x, y = q.popleft()
            if x == N - 1 and y == N - 1:
                return True
            # Move if the wall in that direction is open (False)
            if y > 0 and not walls['up'][y, x] and not visited[y - 1, x]:
                visited[y - 1, x] = True
                q.append((x, y - 1))
            if x < N - 1 and not walls['right'][y, x] and not visited[y, x + 1]:
                visited[y, x + 1] = True
                q.append((x + 1, y))
            if y < N - 1 and not walls['down'][y, x] and not visited[y + 1, x]:
                visited[y + 1, x] = True
                q.append((x, y + 1))
            if x > 0 and not walls['left'][y, x] and not visited[y, x - 1]:
                visited[y, x - 1] = True
                q.append((x - 1, y))
        return False
    
    def maze_to_image(self, maze, start, end, mark_endpoints=True):
        """
        Convert maze grid to pixel image.
        
        Args:
            maze: (grid_size, grid_size) binary array
            start: (sx,sy) start cell
            end: (tx,ty) end cell
            mark_endpoints: If True, mark start (green) and exit (red)
            
        Returns:
            img: (img_size, img_size, 3) RGB image [0-255]
        """
        # Block-based rendering: each block is fully wall or path
        img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        for gy in range(self.grid_size):
            for gx in range(self.grid_size):
                py_start = gy * self.block_size
                px_start = gx * self.block_size
                py_end = py_start + self.block_size
                px_end = px_start + self.block_size
                if maze[gy, gx] == 1:
                    img[py_start:py_end, px_start:px_end] = 255
                else:
                    img[py_start:py_end, px_start:px_end] = 0
        if mark_endpoints:
            sx, sy = start
            tx, ty = end
            sy0 = sy * self.block_size
            sx0 = sx * self.block_size
            ty0 = ty * self.block_size
            tx0 = tx * self.block_size
            img[sy0:sy0 + self.block_size, sx0:sx0 + self.block_size] = [0, 255, 0]
            img[ty0:ty0 + self.block_size, tx0:tx0 + self.block_size] = [255, 0, 0]
        return img
    
    def generate_dataset(self, num_samples, balance=True, min_path_len: int = 0):
        """
        Generate a dataset of maze images.
        
        Args:
            num_samples: Number of samples to generate
            balance: If True, ensure ~50/50 connected/disconnected split
            
        Returns:
            images: List of (img_size, img_size, 3) RGB images
            labels: List of binary labels (1=connected, 0=disconnected)
            mazes: List of (grid_size, grid_size) maze grids (for debugging)
            starts: List of (sx,sy) start cells
            ends:   List of (tx,ty) end cells
        """
        images = []
        labels = []
        mazes = []
        starts = []
        ends = []
        path_lengths = []
        
        if balance:
            # Generate equal numbers of connected and disconnected
            num_connected = num_samples // 2
            num_disconnected = num_samples - num_connected
            
            print(f"Generating {num_connected} connected and {num_disconnected} disconnected mazes...")
            
            # Generate connected mazes
            pbar = tqdm(total=num_connected, desc="Connected mazes")
            count = 0
            max_attempts = num_connected * 5000 if min_path_len > 0 else num_connected * 10
            attempts_total = 0
            while count < num_connected and attempts_total < max_attempts:
                maze, connected, start, end, plen = self.generate_maze(force_disconnected=False)
                attempts_total += 1
                if not connected:
                    continue  # need connected here
                if min_path_len > 0 and (plen is None or plen < min_path_len):
                    continue
                img = self.maze_to_image(maze, start, end)
                images.append(img)
                labels.append(1)
                mazes.append(maze)
                starts.append(start)
                ends.append(end)
                path_lengths.append(plen if plen is not None else -1)
                count += 1
                pbar.update(1)
            if count < num_connected and min_path_len > 0:
                print(f"[warn] Could only generate {count}/{num_connected} connected mazes with path_len >= {min_path_len} after {attempts_total} attempts.")
            pbar.close()
            
            # Generate disconnected mazes
            pbar = tqdm(total=num_disconnected, desc="Disconnected mazes")
            count = 0
            max_attempts_per_maze = 20
            while count < num_disconnected:
                attempts = 0
                disconnected_found = False
                while attempts < max_attempts_per_maze and not disconnected_found:
                    maze, connected, start, end, plen = self.generate_maze(force_disconnected=True)
                    attempts += 1
                    if connected:
                        # If maze accidentally connected, only accept if meets length constraint but we need disconnected here
                        continue
                    img = self.maze_to_image(maze, start, end)
                    images.append(img)
                    labels.append(0)
                    mazes.append(maze)
                    starts.append(start)
                    ends.append(end)
                    path_lengths.append(-1)
                    count += 1
                    disconnected_found = True
                    pbar.update(1)
                if not disconnected_found:
                    # Force disconnect by barrier
                    maze, _, start, end, _ = self.generate_maze(force_disconnected=False)
                    barrier = self.grid_size // 2
                    maze[barrier, :] = 0
                    sx, sy = start
                    tx, ty = end
                    maze[sy, sx] = 1
                    maze[ty, tx] = 1
                    if not self._is_connected(maze, start, end):
                        img = self.maze_to_image(maze, start, end)
                        images.append(img)
                        labels.append(0)
                        mazes.append(maze)
                        starts.append(start)
                        ends.append(end)
                        path_lengths.append(-1)
                        count += 1
                        pbar.update(1)
            pbar.close()
        else:
            # Generate random mazes (mix) honoring min_path_len for connected ones
            pbar = tqdm(total=num_samples, desc="Generating mazes")
            attempts_total = 0
            while len(images) < num_samples:
                want_connected = np.random.rand() < 0.5
                maze, connected, start, end, plen = self.generate_maze(force_disconnected=not want_connected)
                attempts_total += 1
                if connected and min_path_len > 0 and (plen is None or plen < min_path_len):
                    continue
                img = self.maze_to_image(maze, start, end)
                images.append(img)
                labels.append(1 if connected else 0)
                mazes.append(maze)
                starts.append(start)
                ends.append(end)
                path_lengths.append(plen if plen is not None else -1)
                pbar.update(1)
                if attempts_total > num_samples * 2000 and min_path_len > 0:
                    print(f"[warn] High rejection rate when enforcing min_path_len={min_path_len}; proceeding with current set ({len(images)}).")
                    break
            pbar.close()

        return images, labels, mazes, starts, ends, path_lengths


def save_dataset(images, labels, output_dir, split='train', starts=None, ends=None, path_lengths=None):
    """Save images and labels to disk."""
    img_dir = os.path.join(output_dir, 'imgs', split)
    os.makedirs(img_dir, exist_ok=True)
    
    metadata = []
    
    for idx, (img, label) in enumerate(tqdm(zip(images, labels), desc=f"Saving {split}", total=len(images))):
        # Save image with label in filename
        img_filename = f"maze_{idx:06d}_label{int(label)}.png"
        img_path = os.path.join(img_dir, img_filename)
        Image.fromarray(img).save(img_path)

        # Store metadata
        entry = {
            'filename': img_filename,
            'label': int(label),
            'connected': bool(label == 1)
        }
        if starts is not None and ends is not None:
            sx, sy = starts[idx]
            tx, ty = ends[idx]
            entry['start'] = {'x': int(sx), 'y': int(sy)}
            entry['end'] = {'x': int(tx), 'y': int(ty)}
        if path_lengths is not None:
            entry['path_length'] = int(path_lengths[idx]) if path_lengths[idx] is not None else -1
        metadata.append(entry)
    
    # Save metadata JSON
    metadata_path = os.path.join(output_dir, f'{split}_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved {len(images)} images to {img_dir}")
    print(f"Saved metadata to {metadata_path}")
    
    # Print statistics
    num_connected = sum(labels)
    num_disconnected = len(labels) - num_connected
    print(f"  Connected: {num_connected} ({100 * num_connected / len(labels):.1f}%)")
    print(f"  Disconnected: {num_disconnected} ({100 * num_disconnected / len(labels):.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Generate maze dataset for GazeControl")
    parser.add_argument('--output-dir', type=str, default='./Data/Maze', 
                       help='Output directory for dataset')
    parser.add_argument('--train-samples', type=int, default=10000,
                       help='Number of training samples')
    parser.add_argument('--val-samples', type=int, default=2000,
                       help='Number of validation samples')
    parser.add_argument('--test-samples', type=int, default=2000,
                       help='Number of test samples')
    parser.add_argument('--grid-size', type=int, default=6,
                       help='Grid size (blocks per dimension)')
    parser.add_argument('--block-size', type=int, default=4,
                       help='Block size (pixels per block)')
    parser.add_argument('--no-balance', action='store_true',
                       help='Do not balance connected/disconnected samples')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--random-endpoints', action='store_true',
                       help='Randomize start/end positions on edges for each maze')
    parser.add_argument('--min-path-length', type=int, default=0,
                       help='Require connected mazes to have shortest path length >= this (others must be disconnected).')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    print(f"Maze Dataset Generator")
    print(f"======================")
    print(f"Grid size: {args.grid_size}×{args.grid_size} blocks")
    print(f"Block size: {args.block_size}×{args.block_size} pixels")
    print(f"Image size: {args.grid_size * args.block_size}×{args.grid_size * args.block_size} pixels")
    print(f"Balance classes: {not args.no_balance}")
    print()
    
    generator = MazeGenerator(grid_size=args.grid_size, block_size=args.block_size, random_endpoints=bool(args.random_endpoints))
    
    # Generate datasets
    print("Generating training set...")
    train_images, train_labels, _, train_starts, train_ends, train_plens = generator.generate_dataset(
        args.train_samples,
        balance=not args.no_balance,
        min_path_len=args.min_path_length,
    )
    save_dataset(train_images, train_labels, args.output_dir, split='train', starts=train_starts, ends=train_ends, path_lengths=train_plens)
    print()
    
    print("Generating validation set...")
    val_images, val_labels, _, val_starts, val_ends, val_plens = generator.generate_dataset(
        args.val_samples,
        balance=not args.no_balance,
        min_path_len=args.min_path_length,
    )
    save_dataset(val_images, val_labels, args.output_dir, split='val', starts=val_starts, ends=val_ends, path_lengths=val_plens)
    print()
    
    print("Generating test set...")
    test_images, test_labels, _, test_starts, test_ends, test_plens = generator.generate_dataset(
        args.test_samples,
        balance=not args.no_balance,
        min_path_len=args.min_path_length,
    )
    save_dataset(test_images, test_labels, args.output_dir, split='test', starts=test_starts, ends=test_ends, path_lengths=test_plens)
    print()
    
    # Save example visualizations
    print("Saving example visualizations...")
    examples_dir = os.path.join(args.output_dir, 'examples')
    os.makedirs(examples_dir, exist_ok=True)
    
    for i in range(min(10, len(train_images))):
        img = train_images[i]
        label = train_labels[i]
        status = "connected" if label == 1 else "disconnected"
        
        # Save at larger scale for visibility
        img_large = np.repeat(np.repeat(img, 10, axis=0), 10, axis=1)
        Image.fromarray(img_large).save(
            os.path.join(examples_dir, f"example_{i:02d}_{status}.png")
        )
    
    print(f"Saved 10 example images to {examples_dir}")
    print()
    print("Dataset generation complete!")
    print(f"Total samples: {args.train_samples + args.val_samples + args.test_samples}")


if __name__ == '__main__':
    main()
