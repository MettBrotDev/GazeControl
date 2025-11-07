"""
Simple GUI Maze Editor for GazeControl datasets

- Edit an N×N maze grid (default 6×6)
  * Left click toggles wall/path
  * Set Start / Set End modes to place endpoints
  * Shows connectivity (start to end) dynamically
  * Save writes a PNG to imgs/<split>/ and appends metadata to <split>_metadata.json

Dataset compatibility
- Matches Datasets/generate_maze.py saving format
  filename: custom_YYYYmmdd_HHMMSS_<idx>_label{0|1}.png
  metadata entry: filename, label, connected, start{x,y}, end{x,y}

Usage
  python Datasets/maze_editor.py --output-dir ./Data/Maze --split train --grid-size 6 --block-size 4

"""

import os
import sys
import json
import glob
import time
import argparse
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
from PIL import Image

try:
    import tkinter as tk
    from tkinter import ttk, messagebox
except Exception as e:
    print("Tkinter is required for the maze editor.")
    raise


@dataclass
class MazeState:
    grid_size: int
    block_size: int
    start: Tuple[int, int]
    end: Tuple[int, int]
    grid: np.ndarray  # (N,N) uint8, 1=path, 0=wall


def bfs_connected(grid: np.ndarray, start: Tuple[int, int], end: Tuple[int, int]) -> bool:
    """4-neighbor BFS connectivity on binary grid (1=path)."""
    N = grid.shape[0]
    sx, sy = start
    tx, ty = end
    if grid[sy, sx] == 0 or grid[ty, tx] == 0:
        return False
    from collections import deque
    q = deque([(sx, sy)])
    seen = set([(sx, sy)])
    while q:
        x, y = q.popleft()
        if (x, y) == (tx, ty):
            return True
        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < N and 0 <= ny < N and (nx, ny) not in seen and grid[ny, nx] == 1:
                seen.add((nx, ny))
                q.append((nx, ny))
    return False


def render_image(grid: np.ndarray, start: Tuple[int, int], end: Tuple[int, int], block_size: int = 4, mark_endpoints: bool = True) -> np.ndarray:
    """Render grid to RGB image like MazeGenerator.maze_to_image."""
    N = grid.shape[0]
    H = W = N * block_size
    img = np.zeros((H, W, 3), dtype=np.uint8)
    for gy in range(N):
        for gx in range(N):
            py0 = gy * block_size
            px0 = gx * block_size
            if grid[gy, gx] == 1:
                img[py0:py0+block_size, px0:px0+block_size] = 255
            else:
                img[py0:py0+block_size, px0:px0+block_size] = 0
    if mark_endpoints:
        sx, sy = start
        tx, ty = end
        img[sy*block_size:(sy+1)*block_size, sx*block_size:(sx+1)*block_size] = [0, 255, 0]
        img[ty*block_size:(ty+1)*block_size, tx*block_size:(tx+1)*block_size] = [255, 0, 0]
    return img


class MazeEditorApp:
    def __init__(self, root: tk.Tk, output_dir: str, split: str, grid_size: int = 6, block_size: int = 4, cell_px: int = 48):
        self.root = root
        self.root.title("Maze Editor")

        self.output_dir = output_dir
        self.split = split
        self.img_dir = os.path.join(output_dir, 'imgs', split)
        os.makedirs(self.img_dir, exist_ok=True)

        self.N = int(grid_size)
        self.block_size = int(block_size)
        self.cell_px = int(cell_px)

        # State
        self.state = MazeState(
            grid_size=self.N,
            block_size=self.block_size,
            start=(0, 0),
            end=(self.N - 1, self.N - 1),
            grid=np.zeros((self.N, self.N), dtype=np.uint8),
        )
        # default: carve start/end cells as path
        self.state.grid[0, 0] = 1
        self.state.grid[self.N - 1, self.N - 1] = 1

        self.mode = tk.StringVar(value='toggle')  # 'toggle' | 'set_start' | 'set_end' | 'brush_path' | 'brush_wall'
        self._last_painted = None  # track last painted cell during drag

        # UI
        self._build_ui()
        self._redraw()
        self._update_status()

    def _build_ui(self):
        frm = ttk.Frame(self.root)
        frm.pack(fill=tk.BOTH, expand=True)

        # Canvas for grid
        self.canvas = tk.Canvas(frm, width=self.N * self.cell_px, height=self.N * self.cell_px, bg="#222222", highlightthickness=0)
        self.canvas.grid(row=0, column=0, rowspan=6, padx=8, pady=8)
        self.canvas.bind('<Button-1>', self.on_click)
        self.canvas.bind('<B1-Motion>', self.on_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_release)

        # Controls
        btn_toggle = ttk.Button(frm, text="Toggle Cell", command=lambda: self.mode.set('toggle'))
        btn_start = ttk.Button(frm, text="Set Start", command=lambda: self.mode.set('set_start'))
        btn_end = ttk.Button(frm, text="Set End", command=lambda: self.mode.set('set_end'))
        btn_clear = ttk.Button(frm, text="Clear", command=self.clear_grid)
        btn_fill = ttk.Button(frm, text="Fill Paths", command=self.fill_paths)
        btn_save = ttk.Button(frm, text="Save", command=self.save_current)
        btn_brush_p = ttk.Button(frm, text="Brush Path", command=lambda: self.mode.set('brush_path'))
        btn_brush_w = ttk.Button(frm, text="Brush Wall", command=lambda: self.mode.set('brush_wall'))
        btn_quit = ttk.Button(frm, text="Quit", command=self.root.destroy)

        btn_toggle.grid(row=0, column=1, sticky='ew', padx=8, pady=(8, 2))
        btn_start.grid(row=1, column=1, sticky='ew', padx=8, pady=2)
        btn_end.grid(row=2, column=1, sticky='ew', padx=8, pady=2)
        btn_clear.grid(row=3, column=1, sticky='ew', padx=8, pady=2)
        btn_fill.grid(row=4, column=1, sticky='ew', padx=8, pady=2)
        btn_save.grid(row=5, column=1, sticky='ew', padx=8, pady=2)
        btn_brush_p.grid(row=6, column=1, sticky='ew', padx=8, pady=2)
        btn_brush_w.grid(row=7, column=1, sticky='ew', padx=8, pady=2)
        btn_quit.grid(row=8, column=1, sticky='ew', padx=8, pady=(2, 8))

        self.lbl_mode = ttk.Label(frm, text="Mode: toggle")
        self.lbl_conn = ttk.Label(frm, text="Connected: ?")
        self.lbl_info = ttk.Label(frm, text=f"Grid {self.N}×{self.N}  Cell {self.cell_px}px  Block {self.block_size}px")
        self.lbl_mode.grid(row=9, column=0, columnspan=2, sticky='w', padx=8, pady=2)
        self.lbl_conn.grid(row=10, column=0, columnspan=2, sticky='w', padx=8, pady=2)
        self.lbl_info.grid(row=11, column=0, columnspan=2, sticky='w', padx=8, pady=(2, 8))

        # React to mode change
        def on_mode_change(*_):
            self.lbl_mode.config(text=f"Mode: {self.mode.get()}")
        self.mode.trace_add('write', on_mode_change)

    

    def _redraw(self):
        self.canvas.delete('all')
        for gy in range(self.N):
            for gx in range(self.N):
                x0 = gx * self.cell_px
                y0 = gy * self.cell_px
                x1 = x0 + self.cell_px
                y1 = y0 + self.cell_px
                val = self.state.grid[gy, gx]
                color = '#ffffff' if val == 1 else '#000000'
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline='#555555')
        # Draw start and end markers as overlays
        sx, sy = self.state.start
        ex, ey = self.state.end
        # slightly inset for visibility
        inset = max(2, self.cell_px // 10)
        self.canvas.create_rectangle(sx*self.cell_px+inset, sy*self.cell_px+inset,
                                     (sx+1)*self.cell_px-inset, (sy+1)*self.cell_px-inset,
                                     fill='#00cc00', outline='')
        self.canvas.create_rectangle(ex*self.cell_px+inset, ey*self.cell_px+inset,
                                     (ex+1)*self.cell_px-inset, (ey+1)*self.cell_px-inset,
                                     fill='#cc0000', outline='')

    def _update_status(self):
        conn = bfs_connected(self.state.grid, self.state.start, self.state.end)
        self.lbl_conn.config(text=f"Connected: {'YES' if conn else 'NO'}  Label={1 if conn else 0}")

    def on_click(self, event):
        gx = int(event.x // self.cell_px)
        gy = int(event.y // self.cell_px)
        if not (0 <= gx < self.N and 0 <= gy < self.N):
            return
        m = self.mode.get()
        if m == 'toggle':
            # toggle wall/path
            self.state.grid[gy, gx] = 0 if self.state.grid[gy, gx] == 1 else 1
            # ensure endpoints remain path
            sx, sy = self.state.start
            ex, ey = self.state.end
            self.state.grid[sy, sx] = 1
            self.state.grid[ey, ex] = 1
        elif m == 'set_start':
            self.state.start = (gx, gy)
            self.state.grid[gy, gx] = 1
        elif m == 'set_end':
            self.state.end = (gx, gy)
            self.state.grid[gy, gx] = 1
        elif m == 'brush_path':
            self._paint_cell(gx, gy, value=1)
        elif m == 'brush_wall':
            self._paint_cell(gx, gy, value=0)
        self._last_painted = (gx, gy)
        self._redraw()
        self._update_status()

    def on_drag(self, event):
        gx = int(event.x // self.cell_px)
        gy = int(event.y // self.cell_px)
        if not (0 <= gx < self.N and 0 <= gy < self.N):
            return
        if (gx, gy) == self._last_painted:
            return
        m = self.mode.get()
        if m == 'brush_path':
            self._paint_cell(gx, gy, value=1)
        elif m == 'brush_wall':
            self._paint_cell(gx, gy, value=0)
        else:
            return
        self._last_painted = (gx, gy)
        self._redraw()
        self._update_status()

    def on_release(self, _event):
        self._last_painted = None

    def _paint_cell(self, gx: int, gy: int, value: int):
        # avoid painting start/end to wall; always keep endpoints path
        sx, sy = self.state.start
        ex, ey = self.state.end
        if (gx, gy) in [(sx, sy), (ex, ey)]:
            self.state.grid[gy, gx] = 1
        else:
            self.state.grid[gy, gx] = 1 if value else 0

    def clear_grid(self):
        self.state.grid[:, :] = 0
        # keep endpoints as path
        sx, sy = self.state.start
        ex, ey = self.state.end
        self.state.grid[sy, sx] = 1
        self.state.grid[ey, ex] = 1
        self._redraw()
        self._update_status()

    def fill_paths(self):
        self.state.grid[:, :] = 1
        self._redraw()
        self._update_status()

    def _next_custom_index(self) -> int:
        # Count existing custom_* files to choose next index
        pattern = os.path.join(self.img_dir, 'custom_*_label*.png')
        files = glob.glob(pattern)
        return len(files)

    def save_current(self):
        grid = self.state.grid.copy()
        start = tuple(self.state.start)
        end = tuple(self.state.end)
        connected = bfs_connected(grid, start, end)
        label = 1 if connected else 0

        img_arr = render_image(grid, start, end, block_size=self.block_size, mark_endpoints=True)
        ts = time.strftime('%Y%m%d_%H%M%S')
        idx = self._next_custom_index()
        filename = f"custom_{ts}_{idx:06d}_label{label}.png"
        img_path = os.path.join(self.img_dir, filename)
        Image.fromarray(img_arr).save(img_path)

        # Update metadata
        meta_path = os.path.join(self.output_dir, f"{self.split}_metadata.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
                if not isinstance(metadata, list):
                    metadata = []
            except Exception:
                metadata = []
        else:
            metadata = []

        entry = {
            'filename': filename,
            'label': int(label),
            'connected': bool(label == 1),
            'start': {'x': int(start[0]), 'y': int(start[1])},
            'end': {'x': int(end[0]), 'y': int(end[1])},
        }
        metadata.append(entry)
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        messagebox.showinfo("Saved", f"Saved {filename}\nLabel={label}  Connected={'YES' if connected else 'NO'}")


def parse_args():
    p = argparse.ArgumentParser(description='GUI Maze Editor for GazeControl dataset')
    p.add_argument('--output-dir', type=str, default='./Data/Maze', help='Dataset root (contains imgs/<split>/ and <split>_metadata.json)')
    p.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test', 'custom'], help='Dataset split subdir for saving')
    p.add_argument('--grid-size', type=int, default=6, help='Grid size N (N×N)')
    p.add_argument('--block-size', type=int, default=4, help='Pixels per cell in saved PNG')
    p.add_argument('--cell-px', type=int, default=48, help='Onscreen cell size in pixels')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.join(args.output_dir, 'imgs', args.split), exist_ok=True)
    root = tk.Tk()
    app = MazeEditorApp(root, output_dir=args.output_dir, split=args.split, grid_size=args.grid_size, block_size=args.block_size, cell_px=args.cell_px)
    root.mainloop()


if __name__ == '__main__':
    main()
