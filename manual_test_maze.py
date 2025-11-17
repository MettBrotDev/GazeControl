import os
import argparse
import random
import importlib
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, filedialog
from torchvision import transforms
from torch.distributions import Categorical

from models import GazeControlModel, Agent
from train_maze import crop_patch, eight_dir_deltas, MazeDataset


class SimpleImageFolderDataset:
	"""Plain image folder loader (no metadata). Returns (img, 0, start_xy=center).

	- root_dir: directory containing images (png/jpg/jpeg)
	- transform: torchvision transform applied to PIL image
	"""
	def __init__(self, root_dir: str, transform=None):
		self.root = root_dir
		self.transform = transform
		exts = ('.png', '.jpg', '.jpeg')
		self.files = [os.path.join(root_dir, f) for f in sorted(os.listdir(root_dir)) if f.lower().endswith(exts)]
		if not self.files:
			raise FileNotFoundError(f"No images found in folder: {root_dir}")

	def __len__(self):
		return len(self.files)

	def __getitem__(self, idx):
		from PIL import Image
		p = self.files[idx]
		img = Image.open(p).convert('RGB')
		if self.transform:
			img = self.transform(img)
		# label=0 placeholder; start at center
		start_xy = torch.tensor([0.5, 0.5], dtype=torch.float32)
		return img, torch.tensor(0, dtype=torch.long), start_xy


def to_display(x):
	return ((x.detach().cpu().clamp(-1, 1) + 1.0) / 2.0)


def composite_fovea(raw_patches, resized_patches):
	"""
	Compose a centered overlay that shows exactly what the model sees, without borders.
	We paste each encoder input (e.g., 4x4) upscaled with nearest-neighbor to the
	corresponding raw patch size (1x, 2x, 4x), all sharing the same center.
	raw_patches and resized_patches must both be ordered [largest, mid, smallest].
	"""
	if not raw_patches or not resized_patches:
		return None
	# Largest defines canvas
	pL = raw_patches[0]  # (C, HL, WL)
	C, HL, WL = pL.shape
	canvas = torch.zeros(3, HL, WL)
	# Helper to paste a patch centered (expects CHW in [0,1])
	def paste_center(dst, src):
		_, h, w = src.shape
		y0 = (HL - h) // 2
		x0 = (WL - w) // 2
		dst[:, y0:y0+h, x0:x0+w] = src
	# Paste largest encoder-view content (resized 4x4 upscaled to largest size) as background
	enc_large = resized_patches[0]  # (3,h0,w0) e.g., 4x4 in [-1,1]
	enc_large_up = F.interpolate(enc_large.unsqueeze(0), size=(HL, WL), mode='nearest')[0]
	paste_center(canvas, to_display(enc_large_up))
	for i, p in enumerate(raw_patches[:3]):
		# Paste the encoder-view content for this scale (upscaled 4x4 to (h,w)) centered
		_, h, w = p.shape
		y0 = (HL - h) // 2
		x0 = (WL - w) // 2
		enc_p = resized_patches[i]
		enc_up = F.interpolate(enc_p.unsqueeze(0), size=(h, w), mode='nearest')[0]
		paste_center(canvas, to_display(enc_up))
	return canvas.permute(1, 2, 0).numpy()


class MazeManualTestGUI:
	def __init__(self, root, config_module: str, checkpoint: str, split: str = 'val', idx: int | None = None, deterministic: bool = False, root_dir: str | None = None, images_dir: str | None = None):
		self.root = root
		self.root.title("Manual Maze Gaze Test")
		self.cfg = importlib.import_module(config_module).Config
		self.device = self.cfg.DEVICE
		self.split = split
		self.det = deterministic

		# Dataset (match training transforms to [-1,1])
		self.transform = transforms.Compose([
			transforms.Resize(self.cfg.IMG_SIZE),
			transforms.ToTensor(),
			transforms.Lambda(lambda x: x * 2.0 - 1.0),
		])

		self.mode = tk.StringVar(value='maze')  # 'maze' uses metadata dataset, 'folder' uses plain images
		self.root_dir = None
		self.images_dir = None

		def default_maze_root():
			r = getattr(self.cfg, 'MAZE_ROOT', os.path.join('./Data', 'Maze'))
			if not os.path.exists(r):
				alt = os.path.join('./Data', 'Maze')
				if os.path.exists(alt):
					r = alt
			return r

		# Initialize dataset according to provided paths
		if images_dir:
			self.set_dataset_folder(images_dir)
		else:
			self.set_dataset_maze(root_dir or default_maze_root(), split)
		self.idx = idx if idx is not None else random.randrange(len(self.ds))

		# Model and agent (instantiate like training)
		self.model = GazeControlModel(
			encoder_output_size=self.cfg.ENCODER_OUTPUT_SIZE,
			state_size=self.cfg.HIDDEN_SIZE,
			img_size=self.cfg.IMG_SIZE,
			fovea_size=self.cfg.FOVEA_OUTPUT_SIZE,
			pos_encoding_dim=self.cfg.POS_ENCODING_DIM,
			lstm_layers=self.cfg.LSTM_LAYERS,
			decoder_latent_ch=self.cfg.DECODER_LATENT_CH,
			k_scales=getattr(self.cfg, 'K_SCALES', 3),
			fuse_to_dim=getattr(self.cfg, 'FUSION_TO_DIM', None),
			fusion_hidden_mul=getattr(self.cfg, 'FUSION_HIDDEN_MUL', 2.0),
			encoder_c1=getattr(self.cfg, 'ENCODER_C1', None),
			encoder_c2=getattr(self.cfg, 'ENCODER_C2', None),
		).to(self.device)
		self.agent = Agent(state_size=self.cfg.HIDDEN_SIZE, pos_encoding_dim=self.cfg.POS_ENCODING_DIM,
						   stop_init_bias=float(getattr(self.cfg, 'RL_STOP_INIT_BIAS', -8.0))).to(self.device)
		self._load_checkpoint(checkpoint)
		self.model.eval(); self.agent.eval()

		# State
		self.image = None
		self.label = None
		self.start_xy = None
		self.gaze = None
		self.state = None
		self.path = []
		self.step = 0
		self.stopped = False
		self.last_probs = None
		self.last_rec = None
		self.last_patches = None
		# Baseline CNN prediction state
		self.baseline_model_var = tk.StringVar(value='MazeCNNClassifier')
		self.baseline_chkpt_var = tk.StringVar(value='')
		self.baseline_result = None
		self.baseline_last_pred = None  # 0/1
		self.baseline_last_prob = None  # probability for class 1 (connected)

		# GUI layout
		self._build_gui()
		# Load first image and draw
		self._load_sample(self.idx)
		self._draw()

	def _build_gui(self):
		# Main frame
		main_frame = ttk.Frame(self.root)
		main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

		# Controls
		control = ttk.Frame(main_frame)
		control.pack(fill=tk.X, pady=(0, 10))
		self.btn_next = ttk.Button(control, text="Next step", command=self.on_next)
		self.btn_next.pack(side=tk.LEFT, padx=(0, 6))
		self.btn_run100 = ttk.Button(control, text="Run 100 steps", command=lambda: self.start_auto_run(100))
		self.btn_run100.pack(side=tk.LEFT, padx=(0, 6))
		self.btn_stop = ttk.Button(control, text="Stop", command=self.stop_auto_run, state='disabled')
		self.btn_stop.pack(side=tk.LEFT, padx=(0, 6))
		ttk.Button(control, text="Reset on image", command=self.on_reset).pack(side=tk.LEFT, padx=(0, 6))
		self.btn_next_img = ttk.Button(control, text=f"Next {self.split} image", command=self.on_next_image)
		self.btn_next_img.pack(side=tk.LEFT, padx=(0, 6))
		# New: Random image button
		self.btn_rand_img = ttk.Button(control, text="Random image", command=self.on_random_image)
		self.btn_rand_img.pack(side=tk.LEFT, padx=(0, 6))
		self.status = ttk.Label(control, text="Click the original image to set gaze; Next step follows policy")
		self.status.pack(side=tk.LEFT, padx=(10, 0))

		# Dataset controls row
		dsbar = ttk.Frame(main_frame)
		dsbar.pack(fill=tk.X, pady=(0, 8))
		ttk.Label(dsbar, text="Mode:").pack(side=tk.LEFT)
		ttk.Label(dsbar, textvariable=self.mode).pack(side=tk.LEFT, padx=(2, 12))

		ttk.Label(dsbar, text="Split:").pack(side=tk.LEFT)
		self.split_var = tk.StringVar(value=self.split)
		split_cb = ttk.Combobox(dsbar, textvariable=self.split_var, values=['train','val','test','custom'], width=8, state='readonly')
		split_cb.pack(side=tk.LEFT)
		ttk.Button(dsbar, text="Set split", command=self.on_set_split).pack(side=tk.LEFT, padx=(4, 12))

		ttk.Button(dsbar, text="Browse dataset root…", command=self.on_browse_root).pack(side=tk.LEFT, padx=(0, 6))
		ttk.Button(dsbar, text="Open images folder…", command=self.on_browse_folder).pack(side=tk.LEFT)

		# Baseline classifier controls row
		basebar = ttk.Frame(main_frame)
		basebar.pack(fill=tk.X, pady=(0, 8))
		ttk.Label(basebar, text="Baseline model:").pack(side=tk.LEFT)
		# Discover classifier classes dynamically
		try:
			import models as models_mod
			import torch.nn as nn
			model_names = []
			for k, v in models_mod.__dict__.items():
				if isinstance(v, type) and issubclass(v, nn.Module) and 'Classifier' in k:
					model_names.append(k)
			model_names = sorted(set(model_names)) or ['MazeCNNClassifier']
		except Exception:
			model_names = ['MazeCNNClassifier']
		cmb_model = ttk.Combobox(basebar, values=model_names, textvariable=self.baseline_model_var, state='readonly', width=28)
		cmb_model.pack(side=tk.LEFT, padx=(4, 8))
		ttk.Button(basebar, text="Select checkpoint…", command=self.on_select_baseline_checkpoint).pack(side=tk.LEFT, padx=(0, 8))
		ttk.Button(basebar, text="Predict baseline", command=self.on_predict_baseline).pack(side=tk.LEFT)
		self.baseline_status = ttk.Label(basebar, text="")
		self.baseline_status.pack(side=tk.LEFT, padx=(10,0))

		# Manual move controls (8-direction arrows)
		mvframe = ttk.LabelFrame(main_frame, text="Manual move")
		mvframe.pack(fill=tk.X, pady=(0, 8))
		# Direction indices must match eight_dir_deltas in train_maze.py: [E=0, NE=1, N=2, NW=3, W=4, SW=5, S=6, SE=7]
		# Map UI arrows to deltas with image coordinates (y down).
		# eight_dir_deltas is defined with N=(0,+1) and S=(0,-1), so vertical is inverted vs screen.
		# We remap so that ↑ moves visually up (negative y): ↑ -> idx 6 (S), ↓ -> idx 2 (N).
		btn_map = [
			((0,0), ('↖', 5)), ((0,1), ('↑', 6)), ((0,2), ('↗', 7)),
			((1,0), ('←', 4)),                     ((1,2), ('→', 0)),
			((2,0), ('↙', 3)), ((2,1), ('↓', 2)), ((2,2), ('↘', 1)),
		]
		grid = [[None]*3 for _ in range(3)]
		for (r,c), (txt, idx) in btn_map:
			b = ttk.Button(mvframe, text=txt, width=3, command=lambda k=idx: self.on_manual_move(k))
			b.grid(row=r, column=c, padx=2, pady=2)
		# Add an empty disabled center to keep layout neat
		center = ttk.Label(mvframe, text='·')
		center.grid(row=1, column=1)

		# Matplotlib figure embedded in Tk
		self.fig, self.axs = plt.subplots(2, 2, figsize=(12, 10))
		self.fig.suptitle("Maze Manual Test")
		self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
		self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
		self.canvas.mpl_connect('button_press_event', self._on_click)

	def on_select_baseline_checkpoint(self):
		p = filedialog.askopenfilename(title="Select baseline checkpoint", filetypes=[('PyTorch checkpoint','*.pth *.pt'), ('All files','*.*')])
		if p:
			self.baseline_chkpt_var.set(p)
			self.baseline_status.config(text=f"ckpt: {os.path.basename(p)}")

	def on_predict_baseline(self):
		# Prepare current image in [0,1] for baseline
		if self.image is None:
			return
		if not self.baseline_chkpt_var.get():
			self.baseline_status.config(text="Select a checkpoint first")
			return
		try:
			import importlib, torch
			import models as models_mod
			ModelCls = getattr(models_mod, self.baseline_model_var.get(), None)
			if ModelCls is None:
				raise RuntimeError(f"Unknown model class {self.baseline_model_var.get()}")
			# x in [-1,1] -> [0,1]
			x01 = ((self.image.detach().to(self.device).clamp(-1,1) + 1.0) * 0.5)
			model = ModelCls().to(self.device).eval()
			state = torch.load(self.baseline_chkpt_var.get(), map_location=self.device)
			if isinstance(state, dict) and 'model_state_dict' in state:
				state = state['model_state_dict']
			model.load_state_dict(state, strict=False)
			with torch.no_grad():
				logits = model(x01).squeeze()
				prob = torch.sigmoid(logits).item()
				logit = float(logits.item())
				pred = int(prob >= 0.5)
			self.baseline_result = f"Baseline: prob={prob:.3f} pred={pred} logit={logit:.3f}"
			self.baseline_last_pred = int(pred)
			self.baseline_last_prob = float(prob)
			self.baseline_status.config(text=self.baseline_result)
		except Exception as e:
			self.baseline_status.config(text=f"Error: {e}")

	def _load_checkpoint(self, path: str):
		if os.path.exists(path):
			raw = torch.load(path, map_location=self.device)
		else:
			alt = os.path.join('runs', path)
			raw = torch.load(alt, map_location=self.device)
		if isinstance(raw, dict) and 'model_state_dict' in raw:
			self.model.load_state_dict(raw['model_state_dict'], strict=False)
			if 'agent_state_dict' in raw:
				self.agent.load_state_dict(raw['agent_state_dict'], strict=False)
		else:
			try:
				self.model.load_state_dict(raw, strict=True)
			except Exception:
				self.model.load_state_dict(raw, strict=False)

	def _load_sample(self, idx: int):
		img, lbl, start_xy = self.ds[idx]
		self.image = img.unsqueeze(0).to(self.device)
		self.label = int(lbl)
		self.start_xy = start_xy.to(self.device)
		self.gaze = self.start_xy.view(1, 2).clone()
		self.state = self.model.init_memory(1, self.device)
		self.path = []
		self.step = 0
		self.stopped = False
		self.last_probs = None
		self.last_rec = None
		self.last_patches = None
		# reset baseline cache for this image
		self.baseline_last_pred = None
		self.baseline_last_prob = None

	def _policy_step(self):
		# Build patches like training; also keep raw (un-resized) patches for overlay and size checks
		base_h, base_w = self.cfg.FOVEA_CROP_SIZE
		k = int(getattr(self.cfg, 'K_SCALES', 3))
		patches_resized = []
		raw_patches = []
		for i in range(k):
			sc = 2 ** i
			size = (base_h * sc, base_w * sc)
			raw_patches.append(crop_patch(self.image, self.gaze, size, resize_to=None)[0])  # (3,h,w)
			patches_resized.append(crop_patch(self.image, self.gaze, size, resize_to=self.cfg.FOVEA_OUTPUT_SIZE))
		with torch.no_grad():
			rec, self.state = self.model(patches_resized, self.state, self.gaze)
			h_t = self.state[0][-1]
			move_logits, decision_logits, stop_logit, value_t = self.agent.full_policy(h_t, self.gaze)

		# Convert to probabilities for display
		move_probs = torch.softmax(move_logits, dim=1)
		decision_probs = torch.softmax(decision_logits, dim=1)
		stop_prob = torch.sigmoid(stop_logit)
		self.last_probs = {
			'move_probs': move_probs.detach().cpu().flatten().tolist(),
			'decision_probs': decision_probs.detach().cpu().flatten().tolist(),
			'stop_prob': float(stop_prob.detach().cpu().item()),
			'value': float(value_t.detach().cpu().item()),
		}
		self.last_rec = rec.detach().clone()
		# store raw and resized patches for composite (largest->smallest)
		self.last_patches = [p.cpu() for p in reversed(raw_patches)]
		self.last_patches_resized = [p.cpu()[0] for p in reversed(patches_resized)]

		# Decide STOP and movement like training
		with torch.no_grad():
			if self.det:
				mv_idx = move_logits.argmax(dim=1)
				stop_prob = torch.sigmoid(stop_logit)
				st_sample = (stop_prob >= 0.5).float()
			else:
				cat = Categorical(logits=move_logits)
				mv_idx = cat.sample()
				stop_prob = torch.sigmoid(stop_logit)
				st_sample = torch.distributions.Bernoulli(probs=stop_prob).sample()

			# min steps
			min_steps = int(getattr(self.cfg, 'MIN_STEPS_BEFORE_STOP', 0))
			if self.step < min(min_steps, int(getattr(self.cfg, 'MAX_STEPS', 20))) - 1:
				st_sample = torch.zeros_like(st_sample)
			# confidence gate
			conf_th = float(getattr(self.cfg, 'STOP_CONF_THRESH', 0.9))
			if conf_th > 0.0:
				conf = torch.softmax(decision_logits, dim=1).amax(dim=1)
				st_sample = torch.where(conf >= conf_th, st_sample, torch.zeros_like(st_sample))

			# append current gaze before moving
			self.path.append(self.gaze.detach().clone())
			if (st_sample >= 0.5).item():
				self.stopped = True
				return

			# move
			deltas = eight_dir_deltas(self.cfg.MAX_MOVE, device=self.device)
			delta = deltas[mv_idx]
			self.gaze = torch.clamp(self.gaze + delta, 0.0, 1.0)
			if getattr(self.cfg, 'USE_GAZE_BOUNDS', False):
				frac = float(getattr(self.cfg, 'GAZE_BOUND_FRACTION', 0.1))
				lo, hi = frac, 1.0 - frac
				self.gaze = self.gaze.clamp(min=lo, max=hi)
			self.step += 1

	def _draw(self):
		# Panels: (0,0) original+path; (0,1) composite fovea; (1,0) reconstruction; (1,1) logits text
		for ax in self.axs.flat:
			ax.clear(); ax.axis('off')

		# a) Original with current location and previous path marked with x's
		# Use the exact image tensor fed to the encoder (self.image), only mapped to [0,1] for display
		img_disp = to_display(self.image[0]).permute(1, 2, 0).numpy()
		H, W = img_disp.shape[:2]
		# Fix the data coordinate system explicitly to [0,W]x[0,H] with origin at top-left
		self.axs[0, 0].imshow(img_disp, extent=(0, W, H, 0), interpolation='nearest')
		self.axs[0, 0].set_xlim(0, W)
		self.axs[0, 0].set_ylim(H, 0)
		xs, ys = [], []
		for g in self.path:
			# Match training index exactly: cx = int(norm*(W-1)), cy = int(norm*(H-1)); draw at pixel centers
			cx = int(g[0, 0].item() * (W - 1))
			cy = int(g[0, 1].item() * (H - 1))
			xs.append(cx + 0.5)
			ys.append(cy + 0.5)
		if xs:
			# Connect the steps with a thin line for better path visibility
			self.axs[0, 0].plot(xs, ys, color='red', linewidth=1.0, alpha=0.8)
			self.axs[0, 0].scatter(xs, ys, c='red', s=40, marker='x')
		# current
		cx = int(self.gaze[0, 0].item() * (W - 1)) + 0.5
		cy = int(self.gaze[0, 1].item() * (H - 1)) + 0.5
		self.axs[0, 0].scatter([cx], [cy], c='lime', s=60, marker='o')
		self.axs[0, 0].set_title('Original + path (x) and current (o)')

		# Overlay concise decisions and confidences on the image (top-left)
		def _lbl_str(y):
			return 'Connected' if int(y) == 1 else 'Not connected'
		overlay_lines = []
		# Model decision/confidence from last_probs (decision_probs of length 2)
		if self.last_probs is not None and 'decision_probs' in self.last_probs:
			dec = self.last_probs['decision_probs']
			if isinstance(dec, (list, tuple)) and len(dec) == 2:
				if dec[1] >= dec[0]:
					m_pred = 1; m_prob = float(dec[1])
				else:
					m_pred = 0; m_prob = float(dec[0])
				overlay_lines.append(f"Model: {_lbl_str(m_pred)} ({m_prob:.2f})")
		# Baseline CNN decision/confidence if available
		if self.baseline_last_pred is not None and self.baseline_last_prob is not None:
			b_pred = int(self.baseline_last_pred)
			p_pos = float(self.baseline_last_prob)
			b_prob_decision = p_pos if b_pred == 1 else (1.0 - p_pos)
			overlay_lines.append(f"CNN: {_lbl_str(b_pred)} ({b_prob_decision:.2f})")
		if overlay_lines:
			self.axs[0, 0].text(
				0.01, 0.01, "\n".join(overlay_lines),
				transform=self.axs[0, 0].transAxes,
				va='bottom', ha='left',
				fontsize=11, color='white',
				bbox=dict(facecolor='black', alpha=0.55, boxstyle='round,pad=0.4', edgecolor='none')
			)

		# b) Inputs overlay (largest->smallest)
		comp = composite_fovea(self.last_patches or [], getattr(self, 'last_patches_resized', []) or [])
		if comp is not None:
			self.axs[0, 1].imshow(comp, interpolation='nearest')
		self.axs[0, 1].set_title('Foveated inputs (overlay)')

		# c) Decoded image at this step and logits
		if self.last_rec is not None:
			rec_disp = to_display(self.last_rec[0]).permute(1, 2, 0).numpy()
			self.axs[1, 0].imshow(rec_disp)
		self.axs[1, 0].set_title('Reconstruction (this step)')

		# d) Probabilities text
		txt = 'No probabilities yet. Click Next step.'
		if self.last_probs is not None:
			mv = self.last_probs['move_probs']
			dec = self.last_probs['decision_probs']
			stp = self.last_probs['stop_prob']
			val = self.last_probs['value']
			txt = (
				f"Move probs (8):\n{[round(v,3) for v in mv]}\n\n"
				f"Decision probs (2): { [round(v,3) for v in dec] }\n"
				f"Stop prob: {stp:.3f}\nValue: {val:.3f}\n"
				f"Step: {self.step}  Stopped: {self.stopped}  Label: {self.label}"
			)
		# Append baseline prediction if available
		if self.baseline_result:
			txt += f"\n\n{self.baseline_result}"
		self.axs[1, 1].text(0.02, 0.98, txt, va='top', ha='left', fontsize=10)
		self.axs[1, 1].set_title('Current probabilities and info')

		self.canvas.draw()

	# Manual movement: user chooses one of 8 directions instead of policy
	def on_manual_move(self, dir_idx: int):
		if self.image is None:
			return
		# Compute model outputs at current gaze (for visualization) but override movement choice
		base_h, base_w = self.cfg.FOVEA_CROP_SIZE
		k = int(getattr(self.cfg, 'K_SCALES', 3))
		patches_resized = []
		raw_patches = []
		for i in range(k):
			sc = 2 ** i
			size = (base_h * sc, base_w * sc)
			raw_patches.append(crop_patch(self.image, self.gaze, size, resize_to=None)[0])
			patches_resized.append(crop_patch(self.image, self.gaze, size, resize_to=self.cfg.FOVEA_OUTPUT_SIZE))
		with torch.no_grad():
			rec, self.state = self.model(patches_resized, self.state, self.gaze)
			h_t = self.state[0][-1]
			move_logits, decision_logits, stop_logit, value_t = self.agent.full_policy(h_t, self.gaze)
		# Update displays
		move_probs = torch.softmax(move_logits, dim=1)
		decision_probs = torch.softmax(decision_logits, dim=1)
		stop_prob = torch.sigmoid(stop_logit)
		self.last_probs = {
			'move_probs': move_probs.detach().cpu().flatten().tolist(),
			'decision_probs': decision_probs.detach().cpu().flatten().tolist(),
			'stop_prob': float(stop_prob.detach().cpu().item()),
			'value': float(value_t.detach().cpu().item()),
		}
		self.last_rec = rec.detach().clone()
		self.last_patches = [p.cpu() for p in reversed(raw_patches)]
		self.last_patches_resized = [p.cpu()[0] for p in reversed(patches_resized)]

		# Append current gaze, ignore policy STOP and apply chosen move
		self.path.append(self.gaze.detach().clone())
		deltas = eight_dir_deltas(self.cfg.MAX_MOVE, device=self.device)
		delta = deltas[dir_idx]
		self.gaze = torch.clamp(self.gaze + delta, 0.0, 1.0)
		if getattr(self.cfg, 'USE_GAZE_BOUNDS', False):
			frac = float(getattr(self.cfg, 'GAZE_BOUND_FRACTION', 0.1))
			lo, hi = frac, 1.0 - frac
			self.gaze = self.gaze.clamp(min=lo, max=hi)
		self.step += 1
		# Do not change self.stopped here; user can keep moving manually
		self._draw()

	# Button/click callbacks
	def on_next(self):
		# Allow unlimited steps in the demo (ignore MAX_STEPS); stop only if the policy STOPs
		if not self.stopped:
			self._policy_step()
		self._draw()

	# ---- Auto-run controls ----
	def start_auto_run(self, n: int = 100):
		if getattr(self, 'auto_running', False):
			return
		self.auto_running = True
		self.auto_remaining = int(max(0, n))
		self.auto_job = None
		# UI states
		self.btn_run100.config(state='disabled')
		self.btn_stop.config(state='normal')
		self.btn_next.config(state='disabled')
		self.btn_next_img.config(state='disabled')
		self.btn_rand_img.config(state='disabled')
		self._auto_tick()

	def stop_auto_run(self):
		if getattr(self, 'auto_job', None) is not None:
			try:
				self.root.after_cancel(self.auto_job)
			except Exception:
				pass
			self.auto_job = None
		self.auto_running = False
		self.auto_remaining = 0
		# UI states
		self.btn_run100.config(state='normal')
		self.btn_stop.config(state='disabled')
		self.btn_next.config(state='normal')
		self.btn_next_img.config(state='normal')
		self.btn_rand_img.config(state='normal')

	def _auto_tick(self):
		# stop conditions
		if not getattr(self, 'auto_running', False):
			return
		if self.auto_remaining <= 0 or self.stopped:
			self.stop_auto_run()
			return
		# do one step
		self.on_next()
		self.auto_remaining -= 1
		# schedule next
		if self.auto_remaining > 0 and not self.stopped and self.auto_running:
			self.auto_job = self.root.after(1, self._auto_tick)
		else:
			self.stop_auto_run()

	# Dataset switching
	def set_dataset_maze(self, root_dir: str, split: str):
		self.mode.set('maze')
		self.root_dir = root_dir
		self.images_dir = None
		self.split = split
		self.ds = MazeDataset(root_dir, split=split, transform=self.transform)

	def set_dataset_folder(self, folder: str):
		self.mode.set('folder')
		self.images_dir = folder
		self.root_dir = None
		self.ds = SimpleImageFolderDataset(folder, transform=self.transform)
		self.split = 'folder'
        
	def on_set_split(self):
		if self.mode.get() != 'maze':
			return
		new_split = self.split_var.get()
		try:
			self.set_dataset_maze(self.root_dir, new_split)
			self.idx = 0
			self.btn_next_img.config(text=f"Next {self.split} image")
			self._load_sample(self.idx)
			self._draw()
		except Exception as e:
			self.status.config(text=f"Failed to set split: {e}")

	def on_browse_root(self):
		d = filedialog.askdirectory(title="Select dataset root (contains imgs/<split> and <split>_metadata.json)")
		if not d:
			return
		try:
			self.set_dataset_maze(d, self.split_var.get())
			self.idx = 0
			self.btn_next_img.config(text=f"Next {self.split} image")
			self._load_sample(self.idx)
			self._draw()
		except Exception as e:
			self.status.config(text=f"Invalid dataset root: {e}")

	def on_browse_folder(self):
		d = filedialog.askdirectory(title="Select images folder (png/jpg)")
		if not d:
			return
		try:
			self.set_dataset_folder(d)
			self.idx = 0
			self.btn_next_img.config(text=f"Next folder image")
			self._load_sample(self.idx)
			self._draw()
		except Exception as e:
			self.status.config(text=f"Invalid images folder: {e}")

	def on_reset(self):
		# Cancel auto-run if active
		self.stop_auto_run()
		# Reset network on current image
		self.gaze = self.start_xy.view(1, 2).clone()
		self.state = self.model.init_memory(1, self.device)
		self.path = []
		self.step = 0
		self.stopped = False
		self.last_probs = None
		self.last_rec = None
		self.last_patches = None
		self.baseline_last_pred = None
		self.baseline_last_prob = None
		self._draw()

	def on_next_image(self):
		# Cancel auto-run if active
		self.stop_auto_run()
		self.idx = (self.idx + 1) % len(self.ds)
		self._load_sample(self.idx)
		self._draw()

	def on_random_image(self):
		# Cancel auto-run if active
		self.stop_auto_run()
		if len(self.ds) <= 1:
			self.idx = 0
		else:
			new_idx = random.randrange(len(self.ds))
			if new_idx == self.idx:
				new_idx = (new_idx + 1) % len(self.ds)
			self.idx = new_idx
		self._load_sample(self.idx)
		self._draw()

	def _on_click(self, event):
		# Only respond to clicks on the original image (top-left subplot)
		if event.inaxes != self.axs[0, 0] or event.xdata is None or event.ydata is None:
			return
		# Map click to the exact pixel index used in training crop (nearest pixel center), then to normalized
		img_disp = to_display(self.image[0]).permute(1, 2, 0).numpy()
		H, W = img_disp.shape[:2]
		px = int(round(float(event.xdata)))
		py = int(round(float(event.ydata)))
		px = max(0, min(W - 1, px))
		py = max(0, min(H - 1, py))
		gx = (px + 0.5) / max(1.0, float(W - 1))
		gy = (py + 0.5) / max(1.0, float(H - 1))
		gx = max(0.0, min(1.0, gx))
		gy = max(0.0, min(1.0, gy))
		self.gaze = torch.tensor([[gx, gy]], device=self.device, dtype=torch.float32)
		if getattr(self.cfg, 'USE_GAZE_BOUNDS', False):
			frac = float(getattr(self.cfg, 'GAZE_BOUND_FRACTION', 0.1))
			lo, hi = frac, 1.0 - frac
			self.gaze = self.gaze.clamp(min=lo, max=hi)
		# Take a policy step from this gaze if desired; here we just update display
		self._draw()


def main():
	ap = argparse.ArgumentParser(description='Interactive Maze model demo')
	ap.add_argument('--config_module', default='config_maze')
	ap.add_argument('--checkpoint', default='gaze_control_model_local.pth')
	ap.add_argument('--split', choices=['train','val','test','custom','folder'], default='val')
	ap.add_argument('--root-dir', type=str, default=None, help='Maze dataset root (contains imgs/<split> and <split>_metadata.json)')
	ap.add_argument('--images-dir', type=str, default=None, help='Plain images folder (png/jpg). Overrides --root-dir if set')
	ap.add_argument('--index', type=int, default=None)
	ap.add_argument('--deterministic', action='store_true')
	args = ap.parse_args()

	root = tk.Tk()
	app = MazeManualTestGUI(
		root,
		args.config_module,
		args.checkpoint,
		split=args.split,
		idx=args.index,
		deterministic=args.deterministic,
		root_dir=args.root_dir,
		images_dir=args.images_dir,
	)
	root.mainloop()


if __name__ == '__main__':
	main()
