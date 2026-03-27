import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import glob, sys

DATA_DIR = Path("data_gelonly_tube")
OUT_DIR  = Path("vis_output")
OUT_DIR.mkdir(exist_ok=True)

# 管状凝胶外框 200x30x30
LX, LY, LZ = 200, 30, 30

def load_conc(t):
    f = DATA_DIR / f"gel1um{t}.dat"
    if not f.exists():
        return None
    try:
        arr = np.fromstring(f.read_text(), sep='\n')
        # gLSM array indexing: zi*(LX+2)*(LY+2) + yi*(LX+2) + xi
        # Effective size is (LZ+2) x (LY+2) x (LX+2)
        grid = arr.reshape(LZ+2, LY+2, LX+2)
        return grid
    except:
        return None

# Find timesteps
files = sorted(glob.glob(str(DATA_DIR / "gel1um*.dat")))
if not files:
    print("No data found!")
    sys.exit(1)

timesteps = sorted([int(Path(f).stem.replace("gel1um","")) for f in files])
print(f"Found {len(timesteps)} frames: {timesteps}")

fig, ax = plt.subplots(figsize=(12, 4))
ax.set_title("Gel-Only Tube BZ Wave (Middle slice Z=15)")
ax.set_xlabel("X")
ax.set_ylabel("Y")

grid0 = load_conc(timesteps[0])
zmid = LZ // 2 + 1
# Plot XY slice
im = ax.imshow(grid0[zmid, 1:LY+1, 1:LX+1], origin='lower', cmap='plasma',
               vmin=0.0, vmax=0.4, extent=[0, LX, 0, LY])
cb = plt.colorbar(im, ax=ax, label='u (BZ Activator)')

time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='white',
                    fontsize=12, fontweight='bold', va='top')

def update(frame_i):
    t = timesteps[frame_i]
    grid = load_conc(t)
    if grid is not None:
        slice_xy = grid[zmid, 1:LY+1, 1:LX+1]
        im.set_array(slice_xy)
    time_text.set_text(f'Step = {t}')
    return [im, time_text]

ani = animation.FuncAnimation(fig, update, frames=len(timesteps), blit=True)
out_path = OUT_DIR / "gelonly_tube_bz.gif"
ani.save(out_path, writer='pillow', fps=10)
print(f"Saved animation to {out_path}")
