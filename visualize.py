#!/usr/bin/env python3
"""
gLSM_fluid_3D 可视化脚本
生成凝胶耦合流场动画：流体速度场 + 两个凝胶体位置
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import glob, sys, os

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR  = Path(__file__).parent / "vis_output"
OUT_DIR.mkdir(exist_ok=True)

# 流体网格参数（与 sim.cpp 一致）
h = 0.5
fluidSize = (150, 50, 10)
Nx = int(fluidSize[0] / h) + 1   # 301
Ny = int(fluidSize[1] / h) + 1   # 101
Nz = int(fluidSize[2] / h) + 1   # 21

def load_velb(t):
    """读流体速度场"""
    f = DATA_DIR / f"Velb{t}.dat"
    if not f.exists():
        return None
    arr = np.fromstring(f.read_text(), sep='\n').reshape(-1, 3)
    ux = arr[:,0].reshape(Nz, Ny, Nx)
    uy = arr[:,1].reshape(Nz, Ny, Nx)
    return ux, uy          # 只取 XY 分量

def load_gel_rn(gel_id, t):
    """读凝胶节点坐标"""
    f = DATA_DIR / f"gel{gel_id}rn{t}.dat"
    if not f.exists():
        return None
    arr = np.fromstring(f.read_text(), sep='\n').reshape(-1, 3)
    return arr[:,0], arr[:,1]   # x, y

def load_conc(t):
    """读浓度场"""
    f = DATA_DIR / f"Conc{t}.dat"
    if not f.exists():
        return None
    arr = np.fromstring(f.read_text(), sep='\n')
    return arr.reshape(Nz, Ny, Nx)

# 找有哪些时间步
velb_files = sorted(glob.glob(str(DATA_DIR / "Velb*.dat")))
timesteps  = [int(Path(f).stem.replace("Velb","")) for f in velb_files]
print(f"找到 {len(timesteps)} 个时间步: {timesteps}")

if not timesteps:
    print("没有找到输出文件，请检查 data/ 目录。")
    sys.exit(1)

# ─── 静态多帧图 ─────────────────────────────────────────
stride = max(1, Nx // 30)   # 箭头间距
x_coords = np.arange(Nx) * h
y_coords = np.arange(Ny) * h
XX, YY = np.meshgrid(x_coords[::stride], y_coords[::stride])

ncols = min(len(timesteps), 3)
nrows = 1
fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5))
if ncols == 1:
    axes = [axes]

for ax, t in zip(axes, timesteps[:ncols]):
    result = load_velb(t)
    if result is None:
        continue
    ux, uy = result

    # 取 Z 中间层做 XY 截面
    zmid = Nz // 2
    ux2d = ux[zmid]    # shape (Ny, Nx)
    uy2d = uy[zmid]

    speed = np.sqrt(ux2d**2 + uy2d**2)
    im = ax.imshow(speed, origin='lower', cmap='viridis',
                   extent=[0, fluidSize[0], 0, fluidSize[1]],
                   aspect='auto', vmin=0)
    ax.quiver(XX, YY,
              ux2d[::stride, ::stride],
              uy2d[::stride, ::stride],
              color='white', alpha=0.6, scale=0.05,
              scale_units='xy', width=0.003)
    plt.colorbar(im, ax=ax, label='速度大小')

    # 画凝胶
    for gid, color, label in [(1,'red','凝胶1'), (2,'orange','凝胶2')]:
        rn = load_gel_rn(gid, t)
        if rn is not None:
            ax.scatter(rn[0], rn[1], c=color, s=4, label=label, zorder=5)
    ax.set_title(f't = {t}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend(fontsize=7)

fig.suptitle('gLSM 凝胶耦合流场 — XY 截面流速 + 凝胶位置', fontsize=13)
plt.tight_layout()
static_path = OUT_DIR / "fluid_gel_snapshot.png"
plt.savefig(static_path, dpi=150)
plt.close()
print(f"静态图已保存: {static_path}")

# ─── 动画 ─────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('gLSM 凝胶耦合流场动画')

result0 = load_velb(timesteps[0])
ux0, uy0 = result0
zmid = Nz // 2
speed0 = np.sqrt(ux0[zmid]**2 + uy0[zmid]**2)

im2 = ax2.imshow(speed0, origin='lower', cmap='viridis',
                 extent=[0, fluidSize[0], 0, fluidSize[1]],
                 aspect='auto', vmin=0, animated=True)
cb = plt.colorbar(im2, ax=ax2, label='速度大小')

Q = ax2.quiver(XX, YY,
               ux0[zmid][::stride, ::stride],
               uy0[zmid][::stride, ::stride],
               color='white', alpha=0.6, scale=0.05,
               scale_units='xy', width=0.003)

scatters = []
for gid, color, label in [(1,'red','凝胶1'), (2,'orange','凝胶2')]:
    rn = load_gel_rn(gid, timesteps[0])
    xs, ys = (rn if rn is not None else ([], []))
    sc = ax2.scatter(xs, ys, c=color, s=4, label=label, zorder=5)
    scatters.append((gid, sc))
ax2.legend(fontsize=8)
time_text = ax2.text(0.02, 0.96, '', transform=ax2.transAxes,
                     color='white', fontsize=10, va='top')

def animate(i):
    t = timesteps[i]
    result = load_velb(t)
    if result is None:
        return []
    ux, uy = result
    speed = np.sqrt(ux[zmid]**2 + uy[zmid]**2)
    vmax = speed.max() or 1e-10
    im2.set_array(speed)
    im2.set_clim(0, vmax)
    Q.set_UVC(ux[zmid][::stride, ::stride],
              uy[zmid][::stride, ::stride])
    for gid, sc in scatters:
        rn = load_gel_rn(gid, t)
        if rn is not None:
            sc.set_offsets(np.column_stack(rn))
    time_text.set_text(f't = {t}')
    return [im2, Q, time_text] + [sc for _, sc in scatters]

ani = animation.FuncAnimation(fig2, animate, frames=len(timesteps),
                               interval=400, blit=False)
anim_path = OUT_DIR / "fluid_gel_animation.gif"
ani.save(str(anim_path), writer='pillow', fps=3)
plt.close()
print(f"动画已保存: {anim_path}")

# ─── 凝胶中心轨迹图 ─────────────────────────────────────
bodycenter_files = list(DATA_DIR.glob("gel*bodycenter.dat"))
if bodycenter_files:
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    for f in sorted(bodycenter_files):
        gid = f.stem.replace("gel","").replace("bodycenter","")
        data = np.fromstring(f.read_text(), sep='\n').reshape(-1, 12)
        ts  = data[:,0]
        xs  = data[:,1]
        ys  = data[:,2]
        ax3.plot(ts, xs, label=f'凝胶{gid} X中心')
        ax3.plot(ts, ys, label=f'凝胶{gid} Y中心', linestyle='--')
    ax3.set_xlabel('时间步')
    ax3.set_ylabel('坐标')
    ax3.set_title('凝胶中心位置随时间变化')
    ax3.legend()
    traj_path = OUT_DIR / "gel_center_trajectory.png"
    fig3.savefig(traj_path, dpi=150)
    plt.close()
    print(f"轨迹图已保存: {traj_path}")

print("✅ 所有可视化完成！")
print(f"输出目录: {OUT_DIR}")
