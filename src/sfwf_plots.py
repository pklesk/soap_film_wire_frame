__author__ = "Przemysław Klęsk"
__email__ = "pklesk@zut.edu.pl"

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

def sfwf_plot(border, heights, title="", subtitle=""): 
    figsize = (12, 6)    
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, 2, width_ratios=[3, 3])
    ax1 = fig.add_subplot(gs[0])
    ax1.set_box_aspect(0.25)
    ax1.axhline(y=0.0, color="k", linewidth=1)     
    ax1.plot(border, color="r", linewidth=2)
    ax1.set_xlabel("BORDER")
    ax1.set_ylabel("HEIGHT")
    ax1.set_title("WIRE FRAME HEIGHTS ALONG BORDER", fontsize=14)
    border_n = len(border)
    corner_indices = [0, border_n // 4, border_n // 2, 3 * border_n // 4, border_n - 1]
    corner_labels = ["A", "B", "C", "D", "A"]
    ax1.scatter(corner_indices, [border[i] for i in corner_indices], color="k", s=20, zorder=3)
    for i, label in enumerate(corner_labels):
        ax1.text(corner_indices[i], border[corner_indices[i]], f" {label}", fontsize=14, color="k")        
    ax2 = fig.add_subplot(gs[1], projection="3d")         
    X1, X2 = np.meshgrid(np.arange(heights.shape[0]), np.arange(heights.shape[1]), indexing="ij")
    ax2.plot_surface(X1, X2, heights, cmap="coolwarm", edgecolor="k", linewidth=0.5, alpha=0.75)
    ax2.plot(X1[0, :], X2[0, :], heights[0, :], color="r", linewidth=5) 
    ax2.plot(X1[-1, :], X2[-1, :], heights[-1, :], color="r", linewidth=5)
    ax2.plot(X1[:, 0], X2[:, 0], heights[:, 0], color="r", linewidth=5)
    ax2.plot(X1[:, -1], X2[:, -1], heights[:, -1], color="r", linewidth=5)
    corners_3d = [
        (0, 0, heights[0, 0]),
        (0, heights.shape[1] - 1, heights[0, -1]),
        (heights.shape[0] - 1, heights.shape[1] - 1, heights[-1, -1]),        
        (heights.shape[0] - 1, 0 , heights[-1, 0])        
    ]    
    shift_point = 0.5
    shifts = [
        (-shift_point, -shift_point, shift_point),
        (shift_point, -shift_point, shift_point),
        (shift_point, shift_point, shift_point),
        (shift_point, -shift_point, shift_point)        
    ]
    shift_label = 0.1 * (np.max(heights) - np.min(heights))
    for k, (i, j, h) in enumerate(corners_3d):
        ax2.scatter(i + shifts[k][0], j + shifts[k][1], h + shifts[k][2] * np.sign(h), color="k", s=20, depthshade=False, zorder=20)
        ax2.text(i + shifts[k][0], j + shifts[k][1], h + (shifts[k][2] + shift_label) * np.sign(h), f" {corner_labels[k]}", fontsize=14, color="k", zorder=20)                 
    ax2.set_zlabel("HEIGHT")
    ax2.set_box_aspect([1, 1, 0.5])    
    fig.text(0.725, 0.9, title, fontsize=15, ha="center", va="bottom") # instead of: ax2.set_title(title)    
    fig.text(0.725, 0.85, subtitle, fontsize=15, ha="center", va="bottom") # instead of: ax2.set_title(title)    
    plt.subplots_adjust(wspace=0.5)
    plt.tight_layout()  
    plt.subplots_adjust(top=1.0, bottom=0.0, left=0.05, right=0.95)
    fig.canvas.manager.set_window_title("SOAP FILM IN WIRE FRAME: CONTRACTION ITERATION")
    plt.show()

def sfwf_plot_large(border, heights, title="", subtitle="", heights_extra=None, title_extra="", subtitle_extra="", ): 
    figsize = (18, 6) 
    fig = plt.figure(figsize=figsize)
    width_ratios=[2, 3, 3]
    gs = GridSpec(1, 3, width_ratios=width_ratios)
    ax1 = fig.add_subplot(gs[0])
    ax1.set_box_aspect(0.25)
    ax1.axhline(y=0.0, color="k", linewidth=1)     
    ax1.plot(border, color="r", linewidth=2)
    ax1.set_xlabel("BORDER")
    ax1.set_ylabel("HEIGHT")
    ax1.set_title("WIRE FRAME HEIGHTS ALONG BORDER", fontsize=14)
    border_n = len(border)
    corner_indices = [0, border_n // 4, border_n // 2, 3 * border_n // 4, border_n - 1]
    corner_labels = ["A", "B", "C", "D", "A"]
    ax1.scatter(corner_indices, [border[i] for i in corner_indices], color="k", s=20, zorder=3)
    for i, label in enumerate(corner_labels):
        ax1.text(corner_indices[i], border[corner_indices[i]], f" {label}", fontsize=14, color="k")        
    ax2 = fig.add_subplot(gs[1], projection="3d")         
    X1, X2 = np.meshgrid(np.arange(heights.shape[0]), np.arange(heights.shape[1]), indexing="ij")
    ax2.plot_surface(X1, X2, heights, cmap="coolwarm", edgecolor="k", linewidth=0.5, alpha=0.75)
    ax2.plot(X1[0, :], X2[0, :], heights[0, :], color="r", linewidth=5) 
    ax2.plot(X1[-1, :], X2[-1, :], heights[-1, :], color="r", linewidth=5)
    ax2.plot(X1[:, 0], X2[:, 0], heights[:, 0], color="r", linewidth=5)
    ax2.plot(X1[:, -1], X2[:, -1], heights[:, -1], color="r", linewidth=5)
    corners_3d = [
        (0, 0, heights[0, 0]),
        (0, heights.shape[1] - 1, heights[0, -1]),
        (heights.shape[0] - 1, heights.shape[1] - 1, heights[-1, -1]),        
        (heights.shape[0] - 1, 0 , heights[-1, 0])        
    ] 
    shift_point = 0.5
    shifts = [
        (-shift_point, -shift_point, shift_point),
        (shift_point, -shift_point, shift_point),
        (shift_point, shift_point, shift_point),
        (shift_point, -shift_point, shift_point)        
    ]
    shift_label = 0.1 * (np.max(heights) - np.min(heights))
    for k, (i, j, h) in enumerate(corners_3d):
        ax2.scatter(i + shifts[k][0], j + shifts[k][1], h + shifts[k][2] * np.sign(h), color="k", s=20, depthshade=False, zorder=20)
        ax2.text(i + shifts[k][0], j + shifts[k][1], h + (shifts[k][2] + shift_label) * np.sign(h), f" {corner_labels[k]}", fontsize=14, color="k", zorder=20)    
    ax2.set_zlabel("HEIGHT")
    ax2.set_box_aspect([1, 1, 0.5])
    x_shift = (width_ratios[0] + 0.5 * width_ratios[1]) / sum(width_ratios)    
    fig.text(x_shift, 0.9, title, fontsize=15, ha="center", va="bottom") # instead of: ax2.set_title(title)    
    fig.text(x_shift, 0.85, subtitle, fontsize=15, ha="center", va="bottom") # instead of: ax2.set_title(title)    
    plt.subplots_adjust(wspace=0.5)
    ax3 = fig.add_subplot(gs[2], projection="3d")         
    X1, X2 = np.meshgrid(np.arange(heights_extra.shape[0]), np.arange(heights_extra.shape[1]), indexing="ij")
    ax3.plot_surface(X1, X2, heights_extra, cmap="coolwarm", edgecolor="k", linewidth=0.5, alpha=0.75)
    ax3.plot(X1[0, :], X2[0, :], heights_extra[0, :], color="r", linewidth=5) 
    ax3.plot(X1[-1, :], X2[-1, :], heights_extra[-1, :], color="r", linewidth=5)
    ax3.plot(X1[:, 0], X2[:, 0], heights_extra[:, 0], color="r", linewidth=5)
    ax3.plot(X1[:, -1], X2[:, -1], heights_extra[:, -1], color="r", linewidth=5)
    corners_3d = [
        (0, 0, heights_extra[0, 0]),
        (0, heights_extra.shape[1] - 1, heights_extra[0, -1]),
        (heights_extra.shape[0] - 1, heights_extra.shape[1] - 1, heights_extra[-1, -1]),        
        (heights_extra.shape[0] - 1, 0 , heights_extra[-1, 0]) 
    ]        
    for k, (i, j, h) in enumerate(corners_3d):
        ax3.scatter(i + shifts[k][0], j + shifts[k][1], h + shifts[k][2] * np.sign(h), color="k", s=20, depthshade=False, zorder=20)
        ax3.text(i + shifts[k][0], j + shifts[k][1], h + (shifts[k][2] + shift_label) * np.sign(h), f" {corner_labels[k]}", fontsize=14, color="k", zorder=20)                 
    ax3.set_zlabel("HEIGHT")
    ax3.set_box_aspect([1, 1, 0.5])
    x_shift = (width_ratios[0] + width_ratios[1] + 0.5 * width_ratios[2]) / sum(width_ratios)
    titles_shift = 0.025 
    fig.text(x_shift - titles_shift, 0.925, title_extra, fontsize=15, ha="center", va="bottom") # instead of: ax2.set_title(title)    
    fig.text(x_shift - titles_shift, 0.875, subtitle_extra, fontsize=15, ha="center", va="bottom") # instead of: ax2.set_title(title)        
    plt.tight_layout()  
    plt.subplots_adjust(top=1.0, bottom=0.0, left=0.05, right=0.95, wspace=0.05)
    fig.canvas.manager.set_window_title("SOAP FILM IN WIRE FRAME (CONTRACTION ITERATION)")
    plt.show()

def sfwf_plot_mc_trajectories(heights, trajectories, h_mean):
    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(111, projection="3d")
    ax1.set_box_aspect([1, 1, 0.5])
    fig.suptitle("MONTE CARLO TRAJECTORIES TOWARDS WIRE FRAME BORDERS")    
    X1, X2 = np.meshgrid(np.arange(heights.shape[0]), np.arange(heights.shape[1]), indexing="ij")
    red_edge_color = (1, 0, 0, 0.9)
    ax1.plot(X1[0, :], X2[0, :], heights[0, :], color=red_edge_color, linewidth=3) 
    ax1.plot(X1[-1, :], X2[-1, :], heights[-1, :], color=red_edge_color, linewidth=3)
    ax1.plot(X1[:, 0], X2[:, 0], heights[:, 0], color=red_edge_color, linewidth=3)
    ax1.plot(X1[:, -1], X2[:, -1], heights[:, -1], color=red_edge_color, linewidth=3)    
    # rectangle for domain border at level = 0.0
    ax1.plot([0, heights.shape[0] - 1, heights.shape[0] - 1, 0, 0],
             [0, 0, heights.shape[1] - 1, heights.shape[1] - 1, 0],
             [0, 0, 0, 0, 0], color="k", linewidth=1)
    # plane at level = 0.0 with grid on it
    ax1.plot_surface(X1, X2, np.zeros_like(heights), color="gray", alpha=0.05, rstride=10, cstride=10, zorder=-1)
    for i in range(heights.shape[0]):
        ax1.plot([i, i], [0, heights.shape[1] - 1], [0, 0], color="lightgray", linestyle='-', linewidth=0.5, zorder=-1)
    for j in range(heights.shape[1]):
        ax1.plot([0, heights.shape[0] - 1], [j, j], [0, 0], color="lightgray", linestyle='-', linewidth=0.5, zorder=-1)        
    # trajectories
    colors = ["#9B30FF", "#0033B3", "#009940"]
    for k, traj in enumerate(trajectories):
        color = colors[k % len(colors)]    
        ax1.plot(traj[:, 0], traj[:, 1], np.zeros_like(traj[:, 0]), color=color, linewidth=1.0, alpha=1.0)
        end_i, end_j = traj[-1]    
        ax1.scatter(end_i, end_j, heights[end_i, end_j], color=color, s=15, zorder=10)        
        end_h = heights[int(end_i), int(end_j)]
        ax1.plot([end_i, end_i], [end_j, end_j], [0, end_h], linestyle="-", linewidth=2.0, color=color)
    # segment line for h_mean
    ax1.plot([traj[0, 0], traj[0, 0]], [traj[0, 1], traj[0, 1]], [0, h_mean], linestyle="-", color="k")    
    ax1.scatter(traj[0, 0], traj[0, 1], 0, color="k", s=15, zorder=10)
    ax1.scatter(traj[0, 0], traj[0, 1], h_mean, color="k", s=15, zorder=10)  
    ax1.text(traj[0, 0] - 3.0, traj[0, 1] - 2.0 , 0.1 * h_mean, r"$\widehat{h}$", color="k", fontsize=15, zorder=15)    
    # segment lines for corners
    corners = [(0, 0), (0, heights.shape[1] - 1), (heights.shape[0] - 1, heights.shape[1] - 1), (heights.shape[0] - 1, 0)]
    corner_labels = ["A", "B", "C", "D"] # corner labels
    for i, j in corners:
        corner_h = heights[i, j]
        ax1.plot([i, i], [j, j], [0, corner_h], color="k", linestyle="-", linewidth=1.0, zorder=10)        
    corners_3d = [
        (0, 0, heights[0, 0]),
        (0, heights.shape[1] - 1, heights[0, -1]),
        (heights.shape[0] - 1, heights.shape[1] - 1, heights[-1, -1]),        
        (heights.shape[0] - 1, 0 , heights[-1, 0])        
    ] 
    shift_point = 0.25
    shifts = [
        (-shift_point, -shift_point, shift_point),
        (shift_point, -shift_point, shift_point),
        (shift_point, shift_point, shift_point),
        (shift_point, -shift_point, shift_point)        
    ]
    shift_label = 0.1 * (np.max(heights) - np.min(heights))
    for k, (i, j, h) in enumerate(corners_3d):
        ax1.scatter(i + shifts[k][0], j + shifts[k][1], h + shifts[k][2] * np.sign(h), color="k", s=10, depthshade=False, zorder=20)
        ax1.text(i + shifts[k][0], j + shifts[k][1], h + (shifts[k][2] + shift_label) * np.sign(h), f" {corner_labels[k]}", fontsize=14, color="k", zorder=20)    
    ax1.set_zlabel("HEIGHT")
    plt.tight_layout()  
    plt.subplots_adjust(top=1.0, bottom=0.0, left=0.05, right=0.95)
    fig.canvas.manager.set_window_title("SOAP FILM IN WIRE FRAME: MONTE CARLO TRAJECTORIES")    
    plt.show()
