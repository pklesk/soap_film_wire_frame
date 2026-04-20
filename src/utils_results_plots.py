import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    sizes = [32**2, 100**2, 317**2, 1000**2, 3163**2]    
    device_names = ["i7-10700", "Ryzen 9 9950X", "RTX 3090 Ampere", "RTX 5090 Blackwell"]    
    series = [
        {
            "sfwf_contraction_cpu_numpy": [0.017, 0.262, 7.054, 71.297, 2_712.149],
            "sfwf_contraction_cpu_numba_parallel": [0.025, 0.365, 2.664, 10.428, 400.334],
        },
        {
            "sfwf_contraction_cpu_numpy": [0.007, 0.097, 2.888, 26.448, 1_770.113],
            "sfwf_contraction_cpu_numba_parallel": [0.020, 0.163, 1.424, 4.847, 146.057],
        },
        {        
            "sfwf_contraction_cuda_large_atomicmax": [0.072, 0.473, 2.055, 2.076, 20.209],
            "sfwf_contraction_cuda_large_atomicmaxglosten": [0.072, 0.468, 2.065, 2.076, 15.325],
            "sfwf_contraction_cuda_large_hreducemax": [0.044, 0.283, 1.235, 1.251, 19.960],
            "sfwf_contraction_cuda_large_hreducemaxgs": [0.045, 0.284, 1.244, 1.250, 19.935],
            "sfwf_contraction_cuda_large_gridsync": [0.008, 0.052, 0.786, 4.178, 75.746]
        },
        {
            "sfwf_contraction_cuda_large_atomicmax": [0.025, 0.163, 0.737, 0.738, 5.015],
            "sfwf_contraction_cuda_large_atomicmaxglosten": [0.025, 0.163, 0.741, 0.736, 4.066],
            "sfwf_contraction_cuda_large_hreducemax": [0.016, 0.101, 0.460, 0.460, 4.861],
            "sfwf_contraction_cuda_large_hreducemaxgs": [0.016, 0.102, 0.463, 0.459, 4.855],
            "sfwf_contraction_cuda_large_gridsync": [0.006, 0.041, 0.334, 0.988, 12.724]            
        }
    ]

    method_colors = {
        "sfwf_contraction_cpu_numba_parallel": "#FFA500",
        "sfwf_contraction_cuda_large_atomicmax": "#007FFF",          
        "sfwf_contraction_cuda_large_atomicmaxglosten": "#000080",   
        "sfwf_contraction_cuda_large_hreducemax": "#00C957",         
        "sfwf_contraction_cuda_large_hreducemaxgs": "#006400",       
        "sfwf_contraction_cuda_large_gridsync": "#FF0000"            
    }

    title_str = "SOAP FILM IN A WIRE FRAME: LOG-LOG PLOT OF SPEED-UPS"
    fig = plt.figure(figsize=(16, 7))
    try:
        fig.canvas.manager.set_window_title(title_str)
    except AttributeError:
        pass

    markers_cpu = [".", "."]
    markers_cuda = ["o", "s", "^", "v", "*"]
    last_x = sizes[-1]
    first_x = sizes[0]
    
    LABEL_FONT_SIZE = 8.5      
    LEGEND_FONT_SIZE = 10.5    
    AXIS_LABEL_SIZE = 15.0     
    TITLE_SIZE = 20.0          
    TICK_LABEL_SIZE = 12.0     
    COLUMN_SPACING = 1.45

    for i, device_dict in enumerate(series):
        device_name = device_names[i]
        methods = list(device_dict.items())
        
        line_style = "-" if ("5090" in device_name or "Ryzen" in device_name) else (0, (4, 2))
        
        markers = markers_cpu if i < 2 else markers_cuda                    
        for j, (method_name, timings) in enumerate(methods):
            full_label = f"[{device_name}] {method_name}"
            clean_name = method_name # intermediate mechanism (in case sth should be replaced form the method name, now inactive)
            speed_ups = [cpu / gpu for cpu, gpu in zip(list(series[0].values())[0], timings)]
            
            default_color = "black" if i < 2 else "#7f7f7f"
            color = method_colors.get(clean_name, default_color)
            current_lw = 1.7 if (i == 0 and j == 0) else 1.7
            current_ms = 7 if (i == 0 and j == 0) else 7
            
            if i == 0:                
                plt.plot(sizes, speed_ups, marker=markers[j % len(markers)], color=color, label=full_label,
                         linewidth=current_lw, markersize=current_ms, markerfacecolor=color, 
                         linestyle=line_style, zorder=10)                                
                if j == 0:
                    plt.text(last_x * 1.03, speed_ups[-1] * 1.05, "1.0x", color="black", 
                             fontsize=LABEL_FONT_SIZE, va="bottom", ha="left")                
            else:
                plt.plot(sizes, speed_ups, marker=markers[j % len(markers)], markerfacecolor="none",
                         markeredgewidth=1.5, color=color, label=full_label, 
                         linestyle=line_style, linewidth=current_lw, markersize=current_ms, zorder=10)
            if i == 0 and j == 0:
                continue
            
            last_y = speed_ups[-1]
            x_pos = last_x * (COLUMN_SPACING ** (j + 1)) 
            
            plt.hlines(y=last_y, xmin=last_x, xmax=x_pos, colors="black", 
                       linewidth=0.5, alpha=0.15, zorder=2)
            
            bbox_props = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9)
            plt.text(x_pos, last_y, f"{last_y:.1f}x", color=color, 
                     fontsize=LABEL_FONT_SIZE, va="center", ha="center", 
                     bbox=bbox_props, zorder=11)

    plt.xscale("log")
    plt.yscale("log")
    plt.ylim(0.15, 1200) 
    
    plt.tick_params(axis="both", which="major", labelsize=TICK_LABEL_SIZE)
    plt.tick_params(axis="both", which="minor", labelsize=TICK_LABEL_SIZE * 0.8)
    
    plt.xlim(first_x * 0.85, last_x * (COLUMN_SPACING ** 5.6)) 
    
    plt.xlabel("PROBLEM SIZE ($n^2$)", fontsize=AXIS_LABEL_SIZE)
    plt.ylabel("SPEED-UP VS. CPU", fontsize=AXIS_LABEL_SIZE)
    plt.title(title_str, fontsize=TITLE_SIZE, pad=15)
    
    plt.grid(True, which="both", ls="-", color="#F5F5F5", zorder=1)    
        
    plt.legend(loc="upper left", fontsize=LEGEND_FONT_SIZE, framealpha=0.9, ncol=1, handlelength=3.0, labelspacing=0.2)
    
    plt.tight_layout(pad=1.0)
    plt.show()