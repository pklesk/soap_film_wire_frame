import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    sizes = [32**2, 100**2, 317**2, 1000**2, 3163**2]
    
    device_names = ["i7-10700@2.90GHz", "RTX 3090", "RTX 5090 Blackwell"]
    
    series = [
        {"sfwf_contraction_cpu_numpy": [0.037, 0.396, 7.800, 114.355, 4086.422]},
        {        
            "sfwf_contraction_cuda_large_atomicmax": [0.075, 0.471, 2.089, 2.110, 22.557],
            "sfwf_contraction_cuda_large_atomicmaxglosten": [0.075, 0.470, 2.079, 2.115, 18.943],
            "sfwf_contraction_cuda_large_hreducemax": [0.045, 0.283, 1.250, 1.277, 22.732],
            "sfwf_contraction_cuda_large_hreducemaxgs": [0.045, 0.284, 1.265, 1.286, 22.714],
            "sfwf_contraction_cuda_large_gridsync": [0.008, 0.050, 0.820, 4.660, 88.446]
        },
        {
            "sfwf_contraction_cuda_large_atomicmax": [0.025, 0.158, 0.738, 0.720, 5.606],
            "sfwf_contraction_cuda_large_atomicmaxglosten": [0.025, 0.157, 0.731, 0.726, 5.726],
            "sfwf_contraction_cuda_large_hreducemax": [0.016, 0.100, 0.457, 0.447, 5.323],
            "sfwf_contraction_cuda_large_hreducemaxgs": [0.016, 0.100, 0.460, 0.451, 5.339],
            "sfwf_contraction_cuda_large_gridsync": [0.006, 0.040, 0.368, 1.064, 13.800]            
        }
    ]

    method_colors = {
        "atomicmax": "#007FFF",          
        "atomicmaxglosten": "#000080",   
        "hreducemax": "#00C957",         
        "hreducemaxgs": "#006400",       
        "gridsync": "#FF0000"            
    }

    title_str = "SOAP FILM IN A WIRE FRAME: PLOT OF SPEED-UPS"
    fig = plt.figure(figsize=(16, 7))
    try:
        fig.canvas.manager.set_window_title(title_str)
    except AttributeError:
        pass

    markers = ["o", "s", "^", "v", "*"]
    last_x = sizes[-1]
    first_x = sizes[0]
    
    LABEL_FONT_SIZE = 8.5      
    LEGEND_FONT_SIZE = 11.5    
    AXIS_LABEL_SIZE = 15.0     
    TITLE_SIZE = 20.0          
    TICK_LABEL_SIZE = 12.0     

    COLUMN_SPACING = 1.45

    for i, device_dict in enumerate(series):
        device_name = device_names[i]
        methods = list(device_dict.items())
        
        line_style = "-" if "5090" in device_name else (0, (4, 2))
        
        for j, (method_name, timings) in enumerate(methods):
            full_label = f"[{device_name}] {method_name}"
            clean_name = method_name.replace("sfwf_contraction_cuda_large_", "")
            speed_ups = [cpu / gpu for cpu, gpu in zip(list(series[0].values())[0], timings)]
            
            if i == 0:
                plt.plot(sizes, speed_ups, marker="o", color="black", label=full_label,
                         linewidth=1.2, markersize=4, markerfacecolor="black", zorder=10)
                plt.text(last_x * 1.03, speed_ups[-1] * 1.05, "1.0x", color="black", 
                         fontsize=LABEL_FONT_SIZE, va="bottom", ha="left")
                continue

            color = method_colors.get(clean_name, "#7f7f7f")
            plt.plot(sizes, speed_ups, marker=markers[j % len(markers)], markerfacecolor="none",
                     markeredgewidth=1.5, color=color, label=full_label, 
                     linestyle=line_style, linewidth=1.8, markersize=7, zorder=10)

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
    
    # Skupiamy się na danych powyżej 0.4x (SFWF ma punkty startowe poniżej 1.0x)
    plt.ylim(0.4, 1200) 
    
    plt.tick_params(axis="both", which="major", labelsize=TICK_LABEL_SIZE)
    plt.tick_params(axis="both", which="minor", labelsize=TICK_LABEL_SIZE * 0.8)
    
    plt.xlim(first_x * 0.85, last_x * (COLUMN_SPACING ** 5.6)) 
    
    plt.xlabel("PROBLEM SIZE ($n^2$)", fontsize=AXIS_LABEL_SIZE)
    plt.ylabel("SPEED-UP VS. CPU", fontsize=AXIS_LABEL_SIZE)
    plt.title(title_str, fontsize=TITLE_SIZE, pad=15)
    
    plt.grid(True, which="both", ls="-", color="#D3D3D3", alpha=0.3, zorder=1)
    
    # POPRAWIONA LINIA BAZOWA: Skrócona do zakresu danych
    plt.hlines(y=1, xmin=first_x, xmax=last_x, color="black", linestyle="-", 
               linewidth=1, alpha=0.2, zorder=2)
    
    plt.legend(loc="upper left", fontsize=LEGEND_FONT_SIZE, framealpha=0.9, ncol=1, handlelength=3.0)
    
    plt.tight_layout(pad=1.0)
    plt.show()