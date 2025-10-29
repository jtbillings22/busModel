import numpy as np
import pandas as pd
import os 
from scipy import stats
import matplotlib.pyplot as plt

class busRoute:
    def __init__(self, route_id):
        self.route_id = route_id
        self.stops = {}
    
    def generate_stop_pdf(self, stop_id, data): # requires stop_sequence and the data
        # computes the dataset into numeric values
        data = pd.to_numeric(data, errors ="coerce").dropna() 
        
        # guard 1 — skip if no numeric data
        if len(data) == 0:
            print(f"[SKIP] No valid numeric data for stop {stop_id} on route {self.route_id}")
            return
        
        # guard 2 — skip if all values identical (lognorm.fit fails on constant arrays)
        if data.min() == data.max():
            print(f"[SKIP] Constant data for stop {stop_id} on route {self.route_id}")
            return


        shape, loc, scale = stats.lognorm.fit(data, floc=0)
        x = np.linspace(data.min(), data.max(), 200)
        pdf = stats.lognorm.pdf(x, shape, loc, scale)

        # assigns values into stops dictionary - each stop_id has its associated PDF, as well as it's components
        self.stops[stop_id] = {
            'shape': shape,
            'loc': loc,
            'scale': scale,
            'x': x,
            'pdf': pdf,
            'route_name': self.route_id,
            'data': data
        }
    def display_stop_pdf(self, stop_id):
        if stop_id not in self.stops:
            print(f"[SKIP PLOT] No fitted PDF for stop {stop_id} on route {self.route_id}")
            return

        stop = self.stops[stop_id]
        route_name = stop['route_name']
        x = stop['x']
        pdf = stop['pdf']
        data = stop['data']

        # plot empirical data histogram (empirical distribution)
        plt.hist(data, bins=30, density=True, alpha=0.5, label='Empirical Data', color='steelblue', edgecolor='black')
        # plot fitted log normal pdf
        plt.plot(x, pdf, 'r-', lw=2, label=f'{stop_id} Fitted Lognormal PDF')
        plt.legend()
        plt.grid(True, which='both', axis='both', linestyle='--', alpha=0.6)
        plt.xticks(np.arange(0, data.max() + 100, 100))      
        plt.xlabel('Break Duration (seconds)')
        plt.ylabel('Probability Density')
        plt.title(f'{route_name} - {stop_id} (Empirical vs Fitted PDF)')

    def generate_pdf_image(self, stop_id, output_dir='results'):
        if stop_id not in self.stops:
            print(f"[SKIP PLOT] No fitted data for stop {stop_id} on route {self.route_id}")
            return

        stop = self.stops[stop_id]
        route_name = stop['route_name']
        data = stop['data']
        data = data[data > 1e-3]  # remove zeros / near-zeros
        if len(data) < 3:
            print(f"[SKIP] Too few data points to plot for stop {stop_id} on route {route_name}")
            return

        # Range for the fitted curves
        x = np.linspace(data.min(), data.max(), 400)

        # fit log-normal pdf
        shape_l, loc_l, scale_l = stats.lognorm.fit(data, floc=0)
        pdf_l = stats.lognorm.pdf(x, shape_l, loc_l, scale_l)

        # fit gamma pdf
        shape_g, loc_g, scale_g = stats.gamma.fit(data, floc=0)
        pdf_g = stats.gamma.pdf(x, shape_g, loc_g, scale_g)

        # plot the empircal data and both fits
        plt.figure(figsize=(6, 4))
        plt.hist(data, bins=30, density=True, alpha=0.4,
                color='gray', edgecolor='black', label='Empirical Data')

        plt.plot(x, pdf_l, 'r-', lw=2, alpha=0.8, label='Lognormal fit')
        plt.plot(x, pdf_g, 'b--', lw=2, alpha=0.8, label='Gamma fit')

        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xscale('log')
        plt.xlabel('Break Duration (seconds)')
        plt.ylabel('Probability Density')
        plt.title(f'{route_name} – Stop {stop_id} (Lognormal vs Gamma)')

        # saving
        os.makedirs(output_dir, exist_ok=True)
        route_dir = os.path.join(output_dir, f'route_{route_name}')
        os.makedirs(route_dir, exist_ok=True)
        save_path = os.path.join(route_dir, f"compare_{route_name}_{stop_id}.png")

        if os.path.exists(save_path):
            print(f"[SKIP SAVE] File already exists → {save_path}")
            plt.close()
            return

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"[SAVED] Comparison plot (Lognormal vs Gamma) → {save_path}")



