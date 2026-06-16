#!/usr/bin/env python3
# Plotting utilities for visualizing MPPCA results

import matplotlib.pyplot as plt


def plot_clusters_map(df, k):
    cmap = plt.get_cmap('tab20', k)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(df['longitude'], df['latitude'], 
                          c=df['cluster'], cmap=cmap, s=2, alpha=0.6,
                          vmin=-0.5, vmax=k-0.5)
    cbar = plt.colorbar(scatter, ticks=range(k))
    cbar.set_label('Cluster ID')
    plt.title(f'MPPCA Cluster Map (M={k})')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.axis('equal')
    plt.show()


def plot_noise_ratios(noise_ratios):
    plt.figure(figsize=(8, 5))
    plt.bar(range(len(noise_ratios)), noise_ratios, color='crimson', alpha=0.7, edgecolor='black')
    plt.title('Noise Variance Ratio per Cluster')
    plt.xlabel('Cluster ID')
    plt.ylabel('Ratio (Noise / Total Variance)')
    plt.xticks(range(len(noise_ratios)))
    plt.show()


def plot_log_likelihoods(log_likelihoods):
    plt.figure(figsize=(8, 5))
    plt.plot(log_likelihoods, marker='o', color='darkgreen')
    plt.title('Log-Likelihood Convergence')
    plt.xlabel('Iteration')
    plt.ylabel('Log-Likelihood')
    plt.show()


def plot_magnetic_intensity(df):
    plt.figure(figsize=(12, 10))
    sc = plt.scatter(df['longitude'], df['latitude'], c=df['magnetics_final_microlevelled'], cmap='jet', s=2, alpha=1.0)
    plt.colorbar(sc, label='Magnetic Intensity (nT)')
    plt.title('Magnetic Intensity Map', fontsize=16)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()


def plot_laser_altitude_contour(df):
    plt.figure(figsize=(12, 10))
    cntr = plt.tricontourf(df['longitude'], df['latitude'], df['laser_alt'], levels=20, cmap='terrain')
    plt.colorbar(cntr, label='Laser Altitude (m)')
    plt.title('Laser Altitude Contour Map', fontsize=16)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()
