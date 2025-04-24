import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy import signal
from scipy.interpolate import PchipInterpolator
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from scipy import stats
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LinearRegression


radius = 4
surf = radius*radius*np.pi

def find_exceed_under(arr, half, threshold=0.1):
    # Find the first index where value exceeds the threshold
    first_exceed_idx = next((i for i in range(len(arr)) if arr[i] > threshold), -1)

    # Find the max value index in the first half of the array
    max_first_half_idx = max(range(half), key=lambda i: arr[i], default=-1)
    
    # Find the first index where value goes under the threshold between max_first_half_idx and half
    first_under_idx = -1
    if max_first_half_idx != -1:
        first_under_idx = next((i for i in range(max_first_half_idx + 1, half) if arr[i] < threshold), -1)

    # Find the max value index in the second half of the array
    max_second_half_idx = max(range(half, len(arr)), key=lambda i: arr[i], default=-1)
    
    # Find the second index where value goes under the threshold after the max value of the second half
    second_under_idx = -1
    if max_second_half_idx != -1:
        second_under_idx = next((i for i in range(max_second_half_idx + 1, len(arr)) if arr[i] < threshold), -1)

    # Find the second index where value exceeds the threshold, after the first under and within the second half
    second_exceed_idx = -1
    if first_under_idx != -1:
        second_exceed_idx = next((i for i in range(half, len(arr)) if arr[i] > threshold), -1)

    return first_exceed_idx, first_under_idx, second_exceed_idx, second_under_idx

def find_max_index(arr):
    return np.argmax(arr)

def positive_area_under_curve(force, index1, index2):
    # Extract the portion of the force curve from index1 to index2
    force_section = force[index1:index2+1]
    
    # Set negative values to zero
    force_section = np.where(force_section > 0, force_section, 0)
    
    # Compute the area using the trapezoidal rule (ignores negative areas)
    area = np.trapezoid(force_section)
    
    return float(area)

def find_start_end(gap):
    gap = np.array(gap)

    # Find the first significant change (larger than 0.5)
    start = next((i for i in range(1, len(gap)) if abs(gap[i] - gap[i - 1]) > 0.5), 0)
    
    # Find the first "end" using backward iteration
    end = next((i for i in range(len(gap) - 1, 0, -1) if abs(gap[i] - gap[i - 1]) > 0.5), len(gap) - 1)
    
    # Now find the second "end" after the first one
    second_end = next((i for i in range(end - 1, 0, -1) if abs(gap[i] - gap[i - 1]) > 0.5), end)
    
    return start, second_end

def double_compression_data(file,plotting=False, gap_filter=False, force_filter=True, sig=2): 
    file_path = file
    df = pd.read_csv(file_path, sep="\t", skiprows=0)
    numeric_data = df.apply(pd.to_numeric, errors='coerce').to_numpy()
    
    #$both
    gap = numeric_data[:,4]
    force = numeric_data[:,3]
    time = df.index.to_numpy()  
    
    if force_filter:
        force = gaussian_filter1d(force, sigma=sig)
        
    if gap_filter:
        force = gaussian_filter1d(gap, sigma=sig)    

    if plotting:    
        fig, ax1 = plt.subplots(figsize=(4, 3), dpi=300)
        ax1.plot(gap, color='red', linewidth=2, label='Gap [μm]')
        ax1.set_xlabel("index") 
        ax1.set_ylabel("gap [μm]", color='red')
        ax1.tick_params(axis='y', labelcolor='red')
        ax1.invert_yaxis()
        ax2 = ax1.twinx()
        ax2.plot(force, color='black', linewidth=2, label='Force [N]')
        ax2.set_ylabel("force [N]", color='black')
        ax2.tick_params(axis='y', labelcolor='black')
    
    start, end = find_start_end(gap)
    
    half = start +(end-start)/2
    
    first_exceed_idx, first_under_idx, second_exceed_idx, second_under_idx = find_exceed_under(force, int(half), threshold=0.1)
    
    print("Start index:", start)
    print("End index:", end)
    
    print("first_exceed_idx:", first_exceed_idx)
    print("first_under_idx:", first_under_idx)
    print("second_exceed_idx:", second_exceed_idx)
    print("second_under_idx:", second_under_idx)
    
    force1 = force[first_exceed_idx: first_under_idx]
    force2 = force[second_exceed_idx: second_under_idx]
    max_index1 = int(find_max_index(force1))
    max_index2 = int(find_max_index(force2))
    
    A1 = positive_area_under_curve(force,first_exceed_idx,first_exceed_idx+max_index1)
    A2 = positive_area_under_curve(force,first_exceed_idx+max_index1,first_under_idx)
    A3 = positive_area_under_curve(force,second_exceed_idx,second_exceed_idx+max_index2)
    A4 = positive_area_under_curve(force,second_exceed_idx+max_index2,second_under_idx)
    
    t1 = float(time[first_exceed_idx+max_index1] - time[first_exceed_idx])
    t2 = float(time[second_exceed_idx+max_index2] - time[second_exceed_idx])
    
    F1 = float(force[first_exceed_idx+max_index1])
    F2 = float(force[second_exceed_idx+max_index2])
    
    if plotting: 
        fig, ax1 = plt.subplots(figsize=(4, 3), dpi=300)
        # Plot Gap on the primary y-axis
        ax1.plot(gap, color='red', linewidth=2, label='Gap [μm]')
        ax1.set_xlabel("index")  # Adjust x-label as needed
        ax1.set_ylabel("gap [μm]", color='red')
        ax1.tick_params(axis='y', labelcolor='red')
        ax1.invert_yaxis()
        ax1.set_xlim([start, end])
        # Create a secondary y-axis for Force
        ax2 = ax1.twinx()
        ax2.plot(force, color='black', linewidth=2, label='Force [N]')
        ax2.set_ylabel("force [N]", color='black')
        ax2.tick_params(axis='y', labelcolor='black')
        
        
        fig, ax1 = plt.subplots(figsize=(4, 3), dpi=300)
        
        # Plot Gap on the primary y-axis
        ax1.plot(gap, color='red', linewidth=2, label='Gap [μm]')
        ax1.set_xlabel("index")  # Adjust x-label as needed
        ax1.set_ylabel("gap [μm]", color='red')
        ax1.tick_params(axis='y', labelcolor='red')
        ax1.invert_yaxis()
        ax1.set_xlim([start, half])
        # Create a secondary y-axis for Force
        ax2 = ax1.twinx()
        ax2.plot(force, color='black', linewidth=2, label='Force [N]')
        ax2.set_ylabel("force [N]", color='black')
        ax2.tick_params(axis='y', labelcolor='black')
    
    strain = 0.5
    stiffness = (F1/surf)/strain*1000
    hardness = F1
    cohesiveness = (A3+A4)/(A1+A2)
    springiness = t2/t1
    resilience = A2/A1
    chewiness = F1*(A3+A4)/(A1+A2)*t2/t1
    
    print('------------------')
    print(file)
    print('------------------')
    #print(f"stiffness = {stiffness:.3f} kPa")
    print('stiffness = ',stiffness, 'kPa')
    print('hardness = ',hardness, 'N')
    print('cohesiveness = ',cohesiveness, '')
    print('springiness = ',springiness, '')
    print('resilience = ',resilience, '')
    print('chewiness = ',chewiness, 'N')
    
    gap_red = gap[start:end]
    force_red = force[start:end]
    time_red = time[start:end]
    
    return stiffness,hardness,cohesiveness,springiness,resilience,chewiness


def double_compression_process(file,plotting=False, gap_filter=False, force_filter=True, sig=2): 
    file_path = file
    df = pd.read_csv(file_path, sep="\t", skiprows=0)
    numeric_data = df.apply(pd.to_numeric, errors='coerce').to_numpy()
    
    #speed = 0.5
    #switch = 36
    #sample_freq = 244
    #first = 2*sample_freq*speed + switch
    #$both
    gap = numeric_data[:,4]
    force = numeric_data[:,3]
    time = df.index.to_numpy()  
    
    if force_filter:
        force = gaussian_filter1d(force, sigma=sig)
        
    if gap_filter:
        force = gaussian_filter1d(gap, sigma=sig)    
    if plotting:    
        fig, ax1 = plt.subplots(figsize=(4, 3), dpi=300)
        ax1.plot(gap, color='red', linewidth=2, label='Gap [μm]')
        ax1.set_xlabel("index") 
        ax1.set_ylabel("gap [μm]", color='red')
        ax1.tick_params(axis='y', labelcolor='red')
        ax1.invert_yaxis()
        ax2 = ax1.twinx()
        ax2.plot(force, color='black', linewidth=2, label='Force [N]')
        ax2.set_ylabel("force [N]", color='black')
        ax2.tick_params(axis='y', labelcolor='black')
    
    start, end = find_start_end(gap)
    
    half = start +(end-start)/2
    
    first_exceed_idx, first_under_idx, second_exceed_idx, second_under_idx = find_exceed_under(force, int(half), threshold=0.1)

    
    print("Start index:", start)
    print("End index:", end)

    print("first_exceed_idx:", first_exceed_idx)
    print("first_under_idx:", first_under_idx)
    print("second_exceed_idx:", second_exceed_idx)
    print("second_under_idx:", second_under_idx)
    
    force1 = force[first_exceed_idx: first_under_idx]
    force2 = force[second_exceed_idx: second_under_idx]
    max_index1 = int(find_max_index(force1))
    max_index2 = int(find_max_index(force2))
    
    A1 = positive_area_under_curve(force,first_exceed_idx,first_exceed_idx+max_index1)
    A2 = positive_area_under_curve(force,first_exceed_idx+max_index1,first_under_idx)
    A3 = positive_area_under_curve(force,second_exceed_idx,second_exceed_idx+max_index2)
    A4 = positive_area_under_curve(force,second_exceed_idx+max_index2,second_under_idx)
    
    t1 = float(time[first_exceed_idx+max_index1] - time[first_exceed_idx])
    t2 = float(time[second_exceed_idx+max_index2] - time[second_exceed_idx])
    
    F1 = float(force[first_exceed_idx+max_index1])
    F2 = float(force[second_exceed_idx+max_index2])
    
    if plotting: 
        fig, ax1 = plt.subplots(figsize=(4, 3), dpi=300)
        # Plot Gap on the primary y-axis
        ax1.plot(gap, color='red', linewidth=2, label='Gap [μm]')
        ax1.set_xlabel("index")  # Adjust x-label as needed
        ax1.set_ylabel("gap [μm]", color='red')
        ax1.tick_params(axis='y', labelcolor='red')
        ax1.invert_yaxis()
        ax1.set_xlim([start, end])
        # Create a secondary y-axis for Force
        ax2 = ax1.twinx()
        ax2.plot(force, color='black', linewidth=2, label='Force [N]')
        ax2.set_ylabel("force [N]", color='black')
        ax2.tick_params(axis='y', labelcolor='black')
        
        
        fig, ax1 = plt.subplots(figsize=(4, 3), dpi=300)
        
        # Plot Gap on the primary y-axis
        ax1.plot(gap, color='red', linewidth=2, label='Gap [μm]')
        ax1.set_xlabel("index")  # Adjust x-label as needed
        ax1.set_ylabel("gap [μm]", color='red')
        ax1.tick_params(axis='y', labelcolor='red')
        ax1.invert_yaxis()
        ax1.set_xlim([start, half])
        # Create a secondary y-axis for Force
        ax2 = ax1.twinx()
        ax2.plot(force, color='black', linewidth=2, label='Force [N]')
        ax2.set_ylabel("force [N]", color='black')
        ax2.tick_params(axis='y', labelcolor='black')
    
    strain = 0.5
    stiffness = (F1/surf)/strain*1000
    hardness = F1
    cohesiveness = (A3+A4)/(A1+A2)
    springiness = t2/t1
    resilience = A2/A1
    chewiness = F1*(A3+A4)/(A1+A2)*t2/t1
    
    print('------------------')
    print(file)
    print('------------------')
    #print(f"stiffness = {stiffness:.3f} kPa")
    print('stiffness = ',stiffness, 'kPa')
    print('hardness = ',hardness, 'N')
    print('cohesiveness = ',cohesiveness, '')
    print('springiness = ',springiness, '')
    print('resilience = ',resilience, '')
    print('chewiness = ',chewiness, 'N')
    
    gap_red = gap[start:end]
    force_red = force[start:end]
    time_red = time[start:end]
    
    return gap_red, force_red
    
#%%
gap_test, force_test = double_compression_process('double-compression-fast-outplane-1.txt',True, False,True,1)

#%%
# Helper function to process and store curves
def process_force_curves(speed_label, color, filenames):
    force_curves = []
    for filename in filenames:
        gap_reduction, force_reduction = double_compression_process(filename)
        force_curves.append(force_reduction)

    # Trim to shortest curve length
    min_length = min(len(curve) for curve in force_curves)
    trimmed_curves = np.array([curve[:min_length] for curve in force_curves])

    # Calculate mean and standard deviation
    mean_force = np.mean(trimmed_curves, axis=0)
    std_force = np.std(trimmed_curves, axis=0)

    # Time axis
    time = np.arange(min_length) / 244  # Time in seconds

    return time, mean_force, std_force, color, speed_label


# Colors for Combined Plots for Each Speed
col1 = 'blue'  # Color for outplane data
col2 = 'red'   # Color for inplane data
col1 = '#843C0C'  # Color for outplane data
col2 = '#ED7D31'   # Color for inplane data

# Define filenames for each condition
slowestslow_filenames_outplane = [f'double-compression-slowestslow-outplane-{i}.txt' for i in range(1, 11)] 
slowest_filenames_outplane = [f'double-compression-slowest-outplane-{i}.txt' for i in range(1, 11)]
slow_filenames_outplane = [f'double-compression-slow-outplane-{i}.txt' for i in range(1, 11)]
medium_filenames_outplane = [f'double-compression-outplane-{i}.txt' for i in range(1, 11)]
fast_filenames_outplane = [f'double-compression-fast-outplane-{i}.txt' for i in range(1, 11)]

# Additional in-plane files
slowestslow_filenames_inplane = [f'double-compression-slowestslow-inplane-{i}.txt' for i in range(1, 11)]
slowest_filenames_inplane = [f'double-compression-slowest-inplane-{i}.txt' for i in range(1, 11)]
slow_filenames_inplane = [f'double-compression-slow-inplane-{i}.txt' for i in range(1, 11)]
medium_filenames_inplane = [f'double-compression-inplane-{i}.txt' for i in range(1, 11)]
fast_filenames_inplane = [f'double-compression-fast-inplane-{i}.txt' for i in range(1, 11)]

slowestslow_data_outplane = process_force_curves('0.0025/s', (132/255, 60/255, 12/255, 1.0), slowestslow_filenames_outplane)  # Darkest
slowest_data_outplane = process_force_curves('0.0250/s', (197/255, 90/255, 17/255, 1.0), slowest_filenames_outplane)  # Darker
slow_data_outplane = process_force_curves('0.2500/s', (237/255, 125/255, 49/255, 1.0), slow_filenames_outplane)  # Medium
medium_data_outplane = process_force_curves('0.5000/s', (244/255, 177/255, 131/255, 1.0), medium_filenames_outplane)  # Lighter
fast_data_outplane = process_force_curves('1.0000/s', (248/255, 203/255, 173/255, 1.0), fast_filenames_outplane)  # Lightest

slowestslow_data_inplane = process_force_curves('0.0025/s', (132/255, 60/255, 12/255, 1.0), slowestslow_filenames_inplane)  # Darkest
slowest_data_inplane = process_force_curves('0.0250/s', (197/255, 90/255, 17/255, 1.0), slowest_filenames_inplane)  # Darker
slow_data_inplane = process_force_curves('0.2500/s', (237/255, 125/255, 49/255, 1.0), slow_filenames_inplane)  # Medium
medium_data_inplane = process_force_curves('0.5000/s', (244/255, 177/255, 131/255, 1.0), medium_filenames_inplane)  # Lighter
fast_data_inplane = process_force_curves('1.0000/s', (248/255, 203/255, 173/255, 1.0), fast_filenames_inplane)  # Lightest

# Combined Plots for Each Speed with Blue and Red
for (outplane_data, inplane_data, title) in [
    (slowestslow_data_outplane, slowestslow_data_inplane, "double compression - 0.25%/s (n=10)"),
    (slowest_data_outplane, slowest_data_inplane, "double compression - 2.5%/s (n=10)"),
    (slow_data_outplane, slow_data_inplane, "double compression - 25%/s (n=10)"),
    (medium_data_outplane, medium_data_inplane, "double compression - 50%/s (n=10)"),
    (fast_data_outplane, fast_data_inplane, "double compression - 100%/s (n=10)")
]:
    fig, ax = plt.subplots(figsize=(5, 4), dpi=300)
    # Plot inplane data in red
    time, mean_force, std_force, _, label = inplane_data
    ax.plot(time, mean_force, color=col2, linestyle='-', linewidth=3, label=f'{label}')
    ax.fill_between(time, mean_force - std_force, mean_force + std_force, color=col2, alpha=0.3)
    # Plot outplane data in blue
    time, mean_force, std_force, _, label = outplane_data
    ax.plot(time, mean_force, color=col1, linestyle='-', linewidth=3, label=f'{label}')
    ax.fill_between(time, mean_force - std_force, mean_force + std_force, color=col1, alpha=0.3)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("force [N]")
    ax.legend(loc='best', fontsize=8)
    plt.title(title)
    plt.ylim(bottom=0) 
    # plt.xticks([])
    # plt.yticks([])
    # plt.gca().spines['top'].set_visible(False)
    # plt.gca().spines['right'].set_visible(False)
    # plt.gca().spines['left'].set_visible(False)
    # plt.gca().spines['bottom'].set_visible(False)
    plt.tight_layout()
    plt.show()

        
# Combined Plot with Uniform Time Axis (Out-plane data)
fig, ax1 = plt.subplots(figsize=(5, 4), dpi=300)
for data in [fast_data_outplane, medium_data_outplane, slow_data_outplane, slowest_data_outplane, slowestslow_data_outplane]:
    time, mean_force, std_force, color, label = data
    time = np.linspace(0, 1, len(time))  # Ensuring uniform time axis
    ax1.plot(time, mean_force, color=color, linewidth=3, label=f'{label}')
    #ax1.fill_between(time, mean_force - std_force, mean_force + std_force, color=color, alpha=0.3)
ax1.set_xlabel("relative time [-]")
ax1.set_ylabel("force [N]")
ax1.legend(loc='best', fontsize=8)
plt.ylim([0, 9])
plt.xlim([0, 1])
# plt.xticks(ticks=plt.xticks()[0], labels=[])
# plt.yticks(ticks=plt.yticks()[0], labels=[])
# plt.xticks([])
# plt.yticks([])
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
# plt.gca().spines['left'].set_visible(False)
# plt.gca().spines['bottom'].set_visible(False)
plt.tight_layout()
plt.title("double compression : out-plane (n=10)")
plt.show()

# Combined Plot with Uniform Time Axis (In-plane data)
fig, ax2 = plt.subplots(figsize=(5, 4), dpi=300)
for time, mean_force, std_force, color, label in [fast_data_inplane, medium_data_inplane, slow_data_inplane, slowest_data_inplane, slowestslow_data_inplane]:
    time = np.linspace(0, 1, len(time))
    ax2.plot(time, mean_force, color=color, linewidth=3, label=f'{label}')
    #ax2.fill_between(time, mean_force - std_force, mean_force + std_force, color=color, alpha=0.3)
ax2.set_xlabel("relative time [-]")
ax2.set_ylabel("force [N]")
ax2.legend(loc='best', fontsize=8)
plt.tight_layout()
plt.ylim([0, 9])
plt.xlim([0, 1])
# plt.xticks(ticks=plt.xticks()[0], labels=[])
# plt.yticks(ticks=plt.yticks()[0], labels=[])
# plt.xticks([])
# plt.yticks([])
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
# plt.gca().spines['left'].set_visible(False)
# plt.gca().spines['bottom'].set_visible(False)
plt.title("double compression : in-plane (n=10)")
plt.show()

#%%
# List of filenames
medium_filenames_outplane = [f'double-compression-slow-outplane-{i}.txt' for i in range(1, 11)]
medium_filenames_inplane = [f'double-compression-slow-inplane-{i}.txt' for i in range(1, 11)]

def compute_stats(filenames):
    # Lists to store extracted values
    stiffness_vals = []
    hardness_vals = []
    cohesiveness_vals = []
    springiness_vals = []
    resilience_vals = []
    chewiness_vals = []

    # Process each file
    for filename in filenames:
        stiffness, hardness, cohesiveness, springiness, resilience, chewiness = double_compression_data(filename)
        stiffness_vals.append(stiffness)
        hardness_vals.append(hardness)
        cohesiveness_vals.append(cohesiveness)
        springiness_vals.append(springiness)
        resilience_vals.append(resilience)
        chewiness_vals.append(chewiness)

    # Compute mean and standard deviation
    stats_dict = {
        "stiffness": (np.mean(stiffness_vals), np.std(stiffness_vals)),
        "hardness": (np.mean(hardness_vals), np.std(hardness_vals)),
        "cohesiveness": (np.mean(cohesiveness_vals), np.std(cohesiveness_vals)),
        "springiness": (np.mean(springiness_vals), np.std(springiness_vals)),
        "resilience": (np.mean(resilience_vals), np.std(resilience_vals)),
        "chewiness": (np.mean(chewiness_vals), np.std(chewiness_vals))
    }

    return stats_dict, stiffness_vals, hardness_vals, cohesiveness_vals, springiness_vals, resilience_vals, chewiness_vals

# Compute stats for outplane and inplane data
outplane_stats, outplane_stiffness, outplane_hardness, outplane_cohesiveness, outplane_springiness, outplane_resilience, outplane_chewiness = compute_stats(medium_filenames_outplane)
inplane_stats, inplane_stiffness, inplane_hardness, inplane_cohesiveness, inplane_springiness, inplane_resilience, inplane_chewiness = compute_stats(medium_filenames_inplane)

# Function for Welch's t-test
def welch_t_test(group1, group2):
    t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)  # Welch's t-test
    return t_stat, p_val

# Function for Mann-Whitney U test
def mann_whitney_u_test(group1, group2):
    u_stat, p_val = stats.mannwhitneyu(group1, group2, alternative="two-sided")  # Mann-Whitney U test
    return u_stat, p_val

# Perform Welch's t-tests for continuous variables and Mann-Whitney U test for bounded variables
for param, (outplane_mean, outplane_std) in outplane_stats.items():
    inplane_mean, inplane_std = inplane_stats[param]
    
    if param in ["stiffness", "hardness", "chewiness"]:  # Welch’s t-test for continuous data
        t_stat, p_val = welch_t_test(eval(f'outplane_{param}'), eval(f'inplane_{param}'))
    else:  # Mann-Whitney U test for bounded data
        u_stat, p_val = mann_whitney_u_test(eval(f'outplane_{param}'), eval(f'inplane_{param}'))

    # Print results
    print(f"\n{param.capitalize()} Comparison:")
    print(f"Outplane: {outplane_mean:.2f} ± {outplane_std:.2f}")
    print(f"Inplane: {inplane_mean:.2f} ± {inplane_std:.2f}")
    
    if param in ["stiffness", "hardness", "chewiness"]:
        print(f"T-statistic: {t_stat:.2f}, P-value: {p_val:.4f}")
    else:
        print(f"U-statistic: {u_stat:.2f}, P-value: {p_val:.4f}")
    
    if p_val < 0.05:
        print(f"Result: Significant difference (p < 0.05)\n")
    else:
        print(f"Result: No significant difference (p >= 0.05)\n")

#%% compression all
# Constants
radius = 4
surf = radius ** 2 * np.pi
col1 = 'blue'  # Color for outplane data
col2 = 'red'   # Color for inplane data
col1 = '#843C0C'  # Color for outplane data
col2 = '#ED7D31'   # Color for inplane data

# Function to calculate stiffness using linear regression with intercept fixed at (0,0)
def calculate_stiffness(strain_curves, stress_curves):
    stiffness_values = []
    
    for strain, stress in zip(strain_curves, stress_curves):
        x_values = 1 - strain
        y_values = stress
        
        # Exclude invalid points where (1 - strain) <= 0
        valid_indices = x_values > 0
        x_values = x_values[valid_indices].reshape(-1, 1)
        y_values = y_values[valid_indices]
        
        # Perform linear regression with intercept set to 0
        model = LinearRegression(fit_intercept=False)
        model.fit(x_values, y_values)
        
        # The stiffness is the slope of the regression line
        stiffness_values.append(model.coef_[0])
    
    return np.array(stiffness_values)

# Load outplane data (experiments 1 to 11)
out_strain_curves = []
out_stress_curves = []
for i in range(1, 12):
    file_name = f'compression-outplane-{i}.xls'
    data = pd.read_excel(file_name, sheet_name=1)
    data_num = data.iloc[2:].to_numpy(dtype=np.float64)
    strain = data_num[:, 4] / data_num[0, 4]
    stress = data_num[:, 3] / surf * 1000
    out_strain_curves.append(strain)
    out_stress_curves.append(stress)

# Load inplane data (experiments 1 to 11)
in_strain_curves = []
in_stress_curves = []
for i in range(1, 12):
    file_name = f'compression-inplane-{i}.xls'
    data = pd.read_excel(file_name, sheet_name=1)
    data_num = data.iloc[2:].to_numpy(dtype=np.float64)
    strain = data_num[:, 4] / data_num[0, 4]
    stress = data_num[:, 3] / surf * 1000
    in_strain_curves.append(strain)
    in_stress_curves.append(stress)

out_stiffness_values = calculate_stiffness(out_strain_curves, out_stress_curves)
in_stiffness_values = calculate_stiffness(in_strain_curves, in_stress_curves)

# Trim all curves to the shortest length
min_length_out = min(len(curve) for curve in out_strain_curves)
min_length_in = min(len(curve) for curve in in_strain_curves)
min_length = min(min_length_out, min_length_in)

# Trim strains and stresses for both outplane and inplane
trimmed_out_strains = [curve[:min_length] for curve in out_strain_curves]
trimmed_out_stresses = [curve[:min_length] for curve in out_stress_curves]

trimmed_in_strains = [curve[:min_length] for curve in in_strain_curves]
trimmed_in_stresses = [curve[:min_length] for curve in in_stress_curves]

# Compute mean and standard deviation
out_stiffness_mean = np.mean(out_stiffness_values)
out_stiffness_std = np.std(out_stiffness_values, ddof=1)

in_stiffness_mean = np.mean(in_stiffness_values)
in_stiffness_std = np.std(in_stiffness_values, ddof=1)

# Perform Welch's t-test for stiffness comparison (assuming function is implemented elsewhere)
t_stat_stiffness, p_val_stiffness = welch_t_test(out_stiffness_values, in_stiffness_values)

# Print results
print("Stiffness Comparison Results:")
print(f"Outplane Stiffness: Mean = {out_stiffness_mean:.3f}, Std Dev = {out_stiffness_std:.3f}")
print(f"Inplane Stiffness: Mean = {in_stiffness_mean:.3f}, Std Dev = {in_stiffness_std:.3f}")
print(f"Welch's t-test: p-value = {p_val_stiffness:.3f}")


# Calculate mean and std for stress and strain
mean_out_stress = np.mean(trimmed_out_stresses, axis=0)
mean_out_strain = np.mean(trimmed_out_strains, axis=0)
std_out_stress = np.std(trimmed_out_stresses, axis=0)

mean_in_stress = np.mean(trimmed_in_stresses, axis=0)
mean_in_strain = np.mean(trimmed_in_strains, axis=0)
std_in_stress = np.std(trimmed_in_stresses, axis=0)

# Plotting
plt.figure(figsize=(4, 3), dpi=300)
plt.plot(mean_out_strain, mean_out_stress - mean_out_stress[0], color=col1, linewidth=3, label='cross-plane')
plt.fill_between(mean_out_strain, (mean_out_stress - std_out_stress) - mean_out_stress[0], (mean_out_stress + std_out_stress) - mean_out_stress[0],
                 color=col1, alpha=0.3, label='')
plt.plot(mean_in_strain, mean_in_stress - mean_out_stress[0], color=col2, linewidth=3, label='in-plane')
plt.fill_between(mean_in_strain, (mean_in_stress - std_in_stress) - mean_out_stress[0], (mean_in_stress + std_in_stress) - mean_out_stress[0],
                 color=col2, alpha=0.3, label='')

xticks = np.round(np.linspace(min(mean_out_strain.min(), mean_in_strain.min()), 
                              max(mean_out_strain.max(), mean_in_strain.max()), 10), 1)
plt.xticks(xticks)
plt.gca().invert_xaxis()
# plt.xticks([])
# plt.yticks([])
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
# plt.gca().spines['left'].set_visible(False)
# plt.gca().spines['bottom'].set_visible(False)
plt.legend()
plt.xlabel("stretch [-]")
plt.xlim([1.0, 0.8])
plt.ylim([0, 10])
plt.ylabel("stress [kPa]")
plt.title("compression (n=10)")
plt.tight_layout()
plt.show()


#%% compression 50%
# Constants
radius = 4
surf = radius ** 2 * np.pi
col1 = 'blue'  # Color for outplane data
col2 = 'red'   # Color for inplane data
col1 = '#843C0C'  # Color for outplane data
col2 = '#ED7D31'   # Color for inplane data

# Load outplane data (experiments 1 to 11)
out_strain_curves = []
out_stress_curves = []
for i in range(1, 11):
    file_name = f'double-compression-slowestslow-outplane-{i}.xls'
    data = pd.read_excel(file_name, sheet_name=1)
    data_num = data.iloc[2:].to_numpy(dtype=np.float64)
    strain = data_num[:, 4] / data_num[0, 4]
    stress = data_num[:, 3] / surf * 1000
    out_strain_curves.append(strain)
    out_stress_curves.append(stress)

# Load inplane data (experiments 1 to 11)
in_strain_curves = []
in_stress_curves = []
for i in range(1, 11):
    file_name = f'double-compression-slowestslow-inplane-{i}.xls'
    data = pd.read_excel(file_name, sheet_name=1)
    data_num = data.iloc[2:].to_numpy(dtype=np.float64)
    strain = data_num[:, 4] / data_num[0, 4]
    stress = data_num[:, 3] / surf * 1000
    in_strain_curves.append(strain)
    in_stress_curves.append(stress)

# Trim all curves to the shortest length
min_length_out = min(len(curve) for curve in out_strain_curves)
min_length_in = min(len(curve) for curve in in_strain_curves)
min_length = min(min_length_out, min_length_in)

# Trim strains and stresses for both outplane and inplane
trimmed_out_strains = [curve[:min_length] for curve in out_strain_curves]
trimmed_out_stresses = [curve[:min_length] for curve in out_stress_curves]

trimmed_in_strains = [curve[:min_length] for curve in in_strain_curves]
trimmed_in_stresses = [curve[:min_length] for curve in in_stress_curves]

# Calculate mean and std for stress and strain
mean_out_stress = np.mean(trimmed_out_stresses, axis=0)
mean_out_strain = np.mean(trimmed_out_strains, axis=0)
std_out_stress = np.std(trimmed_out_stresses, axis=0)

mean_in_stress = np.mean(trimmed_in_stresses, axis=0)
mean_in_strain = np.mean(trimmed_in_strains, axis=0)
std_in_stress = np.std(trimmed_in_stresses, axis=0)

# Plotting
plt.figure(figsize=(4, 3), dpi=300)
plt.plot(mean_out_strain, mean_out_stress - mean_out_stress[0], color=col1, linewidth=3, label='cross-plane')
plt.fill_between(mean_out_strain, (mean_out_stress - std_out_stress) - mean_out_stress[0], (mean_out_stress + std_out_stress) - mean_out_stress[0],
                 color=col1, alpha=0.3, label='')
plt.plot(mean_in_strain, mean_in_stress - mean_out_stress[0], color=col2, linewidth=3, label='in-plane')
plt.fill_between(mean_in_strain, (mean_in_stress - std_in_stress) - mean_out_stress[0], (mean_in_stress + std_in_stress) - mean_out_stress[0],
                 color=col2, alpha=0.3, label='')

xticks = np.round(np.linspace(min(mean_out_strain.min(), mean_in_strain.min()), 
                              max(mean_out_strain.max(), mean_in_strain.max()), 10), 1)
plt.xticks(xticks)
plt.gca().invert_xaxis()
# plt.xticks([])
# plt.yticks([])
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
# plt.gca().spines['left'].set_visible(False)
# plt.gca().spines['bottom'].set_visible(False)
plt.legend()
plt.xlabel("stretch [-]")
plt.xlim([1.0, 0.5])
plt.ylim([0, 27.5])
plt.ylabel("stress [kPa]")
plt.title("quasi-static compression (n=10)")
plt.tight_layout()
plt.show()


#%% shear all
# Function to calculate stiffness using linear regression with intercept fixed at (0,0)
def calculate_stiffness(strain_curves, stress_curves):
    stiffness_values = []
    
    for strain, stress in zip(strain_curves, stress_curves):
        x_values = strain
        y_values = stress
        
        # Exclude invalid points where (1 - strain) <= 0
        valid_indices = x_values > 0
        x_values = x_values[valid_indices].reshape(-1, 1)
        y_values = y_values[valid_indices]
        
        # Perform linear regression with intercept set to 0
        model = LinearRegression(fit_intercept=False)
        model.fit(x_values, y_values)
        
        # The stiffness is the slope of the regression line
        stiffness_values.append(model.coef_[0])
    
    return 3*np.array(stiffness_values)

# Define color variables for plotting
col1 = 'blue'  # Color for outplane data
col2 = 'red'   # Color for inplane data
col1 = '#843C0C'  # Color for outplane data
col2 = '#ED7D31'   # Color for inplane data

# Load outplane shear data (experiments 1 to 11)
out_stress_curves = []
out_strain_curves = []
for i in range(1, 12):
    file_name = f'shear-outplane-{i}.xls'
    data = pd.read_excel(file_name, sheet_name=2)
    data_num = data.iloc[2:].to_numpy(dtype=np.float64)
    stress = data_num[:, 1] * data_num[:, 2] / 1000
    strain = np.linspace(0, 0.1, len(stress))
    out_stress_curves.append(stress)
    out_strain_curves.append(strain)

# Load inplane shear data (experiments 1 to 11)
in_stress_curves = []
in_strain_curves = []
for i in range(1, 12):
    file_name = f'shear-inplane-{i}.xls'
    data = pd.read_excel(file_name, sheet_name=2)
    data_num = data.iloc[2:].to_numpy(dtype=np.float64)
    stress = data_num[:, 1] * data_num[:, 2] / 1000
    strain = np.linspace(0, 0.1, len(stress))
    in_stress_curves.append(stress)
    in_strain_curves.append(strain)

out_stiffness_values = calculate_stiffness(out_strain_curves, out_stress_curves)
in_stiffness_values = calculate_stiffness(in_strain_curves, in_stress_curves)

# Trim all curves to the shortest length
min_length_out = min(len(curve) for curve in out_strain_curves)
min_length_in = min(len(curve) for curve in in_strain_curves)
min_length = min(min_length_out, min_length_in)

# Trim strains and stresses for both outplane and inplane
trimmed_out_strains = [curve[:min_length] for curve in out_strain_curves]
trimmed_out_stresses = [curve[:min_length] for curve in out_stress_curves]

trimmed_in_strains = [curve[:min_length] for curve in in_strain_curves]
trimmed_in_stresses = [curve[:min_length] for curve in in_stress_curves]


# Compute mean and standard deviation
out_stiffness_mean = np.mean(out_stiffness_values)
out_stiffness_std = np.std(out_stiffness_values, ddof=1)

in_stiffness_mean = np.mean(in_stiffness_values)
in_stiffness_std = np.std(in_stiffness_values, ddof=1)

# Perform Welch's t-test for stiffness comparison (assuming function is implemented elsewhere)
t_stat_stiffness, p_val_stiffness = welch_t_test(out_stiffness_values, in_stiffness_values)

# Print results
print("Stiffness Comparison Results:")
print(f"Outplane Stiffness: Mean = {out_stiffness_mean:.3f}, Std Dev = {out_stiffness_std:.3f}")
print(f"Inplane Stiffness: Mean = {in_stiffness_mean:.3f}, Std Dev = {in_stiffness_std:.3f}")
print(f"Welch's t-test: p-value = {p_val_stiffness:.3f}")


# Calculate mean and std for shear stress and strain
mean_out_stress = np.mean(trimmed_out_stresses, axis=0)
mean_out_strain = np.mean(trimmed_out_strains, axis=0)
std_out_stress = np.std(trimmed_out_stresses, axis=0)

mean_in_stress = np.mean(trimmed_in_stresses, axis=0)
mean_in_strain = np.mean(trimmed_in_strains, axis=0)
std_in_stress = np.std(trimmed_in_stresses, axis=0)

# Plotting
plt.figure(figsize=(4, 3), dpi=300)
plt.plot(mean_out_strain, mean_out_stress, color=col1, linewidth=3, label='cross-plane')
plt.fill_between(mean_out_strain, mean_out_stress - std_out_stress, mean_out_stress + std_out_stress,
                 color=col1, alpha=0.3, label='')
plt.plot(mean_in_strain, mean_in_stress, color=col2, linewidth=3, label='in-plane')
plt.fill_between(mean_in_strain, mean_in_stress - std_in_stress, mean_in_stress + std_in_stress,
                 color=col2, alpha=0.3, label='')

xticks = np.round(np.linspace(0, 0.1, 10), 1)
plt.xticks(xticks)
# plt.xticks([])
# plt.yticks([])
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
# plt.gca().spines['left'].set_visible(False)
# plt.gca().spines['bottom'].set_visible(False)
plt.legend()
plt.xlabel("shear strain [-]")
plt.xlim([0, 0.1])
plt.ylim([0, 1.5])
plt.ylabel("shear stress [kPa]")
plt.title("shear (n=10)")
plt.tight_layout()
plt.show()

#%%
#shear
# Function to compute statistics from excel file
def compute_stats(filename, sheet_name, start_row, end_row):
    data = pd.read_excel(filename, sheet_name=sheet_name)
    data_num = data.iloc[2:].to_numpy(dtype=np.float64)
    
    selected_data = data_num[start_row:end_row+1, :3]  # Selecting only first 3 columns
    mean_values = np.mean(selected_data, axis=0)
    std_values = np.std(selected_data, axis=0)
    
    return mean_values, std_values

# Sample names
inplane_samples = [f'frequency-inplane-{i}.xls' for i in range(1, 11)]
outplane_samples = [f'frequency-outplane-{i}.xls' for i in range(1, 11)]  # Omit outplane-3

# Lists to store inplane and outplane mean and std values
inplane_means = []
inplane_stds = []
outplane_means = []
outplane_stds = []
inplane_complex = []
outplane_complex = []

# Processing inplane samples
for i, sample in enumerate(inplane_samples, start=1):
    start_row, end_row = (13, 30) if i == 1 else (0, 17)
    mean_vals, std_vals = compute_stats(sample, sheet_name=2, start_row=start_row, end_row=end_row)
    mean_vals[:2] /= 1000  # Convert Pa to kPa
    std_vals[:2] /= 1000

    inplane_means.append(mean_vals)
    inplane_stds.append(std_vals)

    # Compute complex storage modulus G* = G' + iG''
    G_star = np.abs(mean_vals[0] + 1j * mean_vals[1])  # |G*|
    inplane_complex.append(G_star)

# Processing outplane samples
for i, sample in enumerate(outplane_samples, start=1):
    start_row, end_row = (13, 30) if i == 1 else (0, 17)
    mean_vals, std_vals = compute_stats(sample, sheet_name=2, start_row=start_row, end_row=end_row)
    mean_vals[:2] /= 1000  # Convert Pa to kPa
    std_vals[:2] /= 1000

    outplane_means.append(mean_vals)
    outplane_stds.append(std_vals)

    # Compute complex storage modulus G* = G' + iG''
    G_star = np.abs(mean_vals[0] + 1j * mean_vals[1])  # |G*|
    outplane_complex.append(G_star)

# Compute mean and std of complex modulus
complex_mean_inplane = np.mean(inplane_complex)
complex_std_inplane = np.std(inplane_complex)
complex_mean_outplane = np.mean(outplane_complex)
complex_std_outplane = np.std(outplane_complex)

# Welch's t-test for each parameter (Storage modulus, Loss modulus, Phase angle, and Complex modulus)
def welch_t_test(group1, group2):
    t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)  # Welch's t-test
    return t_stat, p_val

params = ["Storage modulus G'", "Loss modulus G''", "Phase angle δ", "Complex modulus |G*|"]
data_groups = [
    np.array([mean[0] for mean in inplane_means]),  # G'
    np.array([mean[1] for mean in inplane_means]),  # G''
    np.array([mean[2] for mean in inplane_means]),  # δ
    np.array(inplane_complex)  # |G*|
]
data_groups_out = [
    np.array([mean[0] for mean in outplane_means]),  # G'
    np.array([mean[1] for mean in outplane_means]),  # G''
    np.array([mean[2] for mean in outplane_means]),  # δ
    np.array(outplane_complex)  # |G*|
]

for i, param in enumerate(params):
    t_stat, p_val = welch_t_test(data_groups[i], data_groups_out[i])

    # Print results for each parameter
    print(f"{param} Comparison:")
    print(f"Inplane: {np.mean(data_groups[i]):.3f} ± {np.std(data_groups[i]):.3f} kPa")
    print(f"Outplane: {np.mean(data_groups_out[i]):.3f} ± {np.std(data_groups_out[i]):.3f} kPa")
    print(f"T-statistic: {t_stat:.2f}, P-value: {p_val:.4f}")
    if p_val < 0.05:
        print(f"Result: Significant difference (p < 0.05)\n")
    else:
        print(f"Result: No significant difference (p >= 0.05)\n")

# Print complex storage modulus values
print(f"Complex Storage Modulus (In-plane): {complex_mean_inplane:.3f} ± {complex_std_inplane:.3f} kPa")
print(f"Complex Storage Modulus (Out-plane): {complex_mean_outplane:.3f} ± {complex_std_outplane:.3f} kPa")

# Print Welch's t-test for |G*|
t_stat_Gstar, p_val_Gstar = welch_t_test(inplane_complex, outplane_complex)

print("\nComplex Storage Modulus |G*| Comparison:")
print(f"Inplane |G*|: {complex_mean_inplane:.3f} ± {complex_std_inplane:.3f} kPa")
print(f"Outplane |G*|: {complex_mean_outplane:.3f} ± {complex_std_outplane:.3f} kPa")
print(f"T-statistic: {t_stat_Gstar:.2f}, P-value: {p_val_Gstar:.4f}")
if p_val_Gstar < 0.05:
    print(f"Result: Significant difference (p < 0.05)\n")
else:
    print(f"Result: No significant difference (p >= 0.05)\n")



#%%
# Define the parameters in the correct order
ordered_parameters = [
    "soft", "hard", "brittle", "chewy", "gummy",
    "viscous", "springy", "sticky", "fibrous", "fatty",
    "moist", "meaty"
]

# Raw survey data for out-plane direction
outplane_data = {
    "soft": [0, 4, 0, 12, 0],
    "hard": [1, 10, 1, 4, 0],
    "brittle": [4, 8, 0, 3, 1],
    "chewy": [0, 0, 0, 6, 10],
    "gummy": [1, 1, 1, 8, 5],
    "viscous": [1, 2, 1, 10, 2],
    "springy": [3, 6, 0, 6, 1],
    "sticky": [1, 4, 0, 7, 4],
    "fibrous": [0, 1, 1, 7, 7],
    "fatty": [6, 7, 1, 2, 0],
    "moist": [0, 0, 1, 12, 3],
    "meaty": [2, 3, 2, 7, 2]
}

# Raw survey data for in-plane direction
inplane_data = {
    "soft": [0, 1, 0, 10, 5],
    "hard": [2, 12, 1, 1, 0],
    "brittle": [6, 6, 1, 1, 2],
    "chewy": [0, 2, 1, 9, 4],
    "gummy": [2, 4, 2, 4, 4],
    "viscous": [0, 5, 1, 9, 1],
    "springy": [5, 5, 2, 4, 0],
    "sticky": [0, 7, 3, 5, 1],
    "fibrous": [0, 0, 4, 4, 8],
    "fatty": [6, 6, 3, 1, 0],
    "moist": [0, 2, 1, 10, 3],
    "meaty": [1, 2, 4, 6, 3]
}

# Function to expand Likert scale responses into individual values
def expand_responses(data):
    expanded = []
    for rating, count in zip([1, 2, 3, 4, 5], data):
        expanded.extend([rating] * count)
    return expanded

# Compute mean ratings
def compute_mean_ratings(data):
    mean_ratings = {}
    for key, values in data.items():
        mean_ratings[key] = np.average([1, 2, 3, 4, 5], weights=values)
    return mean_ratings

outplane_means = compute_mean_ratings(outplane_data)
inplane_means = compute_mean_ratings(inplane_data)

# Perform Mann-Whitney U test for each parameter
p_values_mannwhitney = []
for param in ordered_parameters:
    outplane_responses = expand_responses(outplane_data[param])
    inplane_responses = expand_responses(inplane_data[param])
    
    # Perform the Mann-Whitney U test
    u_stat, p_val = mannwhitneyu(outplane_responses, inplane_responses, alternative='two-sided')
    p_values_mannwhitney.append(p_val)

# Compute mean and standard deviation for out-plane and in-plane per parameter
def compute_stats(data):
    stats = {}
    for key, values in data.items():
        responses = expand_responses(values)
        mean_val = np.mean(responses)
        std_val = np.std(responses, ddof=1)  # Use ddof=1 for sample standard deviation
        stats[key] = (mean_val, std_val)
    return stats

# Compute statistics for out-plane and in-plane data
outplane_stats = compute_stats(outplane_data)
inplane_stats = compute_stats(inplane_data)

# Print the mean and standard deviation for out-plane and in-plane per parameter
print(f"{'Parameter':<10} {'Outplane Mean':<15} {'Outplane Std':<15} {'Inplane Mean':<15} {'Inplane Std':<15}")
print("-" * 65)

for param in ordered_parameters:
    out_mean, out_std = outplane_stats[param]
    in_mean, in_std = inplane_stats[param]
    print(f"{param:<10} {out_mean:<15.4f} {out_std:<15.4f} {in_mean:<15.4f} {in_std:<15.4f}")

# Create a 4x3 plot layout with significance from Mann-Whitney test
fig, axes = plt.subplots(4, 3, figsize=(12, 12), dpi=300)
fig.suptitle("Out-plane vs In-plane Meati Results", fontsize=14, fontweight="bold")

# Define bar width and x positions
bar_width = 0.1  
x = np.linspace(0, 1, 10)


# Plot each parameter in its respective subplot
for ax, param, p_val in zip(axes.flatten(), ordered_parameters, p_values_mannwhitney):
    out_val = outplane_means[param]
    in_val = inplane_means[param]
    
    ax.bar(x[4], out_val, width=bar_width, color='#843C0C', label="Out")
    ax.bar(x[5], in_val, width=bar_width, color='#ED7D31', label="In")
    ax.set_ylim(1, 5)
    ax.set_title(param, fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["","","","","out","in","","","",""])
    ax.set_yticks([1,2,3,4,5])
    ax.tick_params(axis='x', length=0) 

    # Add p-value in the plot box
    significance_marker = "*" if p_val < 0.05 else ""
    ax.text(0.95, 4.5, f"p={p_val:.3f}{significance_marker}", fontsize=10, ha="right", color="black")

    # Show all subplot borders
    for spine in ax.spines.values():
        spine.set_visible(True)

# Adjust layout
plt.tight_layout()
plt.show()

