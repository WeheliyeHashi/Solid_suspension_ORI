# %%
%matplotlib qt 
import numpy as np
from PIL import Image
from scipy.stats import entropy
import matplotlib.pyplot as plt

plt.rcParams['image.cmap'] = 'gray'
import skimage.io
from pathlib import Path
import cv2 
from skimage.color import rgb2gray

# %%

def _adpative_thresholding(img: np.uint8, blocksize=15, Constant=1):
    th = np.array(
            [
                cv2.adaptiveThreshold(
                    img,
                    255,
                    cv2.ADAPTIVE_THRESH_MEAN_C,
                    cv2.THRESH_BINARY,
                    blocksize,
                    Constant,
                )
                == 0
            ]
        )
    return np.squeeze(th)

def calculate_shannon_entropy(image_grid):
    # Flatten the 2D grid into a 1D array
    pixel_values = np.array(image_grid).flatten()

    # Calculate the histogram of pixel intensities
    #histogram, bin_edges = np.histogram(
     #   pixel_values, bins=256, range=(0, 256), density=True
    #)

    # Remove zeros (log2(0) is undefined) and calculate entropy using scipy
    #histogram = histogram[histogram > 0]
    pk = np.array([(len(pixel_values[pixel_values==0])/len(pixel_values))+1e-6, (len(pixel_values[pixel_values==1])/len(pixel_values))+1e-6])
    #pixel_values[pixel_values ==0] = 1e-6
    return entropy(
        pk, base=2
    )  # Shannon entropy with log base 2 or -np.sum(histogram * np.log2(histogram))


def subdivide_image(image, grid_size=(16, 16)):
    width, height = image.shape
    grids = []
    for i in range(0, width, grid_size[0]):
        row = []
        for j in range(0, height, grid_size[1]):
            # Crop the image into 16x16 grids
           # box = (i, j, i + grid_size[0], j + grid_size[1])
            grid = image[i: i + grid_size[0], j: j + grid_size[1]]
            row.append(grid)
        grids.append(row)
    return grids


def calculate_entropy_grid(image, grid_size=(16, 16)):
    # Load the image in grayscale mode
   # image = Image.open(image_path).convert("L")  # 'L' converts it to grayscale

    # Subdivide the image into grids
    grids = subdivide_image(image, grid_size)

    # Calculate Shannon entropy for each grid
    entropy_grid = np.zeros(
        (len(grids), len(grids[0]))
    )  # Create a grid to store entropy values
    for i, row in enumerate(grids):
        for j, grid in enumerate(row):
            entropy_grid[i, j] = calculate_shannon_entropy(grid)

    return entropy_grid



def _return_circle(x1, y1, x2, y2, x3, y3):
     # Calculate the midpoints of the line segments
    mid_x1_x2 = (x1 + x2) / 2
    mid_y1_y2 = (y1 + y2) / 2
    mid_x2_x3 = (x2 + x3) / 2
    mid_y2_y3 = (y2 + y3) / 2
    
    # Calculate the slope of the lines
    slope_12 = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')
    slope_23 = (y3 - y2) / (x3 - x2) if x3 != x2 else float('inf')
    
    # Perpendicular slopes
    perp_slope_12 = -1 / slope_12 if slope_12 != 0 else float('inf')
    perp_slope_23 = -1 / slope_23 if slope_23 != 0 else float('inf')
    
    # Solve the intersection of the two perpendicular bisectors (for the center)
    if perp_slope_12 == float('inf'):
        center_x = mid_x1_x2
        center_y = perp_slope_23 * (center_x - mid_x2_x3) + mid_y2_y3
    elif perp_slope_23 == float('inf'):
        center_x = mid_x2_x3
        center_y = perp_slope_12 * (center_x - mid_x1_x2) + mid_y1_y2
    else:
        center_x = (perp_slope_12 * mid_x1_x2 - perp_slope_23 * mid_x2_x3 + mid_y2_y3 - mid_y1_y2) / (perp_slope_12 - perp_slope_23)
        center_y = perp_slope_12 * (center_x - mid_x1_x2) + mid_y1_y2
    
    # Calculate the radius (distance from the center to any of the points)
    radius = np.sqrt((center_x - x1) ** 2 + (center_y - y1) ** 2)
    
    # Generate points to plot the circle
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_x = center_x + radius * np.cos(theta)
    circle_y = center_y + radius * np.sin(theta)
    print(center_x, center_y, radius)
  
    return center_x, center_y, radius


def plot_circle_from_points(x1, y1, x2, y2, x3, y3):
    # Calculate the midpoints of the line segments
    mid_x1_x2 = (x1 + x2) / 2
    mid_y1_y2 = (y1 + y2) / 2
    mid_x2_x3 = (x2 + x3) / 2
    mid_y2_y3 = (y2 + y3) / 2
    
    # Calculate the slope of the lines
    slope_12 = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')
    slope_23 = (y3 - y2) / (x3 - x2) if x3 != x2 else float('inf')
    
    # Perpendicular slopes
    perp_slope_12 = -1 / slope_12 if slope_12 != 0 else float('inf')
    perp_slope_23 = -1 / slope_23 if slope_23 != 0 else float('inf')
    
    # Solve the intersection of the two perpendicular bisectors (for the center)
    if perp_slope_12 == float('inf'):
        center_x = mid_x1_x2
        center_y = perp_slope_23 * (center_x - mid_x2_x3) + mid_y2_y3
    elif perp_slope_23 == float('inf'):
        center_x = mid_x2_x3
        center_y = perp_slope_12 * (center_x - mid_x1_x2) + mid_y1_y2
    else:
        center_x = (perp_slope_12 * mid_x1_x2 - perp_slope_23 * mid_x2_x3 + mid_y2_y3 - mid_y1_y2) / (perp_slope_12 - perp_slope_23)
        center_y = perp_slope_12 * (center_x - mid_x1_x2) + mid_y1_y2
    
    # Calculate the radius (distance from the center to any of the points)
    radius = np.sqrt((center_x - x1) ** 2 + (center_y - y1) ** 2)
    
    # Generate points to plot the circle
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_x = center_x + radius * np.cos(theta)
    circle_y = center_y + radius * np.sin(theta)
    print(center_x, center_y, radius)
    
    # Plot the circle and the points
    plt.plot(circle_x, circle_y, c = 'r', linewidth=4)
   # plt.scatter([x1, x2, x3], [y1, y2, y3], color='red', zorder=5)
    #plt.scatter(center_x, center_y, color='blue', zorder=5, label='Center')
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


# Example usage with three points
def plot_entropy_contour(entropy_grid,  image,  pts, grid_size=(16, 16), ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    (x1, y1), (x2, y2), (x3, y3) = pts[:3]
    center_x, center_y, radius = _return_circle(x1, y1, x2, y2, x3, y3)
    x = np.arange(0, entropy_grid.shape[1] * grid_size[0], grid_size[0])
    y = np.arange(0, entropy_grid.shape[0] * grid_size[1], grid_size[1])
    X, Y = np.meshgrid(x, y)
    x0, y0 = center_x, center_y
    R = radius
    mask = (X - x0)**2 + (Y - y0)**2 > R**2
    entropy_grid[mask] = np.nan
    average_entropy = np.nanmean(entropy_grid)
    print(f'this is the real value {average_entropy}')
    ax.imshow(image)
    contour = ax.contourf(X, Y, entropy_grid, cmap="jet", alpha=0.5, levels=20, v_min=0, vmax=1)
    cbar =plt.colorbar(contour, ax=ax, label="Shannon Entropy")

    # Plot the circle on the same axis
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_x = x0 + R * np.cos(theta)
    circle_y = y0 + R * np.sin(theta)
    ax.plot(circle_x, circle_y, c='r', linewidth=2)
    ax.axis('off')

def average_shannon_entropy_with_plot(image, org_image, pts, grid_size=(16, 16), ax=None):
    entropy_grid = calculate_entropy_grid(image, grid_size)
   # plot_entropy_contour(entropy_grid, org_image, pts, grid_size, ax=ax)
    average_entropy = np.nanmean(entropy_grid)
    return average_entropy


# %%


def _process_image(imgs, image_num=0, lower_blue=(0, 0, 100), upper_blue=(100, 100, 255)):
   
   
    img_rgb = imgs[image_num]



    # Create a mask for blue regions
    blue_mask = cv2.inRange(img_rgb, lower_blue, upper_blue)
    blue_mask = np.where(blue_mask == 255, 1, 0).astype(np.uint8)
    # Apply morphological operations to clean up the mask
    kernel = np.ones((1, 1), np.uint8)
    blue_cleaned = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
    #blue_cleaned = cv2.morphologyEx(blue_cleaned, cv2.MORPH_CLOSE, kernel)

    # Apply the mask to the original image
    result = cv2.bitwise_and(img_rgb, img_rgb, mask=blue_cleaned)
    return blue_cleaned, result, blue_mask

#%%
"""
input variables:
imgs: List of images to process

"""
image_num = 40
folder_path = Path('Data/100 ml -rock - Microcarriers/60 rpm')
img_names = sorted(list((folder_path).glob('*.tif')))


imgs = [((plt.imread(f))).astype(np.uint8) for f in img_names]
imgs1 = [(rgb2gray(skimage.io.imread(f)) * 255).astype(np.uint8) for f in img_names]

masked_blue_mask = imgs[image_num].copy()

# Display the image and allow manual interaction
fig, ax = plt.subplots()
ax.imshow(masked_blue_mask, cmap='gray')
ax.set_title("Click points to define mask area (Press Enter when done)")
plt.axis("off")

# Allow user to click on the image
pts = plt.ginput(n=-1, timeout=0)  # n=-1 means unlimited points
plt.close()
#%%
# Convert points to NumPy array
lower_blue = np.array([0, 0, 100])  # Lower bound for blue
upper_blue = np.array([80, 80, 255])  # Upper bound for blue)
blue_cleaned, result, blue_mask = _process_image(imgs, image_num,lower_blue, upper_blue)
pts = np.array(pts, dtype=np.int32)

fig, ax = plt.subplots(1,3, figsize=(15, 5))
ax[0].imshow(masked_blue_mask, cmap='gray')
ax[0].set_title("Masked Blue Particles")            
ax[0].axis("off")
ax[1].imshow(blue_mask, cmap='gray')
ax[1].set_title("Blue Mask")
ax[1].axis("off")

average_entropy = average_shannon_entropy_with_plot(blue_mask, masked_blue_mask,pts, grid_size=(16, 16), ax=ax[2])
ax[2].set_title(f"ASE: {average_entropy:.2f}")
ax[2].axis("off")
#%%
import tqdm
max_entropy= [1, 1, 0.91]

List_avg_entropy = []
for image_num in tqdm.tqdm(range(0,len(imgs),1), total=len(imgs)):
    blue_cleaned, result, blue_mask = _process_image(imgs, image_num,lower_blue, upper_blue)
    img_rgb = imgs[image_num]
    average_entropy = average_shannon_entropy_with_plot(blue_mask, img_rgb,pts, grid_size=(16, 16))
    List_avg_entropy.append(average_entropy)
    #plt.close()
    print(f"Average Shannon Entropy: {average_entropy}")
    plt.title(f"Average Shannon Entropy: {average_entropy}")
#%%
plt.figure()
plt.plot(np.array(List_avg_entropy))
plt.xlabel("Sample Index")
plt.ylabel("S/Smax")  
plt.ylim(0, 1)
#plt.xticks(np.arange(0, len(List_avg_entropy)/max(List_avg_entropy), 1)) 
plt.title("Average Shannon Entropy across Images")
plt.grid()






# %%
sample=150
t_cnn =0.135/sample
t_shannon = 221/sample
# Data for plotting
methods = ['Shannon entropy', 'CNN-based method']
times = [ t_shannon, t_cnn]

plt.figure(figsize=(6, 5))
bars = plt.bar(methods, times, color=['skyblue', 'salmon'])
plt.yscale('log')
plt.ylabel('Process time per image (s)')
plt.xlabel('Methods')
plt.title('Comparison of Process Time per Image')
plt.grid(axis='y', which='both', linestyle='--', linewidth=0.5)

# Optionally, add value labels on top of bars
for bar, time in zip(bars, times):
    plt.text(bar.get_x() + bar.get_width()/2, time, f'{time:.2e}', 
             ha='center', va='bottom', fontsize=10)

plt.show()
# %%
