#%%

from tensorflow.keras.utils import img_to_array, load_img

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import argparse


#%%

# * Load the folders to process and the model to use for calculating the results

rawvideos_path = ('Data/20250602/RawVideos')

rawvideos_path = Path(rawvideos_path)
IM_SIZE = 256  # Example image size, adjust as needed
Batch_size = 32  # Adjust batch size as needed
finetuned_model = tf.keras.models.load_model('model/solid_model')

results_path = rawvideos_path.parent / 'Results'
results_path.mkdir(exist_ok=True, parents=True)
subfolders = [f for f in rawvideos_path.iterdir() if f.is_dir()]

#* All the follwoing code are the functions used to process the images and calculate the results

def median_based_on_predicted_labels(predicted_labels, smooth_pred, threshold=0.01):
    """
    Prints the median of smooth_pred for indices where predicted_labels == 1 if
    at least `threshold` fraction are 1, else for indices where predicted_labels == 0.
    """
    percent_ones = np.mean(predicted_labels)
    if percent_ones >= threshold:
        # Median for indices where predicted_labels == 1
        median_val = np.median(smooth_pred[predicted_labels.flatten() == 1])
        error = np.std(smooth_pred[predicted_labels.flatten() == 1])
        #print(f"{threshold*100:.0f}% or more are 1. Median of smooth_pred for 1s: {median_val}")
    else:
        # Median for indices where predicted_labels == 0
        median_val = np.median(smooth_pred[predicted_labels.flatten() == 0])
       # print(f" Median of smooth_pred for 0s: {median_val}")
        error = np.std(smooth_pred[predicted_labels.flatten() == 0])
    return median_val, error

def process_images_and_predict(image_paths, model, im_size=256, batch_size=32):
    """
    Load, resize, scale images and predict using the provided model.
    """
    images = []
    for img_path in image_paths:
        img = load_img(img_path, target_size=(im_size, im_size))
        img = img_to_array(img)
        img = img / 255.0  # Normalize
        images.append(img)
    
    # Convert to numpy array and batch
    images = np.array(images)
    
    # Predict
    predicted = model.predict(images, batch_size=batch_size, verbose=1)
    # 5. Threshold predictions to get binary labels
    predicted_labels = (predicted > 0.5).astype(int)
    window_size = 5 # Adjust for more/less smoothing
    predicted_flat = predicted.flatten()
    smooth_pred = np.convolve(predicted_flat, np.ones(window_size)/window_size, mode='same')
    median_val, error = median_based_on_predicted_labels(predicted_labels, predicted)

    
    return predicted_flat, smooth_pred, predicted_labels, median_val, error

#%%
records = []
valid_extensions = {".png", ".tif", ".jpeg", ".jpg"}

for subfolder in subfolders:
    for images_subfolder in [f for f in subfolder.iterdir() if f.is_dir()]:
        image_paths = [f for f in images_subfolder.iterdir() if f.suffix.lower() in valid_extensions]
        if not image_paths:
            continue
        predicted_flat, smooth_pred, predicted_labels, median_val, error = process_images_and_predict(
            image_paths, finetuned_model, IM_SIZE, Batch_size
        )
        records.append({
            'Image Path': str(images_subfolder),
            'Condition': str(images_subfolder.parent.name),
            'Speed [rpm]': str(images_subfolder.name),
            'Predicted Value': predicted_flat,
            'Smooth Predicted Value': smooth_pred,
            'Predicted Label': predicted_labels.flatten(),
            'Median Value': median_val,
            'Error': error,
        })

df = pd.DataFrame.from_records(records)
df['Speed [rpm]'] = df['Speed [rpm]'].str.extract(r'(\d+)').astype(float)
df = df.dropna(subset=['Speed [rpm]', 'Median Value'])
df_exploded = df.explode(['Predicted Value', 'Smooth Predicted Value', 'Predicted Label'])

# Save the DataFrame to a CSV file
df.to_csv(results_path / 'predictions_compact.csv', index=False)
df_exploded.to_csv(results_path / 'predictions_exploded.csv', index=False)
#%%
plt.figure(figsize=(10, 6))
plt.axhspan(0, 0.8, facecolor='red', alpha=0.2)
plt.axhspan(0.8, 1, facecolor='green', alpha=0.2)

sns.lineplot(
    data=df,
    x='Speed [rpm]',
    y='Median Value',
    hue='Condition',
    linewidth=4,
    marker='o',
    markersize=10,
    palette='Set1',
    
)
plt.xlabel('Speed [rpm]')
plt.ylabel('Mixing Degree [-]')  # Assuming 'Mixing Degree' is the median value
#plt.title('Median Value vs Speed [rpm] by Condition')
plt.legend(loc='best')
plt.grid()
plt.ylim(0, 1)  # Assuming median values are between 0 and 1
plt.tight_layout()
# Add text annotations for the regions
plt.text(df['Speed [rpm]'].min(), 0.4, 'Aggregated', color='black', fontsize=12, va='center', ha='left', alpha=1)
plt.text(df['Speed [rpm]'].min(), 0.9, 'Dispersed', color='black', fontsize=12, va='center', ha='left', alpha=1)

plt.savefig(results_path / 'mixing_degree_vs_speed.png')
plt.show()
# %% for debugging purposes
# Uncomment the following lines to visualize a specific prediction
# loc = 4

# predicted_flat,smooth_pred,predicted_labels, median_val = df.iloc[loc][['Predicted Value','Smooth Predicted Value','Predicted Label','Median Value']].values
# # # 6. Plot the predicted labels as a bar plot
# plt.figure(figsize=(10, 4))
# plt.plot(predicted_flat, label='Raw Prediction', marker='o', linestyle='--')
# plt.plot(smooth_pred, label='Smoothed Prediction', color='red')
# plt.plot(predicted_labels, label='Thresholded (0.5)', marker='x', linestyle='None')
# plt.title(f'Predicted Values (Raw, Smoothed, Thresholded) with Median: {median_val:.2f}')
# plt.xlabel('Sample Index')
# plt.ylabel('Prediction')
# plt.legend()

# plt.show()
# %%
