# Solid Suspension Processor GUI

This project provides a graphical user interface (GUI) for processing solid suspension images using a pre-trained deep learning model. The GUI is built with Tkinter and allows users to select input folders, model paths, and processing parameters, then runs the analysis and saves results.

## Features

- Select raw video/image folders and model directory via file dialogs
- Set batch size and image resize parameters
- Start processing with a single button click
- Displays status updates during processing
- Results and plots are saved automatically

## Requirements

- Python 3.8+
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn

Install dependencies with:

```sh
conda env create -f requirements.yaml
conda activate Solidsuspension_code_ORI
pip install -e .
```

## Usage

1. Clone this repository:

    ```sh
    https://github.com/WeheliyeHashi/Solid_suspension_ORI.git
    cd Solid_suspension_ORI
    ```

2. Run the GUI:

    ```sh
    ss_gui
    ```

3. In the GUI:
    - Browse and select the folder containing your raw videos/images.
    - Browse and select the folder containing your trained model.
    - Set the batch size and image resize value as needed.
    - Click "Run Processing" to start the analysis.

4. Results will be saved in a `Results` folder next to your raw videos directory.


5. Use the interface to select the necessary files and options, then start the processing.


## File Structure

- `ss_gui.py` - Main GUI application
- `Process_main_images_GUI.py` - Image processing and prediction logic

## Notes

- The model should be a Keras/TensorFlow model saved in the specified directory.
- Only image files with extensions `.png`, `.tif`, `.jpeg`, `.jpg` are processed.

## Model

The model for detecting the walls can be found in sharepoint: Bioprocessing/Documents/OBP-EMC- Engineering Mixing Characterisation/PIV/EMC-0061

---

For questions or issues, please open an issue on this repository.