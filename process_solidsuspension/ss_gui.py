#%%
from tkinter import Tk, Label, Entry, Button, IntVar, StringVar, filedialog
import os
from pathlib import Path
from Process_main_images_GUI import main_processor
#import process_solidsuspension as pss
#%%
class SolidSuspensionProcessorGUI:
    def __init__(self, master):
        self.master = master
        master.title("Solid Suspension Processor GUI")
        master.geometry("800x600")  # Set the size of the window

        self.raw_videos_path = StringVar()
        self.model_path = StringVar()
        self.batch_size = IntVar(value=32)
        self.image_size = IntVar(value=256)
        self.status_message = StringVar()  

        Label(master, text="Raw Videos Path:", font=("Helvetica", 12)).grid(row=0, column=0, sticky='w', padx=10, pady=5)
        Entry(master, textvariable=self.raw_videos_path, width=70).grid(row=0, column=1, padx=10, pady=5)
        Button(master, text="Browse", command=self.browse_raw_videos).grid(row=0, column=2, padx=10, pady=5)

        Label(master, text="Model Path:", font=("Helvetica", 12)).grid(row=1, column=0, sticky='w', padx=10, pady=5)
        Entry(master, textvariable=self.model_path, width=70).grid(row=1, column=1, padx=10, pady=5)
        Button(master, text="Browse", command=self.browse_model).grid(row=1, column=2, padx=10, pady=5)

        Label(master, text="Batch Size:", font=("Helvetica", 12)).grid(row=2, column=0, sticky='w', padx=10, pady=5)
        Entry(master, textvariable=self.batch_size).grid(row=2, column=1, padx=10, pady=5)

        Label(master, text="Image Re-size:", font=("Helvetica", 12)).grid(row=3, column=0, sticky='w', padx=10, pady=5)
        Entry(master, textvariable=self.image_size).grid(row=3, column=1, padx=10, pady=5)

        Button(master, text="Run Processing", command=self.run_processing).grid(row=4, columnspan=3, pady=20)
        Label(master, textvariable=self.status_message, font=("Helvetica", 10), fg="blue", wraplength=700, justify="left").grid(row=5, column=0, columnspan=3, padx=10, pady=10, sticky='w')
         # Add this status label for process updates
        self.status_label = Label(master, text="", font=("Helvetica", 12))
        self.status_label.grid(row=8, columnspan=3, pady=10)
    def browse_raw_videos(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.raw_videos_path.set(folder_selected)

    def browse_model(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.model_path.set(folder_selected)

    def run_processing(self):
        raw_videos = self.raw_videos_path.get()
        model = self.model_path.get()
        batch_size = self.batch_size.get()
        image_size = self.image_size.get()

        # Here you would add the logic to process the videos using the provided parameters
        status = f"Processing with:\nRaw Videos Path: {raw_videos}\nModel Path: {model}\nBatch Size: {batch_size}\nImage Size: {image_size}"
        self.status_message.set(status) 
        self.status_label.config(text="Analysis in process...", fg="red")
        self.master.update_idletasks()  
        main_processor(raw_videos, model, batch_size, image_size)  # Call the main processing function ...use pss later 
        self.status_label.config(text="Analysis completed", fg="green")

def main():
    root = Tk()
    gui = SolidSuspensionProcessorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()