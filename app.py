import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
import torchvision.transforms as transforms
import torchvision.models as models


model = models.resnet50(pretrained=False)
num_classes = len(open('classes.txt').readlines())
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load('resnet50_0.497.pkl', map_location=torch.device('cpu')))
model.eval()


with open('classes.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = Image.open(file_path)
        image.thumbnail((250, 250))
        img_display = ImageTk.PhotoImage(image)

        
        img_label.config(image=img_display)
        img_label.image = img_display
        
        
        predict(image)

def predict(image):
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)

    _, predicted_idx = torch.max(output, 1)
    predicted_class = class_names[predicted_idx.item()]
    
    
    prediction_label.config(text=f"Prediction: {predicted_class}")

resnet50_0.497.pkl
root = tk.Tk()
root.title("Insect Identifier")


title_frame = tk.Frame(root, pady=10)
title_frame.pack()

title_label = tk.Label(title_frame, text="Insect Identifier", font=("Helvetica", 16, "bold"))
title_label.pack()


instruction_frame = tk.Frame(root, pady=5)
instruction_frame.pack()

instruction_label = tk.Label(instruction_frame, text="Upload an image of an insect to identify.", font=("Helvetica", 12))
instruction_label.pack()


image_frame = tk.Frame(root, padx=20, pady=20)
image_frame.pack()

img_label = tk.Label(image_frame)
img_label.pack()


button_frame = tk.Frame(root, pady=10)
button_frame.pack()

upload_btn = tk.Button(button_frame, text="Upload Image", command=upload_image, width=20)
upload_btn.pack()


prediction_frame = tk.Frame(root, pady=10)
prediction_frame.pack()


prediction_label = tk.Label(prediction_frame, text="Prediction: ", font=("Helvetica", 14))
prediction_label.pack()


root.mainloop()

