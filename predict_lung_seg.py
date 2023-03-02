import torch
import torchvision
from src.data import blend
from src.models import PretrainedUNet
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def predict(model_name, origin_filename, path_save_image):
    
    '''
    model_name(Path): Path of the Unet model
    
    origin_filename(Path): Path of the input image
    
    path_save_image(Path): Path to save the image
    
    '''
    unet = PretrainedUNet(
    in_channels=1,
    out_channels=2, 
    batch_norm=True, 
    upscale_mode="bilinear"
    )

    unet.load_state_dict(torch.load(model_name, map_location=torch.device("cpu")))
    unet.to(device)
    unet.eval();


    origin = Image.open(origin_filename).convert("P")
    origin = torchvision.transforms.functional.resize(origin, (512, 512))
    origin = torchvision.transforms.functional.to_tensor(origin) - 0.5

    with torch.no_grad():
        origin = torch.stack([origin])
        origin = origin.to(device)
        out = unet(origin)
        softmax = torch.nn.functional.log_softmax(out, dim=1)
        out = torch.argmax(softmax, dim=1)
    
        origin = origin[0].to("cpu")
        out = out[0].to("cpu")
    
    plt.figure(figsize=(20, 10))

    pil_origin = torchvision.transforms.functional.to_pil_image(origin + 0.5).convert("RGB")

    plt.subplot(1, 2, 1)
    plt.title("origin image")
    plt.imshow(np.array(pil_origin))

    plt.subplot(1, 2, 2)
    plt.title("Processed image")
    plt.imshow(np.array(blend(origin, out)));

    plt.savefig(path_save_image)
    
    print("Prediction Finished !")
    
#origin_filename = Path("C://Users//JORDAN//Downloads//Compressed//dataset_jordan//train//PNEUMONIA//person6_bacteria_22.jpeg")
#path_save_image = Path("report.png")
#model_name = Path("unet-6v.pt")

#predict(model_name, origin_filename, path_save_image)