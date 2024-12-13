import random

import cv2
import numpy as np
from PIL import Image

import torch
from segment_anything import sam_model_registry, SamPredictor

import gradio as gr
from gradio_image_prompter import ImagePrompter

#######################################################################################################################################

sam_checkpoint = './weight/sam_vit_b_01ec64.pth'
model_type = "vit_b"
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)

#######################################################################################################################################

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def mask_generate(image, box, sam):

    output = {
        'Mask_With_Image' : [],
        'Mask' : []
    }

    x1, y1, x2, y2 = map(int, box)
    box = np.array(box) 

    predictor = SamPredictor(sam)
    predictor.set_image(image)

    input_point = None
    input_label = None
    color = np.array([30, 144, 255], dtype=np.uint8)

    with torch.no_grad():
        masks, scores, logits = predictor.predict(
            point_coords = input_point,
            point_labels = input_label,
            box = box,
            multimask_output = True, 
        )

    for mask in masks:
        h, w = mask.shape[-2:]
        mask_with_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        mask_with_image = cv2.addWeighted(image.astype(np.uint8), 0.75, mask_with_image.astype(np.uint8), 0.5, 0)
        mask_with_image = cv2.rectangle(mask_with_image, (x1, y1), (x2, y2), (0,0,0), 2)
        
        mask = np.array(mask, dtype=np.uint8)
        mask[mask > 0] = 1
        mask = mask * 255

        mask_with_image = Image.fromarray(mask_with_image)
        mask = Image.fromarray(mask)

        output['Mask_With_Image'].append(mask_with_image)
        output['Mask'].append(mask)

    return output


def preprocess(input):

    image = input['image']
    box = [input['points'][0][0], input['points'][0][1], input['points'][0][3], input['points'][0][4]]

    output = mask_generate(image, box, sam)

    for i, mask in enumerate(output['Mask']):

        mask_path = f"./mask_list/mask_{i}.png"
        mask.save(mask_path)

    return (output['Mask_With_Image'][0], output['Mask'][0], 
            output['Mask_With_Image'][1], output['Mask'][1], 
            output['Mask_With_Image'][2], output['Mask'][2])


#######################################################################################################################################

demo = gr.Interface(
    preprocess,
    ImagePrompter(show_label=True),
    [gr.Image(label="Mask1_with_image", format="png"), gr.Image(label="Mask1", format="png"),
     gr.Image(label="Mask2_with_image", format="png"), gr.Image(label="Mask2", format="png"),
     gr.Image(label="Mask3_with_image", format="png"), gr.Image(label="Mask3", format="png")],
    title="segment",
    description="Drawing a BBox and then Select Mask",
    allow_flagging = "auto"
)

if __name__=="__main__":
    demo.launch(share=True, server_name="0.0.0.0", server_port=8000)