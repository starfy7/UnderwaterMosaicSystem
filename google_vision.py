# -*- coding: utf-8 -*-
"""
Created on 02/02/2020

@author: Zeina Antar
"""

import io, os
from numpy import random
from google.cloud import vision
#from Pillow_Utility import draw_borders, Image
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import cv2

def draw_borders(pillow_image, bounding, color, image_size, caption='', confidence_score=0):

    width, height = image_size
    draw = ImageDraw.Draw(pillow_image)
    draw.polygon([
        bounding.normalized_vertices[0].x *
        width, bounding.normalized_vertices[0].y * height,
        bounding.normalized_vertices[1].x *
        width, bounding.normalized_vertices[1].y * height,
        bounding.normalized_vertices[2].x *
        width, bounding.normalized_vertices[2].y * height,
        bounding.normalized_vertices[3].x * width, bounding.normalized_vertices[3].y * height], fill=None, outline=color)



    font = ImageFont.truetype(r'C:\Windows\Fonts\comic.ttf', 16)

    draw.text((bounding.normalized_vertices[0].x * width,
               bounding.normalized_vertices[0].y * height), font=font, text=caption, fill=color)

    # insert confidence score
    draw.text((bounding.normalized_vertices[0].x * width, bounding.normalized_vertices[0].y *
               height + 15), font=font, text='{0:.2f}%'.format(confidence_score), fill=color)

    return pillow_image

def google_vision(image):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"cogent-coyote-267015-0a92640bd8fd.json"
    client = vision.ImageAnnotatorClient()
    
    file_name = image
    image_path = os.path.join('', file_name)
    
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()
    
    image = vision.types.Image(content=content)
    response = client.object_localization(image=image)
    localized_object_annotations = response.localized_object_annotations
    
    pillow_image = Image.open(image_path)
    df = pd.DataFrame(columns=['name', 'score'])
    for obj in localized_object_annotations:
        df = df.append(
            dict(
                name=obj.name,
                score=obj.score
            ),
            ignore_index=True)
        
        
    pillow_image = Image.open(image_path)
    for obj in localized_object_annotations:
        r, g, b = random.randint(0, 255), random.randint(
            0, 255), random.randint(0, 255)
    
        draw_borders(pillow_image, obj.bounding_poly, (r, g, b),
                     pillow_image.size, obj.name, obj.score)
    
    print(df)
    pillow_image.show()
    
    
    
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()
    image = vision.types.Image(content=content)
    response1 = client.label_detection(image=image)
    labels = response1.label_annotations
    df1 = pd.DataFrame(columns=['description','score'])
    for label in labels:
        df1 = df1.append(
            dict(
                description= label.description,
                score= label.score,
            ),
            ignore_index=True)
            
    print(df1)
    return image

