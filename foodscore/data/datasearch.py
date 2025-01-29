import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from foodscore import params
import cv2
import requests

# Get Data
def get_data(path = params.LOCAL_PATH, validation_split = 0.2, img_height = params.IMG_HEIGHT, img_width = params.IMG_WIDTH, bs = params.BATCH_SIZE):

    train_ds = image_dataset_from_directory(
                path,
                validation_split=validation_split,
                subset='training',
                seed=123,
                image_size=(img_height,img_width),
                batch_size=bs,
                shuffle=True,
                interpolation='bilinear',
                label_mode='categorical',
                )

    val_ds = image_dataset_from_directory(
                path,
                validation_split=validation_split,
                subset='validation',
                seed=123,
                image_size=(img_height,img_width),
                batch_size=bs,
                shuffle=True,
                interpolation='bilinear',
                label_mode='categorical',
                )


    val_batches = tf.data.experimental.cardinality(val_ds)
    test_ds = val_ds.take(val_batches // 2)
    val_ds = val_ds.skip(val_batches // 2)

    return train_ds, val_ds, test_ds

# Convert Image
def convert_test_image(img, img_height = params.IMG_HEIGHT, img_width = params.IMG_WIDTH):
    dims = (img_height, img_width)
    # img = cv2.imread(img)
    ## Resizing image
    img = cv2.resize(img, dims, interpolation=cv2.INTER_AREA)
    ## changing to BGR and normalizing
    img = tf.expand_dims(img, axis=0)
    return img

# Get Nutrients
def get_nutritions(food_name):
    nutrition_data = pd.DataFrame(columns=['name', 'protein', 'calcium', 'fat', 'carbohydrates', 'vitamins'])
    carbs = 0
    vitamin_a = 0
    vitamin_c = 0
    for name in food_name:
        try:
            url = "https://api.nal.usda.gov/fdc/v1/foods/search?api_key=d4D6dSOc81pTAOY2gsNZ0YhjkMlhStLJRoII5SJu&query=" + name
            response = requests.get(url)
            data = response.json()
            flatten_json = pd.json_normalize(data["foods"])
            if not flatten_json.empty:
                first_food = flatten_json.iloc[0]
                first_food_nutrition_list = first_food.foodNutrients

                data_to_concat = []  # moved inside try block
                for item in first_food_nutrition_list:
                    if item['nutrientNumber'] == "203":
                        protein = item['value']
                        continue
                    if item['nutrientNumber'] == "301":
                        calcium = item['value']
                        continue
                    if item['nutrientNumber'] == "204":
                        fat = item['value']
                        continue
                    if item['nutrientNumber'] == "205":
                        carbs = item['value']
                        continue
                    if item['nutrientNumber'] == "318":
                        vitamin_a = item['value']
                        continue
                    if item['nutrientNumber'] == "401":
                        vitamin_c = item['value']
                        continue

                vitamins = float(vitamin_a) + float(vitamin_c)
                data_to_concat.append({
                    'name': name,
                    'protein': protein or 'nan',
                    'calcium': calcium / 1000 if calcium else 'nan',
                    'fat': fat or 'nan',
                    'carbohydrates': carbs or 'nan',
                    'vitamins': vitamins / 1000 if vitamins else 'nan'
                })

                nutrition_data = pd.concat([nutrition_data, pd.DataFrame(data_to_concat)], ignore_index=True)
            else:
                print(f"Sorry {name}, not in database. But it's propably not healthy. I suggest green salad instead. :)")
        except KeyError:
            print(f"Sorry {name}, not in database. But it's propably not healthy. I suggest green salad instead. :)")

    return nutrition_data.to_dict()
