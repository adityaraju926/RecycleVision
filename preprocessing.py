import os
import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from typing import Tuple, List, Dict, Optional
import json

recycling_categories = {
    'aerosol_cans': True,
    'aluminum_food_cans': True,
    'aluminum_soda_cans': True,
    'cardboard_boxes': True,
    'cardboard_packaging': True,
    'clothing': False,
    'coffee_grounds': False,
    'disposable_plastic_cutlery': False,
    'eggshells': False,
    'food_waste': False,
    'glass_beverage_bottles': True,
    'glass_cosmetic_containers': True,
    'glass_food_jars': True,
    'magazines': True,
    'newspaper': True,
    'office_paper': True,
    'paper_cups': False,
    'plastic_cup_lids': False,
    'plastic_detergent_bottles': True,
    'plastic_food_containers': True,
    'plastic_shopping_bags': False,
    'plastic_soda_bottles': True,
    'plastic_straws': False,
    'plastic_trash_bags': False,
    'plastic_water_bottles': True,
    'shoes': False,
    'steel_food_cans': True,
    'styrofoam_cups': False,
    'styrofoam_food_containers': False,
    'tea_bags': False
}

def get_categories(data_path="../data/images"):
    categories = []
    if os.path.exists(data_path):
        for item in os.listdir(data_path):
            if os.path.isdir(os.path.join(data_path, item)):
                categories.append(item)
    result = sorted(categories)

    return result

def load_image(image_path, target_size=(224, 224)):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    image = cv2.resize(image, target_size)
    image = image.astype(np.float32) / 255.0
    
    return image

def original_data(data_path="../data/images", target_size=(224, 224), max_samples_per_category=None):
    if not os.path.exists(data_path):
        alt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "images")
        if os.path.exists(alt_path):
            data_path = alt_path
    
    categories = get_categories(data_path)
    category_to_label = {cat: idx for idx, cat in enumerate(categories)}
    
    images = []
    labels = []
    
    for category in categories:
        category_path = os.path.join(data_path, category)
            
        image_files = []
        for subdir in ['default', 'real_world']:
            subdir_path = os.path.join(category_path, subdir)
            if os.path.exists(subdir_path):
                for file in os.listdir(subdir_path):
                    if file.lower().endswith('.png'):
                        image_files.append(os.path.join(subdir_path, file))
        
        if max_samples_per_category and len(image_files) > max_samples_per_category:
            image_files = image_files[:max_samples_per_category]
        
        for i, image_file in enumerate(image_files):
            image = load_image(image_file, target_size)
            if image is not None:
                images.append(image)
                labels.append(category_to_label[category])
    
    result_images = np.array(images)
    result_labels = np.array(labels)

    return result_images, result_labels

def create_data_generators(images, labels, test_size=0.2, validation_size=0.1, batch_size=32):
    n_samples = len(images)
    n_classes = len(np.unique(labels))
    min_samples_per_class = n_samples // n_classes
    
    if min_samples_per_class<3:
        test_size = min(0.1, 1.0/n_classes)
        validation_size = min(0.05, 0.5/n_classes)
    
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=test_size, random_state=42, stratify=labels)
    
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_size, random_state=42, stratify=y_train)
    
    train_datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
    
    val_datagen = ImageDataGenerator()
    test_datagen = ImageDataGenerator()
    
    train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
    val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)
    test_generator = test_datagen.flow(X_test, y_test, batch_size=batch_size)
    
    result = {'train': train_generator,'validation': val_generator,'test': test_generator,'train_data': (X_train, y_train),'val_data': (X_val, y_val),'test_data': (X_test, y_test)}

    return result

def convert_image(pil_image, target_size=(224, 224)):
    image = np.array(pil_image)
    
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    image = cv2.resize(image, target_size)
    image = image.astype(np.float32) / 255.0
    
    result = np.expand_dims(image, axis=0)
    
    return result

def preprocessed_info(categories, output_path="../data/processed/preprocessing_info.json"):
    category_to_label = {cat: idx for idx, cat in enumerate(categories)}
    label_to_category = {idx: cat for cat, idx in category_to_label.items()}
    
    info = {'categories': categories, 'category_to_label': category_to_label, 'label_to_category': label_to_category, 'num_classes': len(categories)}
    
    with open(output_path, 'w') as f:
        json.dump(info, f, indent=2)

def processed_data(images, labels, output_dir="../data/processed", filename_prefix="recyclevision"):
    images_path = os.path.join(output_dir, f"{filename_prefix}_images.npy")
    labels_path = os.path.join(output_dir, f"{filename_prefix}_labels.npy")
    
    np.save(images_path, images)
    np.save(labels_path, labels)

def checking_processed(data_dir="../data/processed", filename_prefix="recyclevision"):
    images_path = os.path.join(data_dir, f"{filename_prefix}_images.npy")
    labels_path = os.path.join(data_dir, f"{filename_prefix}_labels.npy")
    
    images = np.load(images_path)
    labels = np.load(labels_path)
    
    return images, labels

def save_generators(generators, categories, output_dir="../data/processed", filename_prefix="recyclevision"):
    splits = ['train', 'val', 'test']
    for split in splits:
        if f'{split}_data' in generators:
            X, y = generators[f'{split}_data']
            X_path = os.path.join(output_dir, f"{filename_prefix}_{split}_X.npy")
            y_path = os.path.join(output_dir, f"{filename_prefix}_{split}_y.npy")
            
            np.save(X_path, X)
            np.save(y_path, y)
    
    config = {'batch_size': generators.get('train').batch_size if 'train' in generators else 32, 'num_classes': len(categories), 'categories': categories}
    
    config_path = os.path.join(output_dir, f"{filename_prefix}_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

def load_generators(data_dir="../data/processed", filename_prefix="recyclevision"):
    splits = ['train', 'val', 'test']
    data = {}
    
    for split in splits:
        X_path = os.path.join(data_dir, f"{filename_prefix}_{split}_X.npy")
        y_path = os.path.join(data_dir, f"{filename_prefix}_{split}_y.npy")
        
        if os.path.exists(X_path) and os.path.exists(y_path):
            X = np.load(X_path)
            y = np.load(y_path)
            data[f'{split}_data'] = (X, y)
    
    config_path = os.path.join(data_dir, f"{filename_prefix}_config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        batch_size = config.get('batch_size', 32)
        
        train_datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
        
        val_datagen = ImageDataGenerator()
        test_datagen = ImageDataGenerator()
        
        if 'train_data' in data:
            X_train, y_train = data['train_data']
            data['train'] = train_datagen.flow(X_train, y_train, batch_size=batch_size)
        
        if 'val_data' in data:
            X_val, y_val = data['val_data']
            data['validation'] = val_datagen.flow(X_val, y_val, batch_size=batch_size)
        
        if 'test_data' in data:
            X_test, y_test = data['test_data']
            data['test'] = test_datagen.flow(X_test, y_test, batch_size=batch_size)
    
    return data

def main():
    processed_data_dir = "../data/processed"
    
    images_path = os.path.join(processed_data_dir, "recyclevision_images.npy")
    labels_path = os.path.join(processed_data_dir, "recyclevision_labels.npy")
    
    if os.path.exists(images_path) and os.path.exists(labels_path):
        images, labels = checking_processed(processed_data_dir)
        generators = load_generators(processed_data_dir)
    else:
        images, labels = original_data(target_size=(224, 224), max_samples_per_category=20)
        
        processed_data(images, labels, processed_data_dir)
        generators = create_data_generators(images, labels, batch_size=32)
        save_generators(generators, get_categories(), processed_data_dir)
    
    preprocessed_info(get_categories())

if __name__ == "__main__":
    main()
