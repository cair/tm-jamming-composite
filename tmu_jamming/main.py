#!/usr/bin/env python
# coding: utf-8

import os
from typing import Dict
import cv2
import numpy as np
from tqdm import tqdm
import pickle
from tmu.models.classification.vanilla_classifier import TMClassifier
from tmu.preprocessing.standard_binarizer.binarizer import StandardBinarizer
from tmu.data import TMUDataset
from tmu.composite.components.base import TMComponent
from tmu.composite.composite import TMComposite
from tmu.composite.config import TMClassifierConfig
from tmu.composite.callbacks.base import TMCompositeCallback
import logging

_LOGGER = logging.getLogger(__name__)


class SpectrogramProcessor(TMUDataset):
    def __init__(self, dataset_name, use_cache=False):
        super().__init__()
        self.dataset_name = dataset_name
        self.use_cache = use_cache
        self.classes = {"11": 0, "12": 1, "31": 2, "32": 3, "34": 4, "40": 5, "chirp_uneven": 6, "not_sweep": 7, "pulse": 8, "sine": 9}
        self.cache_dir = './cache'
        os.makedirs(self.cache_dir, exist_ok=True)

    def _cache_path(self, dataset_type):
        return os.path.join(self.cache_dir, f"{self.dataset_name}_{dataset_type}.pkl")

    def _load_from_cache(self, dataset_type):
        path = self._cache_path(dataset_type)
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
        return None

    def _save_to_cache(self, dataset_type, data):
        path = self._cache_path(dataset_type)
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def _transform(self, name, dataset):
        if 'x' in name:  # Apply only to image data
            dataset = self.convert_to_gray(dataset)
            dataset = self.convert_to_binary(dataset)
        return dataset

    def _retrieve_dataset(self) -> Dict[str, np.ndarray]:
        dataset = {}
        for dtype in ['train', 'test']:
            if self.use_cache:
                cached_data = self._load_from_cache(dtype)
                if cached_data:
                    dataset.update(cached_data)
                    continue
            data = self.create_train_data(dtype)
            if self.use_cache:
                self._save_to_cache(dtype, {f'x_{dtype}': data[0], f'y_{dtype}': data[1]})
            dataset[f'x_{dtype}'] = data[0]
            dataset[f'y_{dtype}'] = data[1]
        return dataset

    def assign_label(self, img_type):
        return self.classes.get(img_type, -1)  # Return -1 if class is not found

    def make_data(self, img_type, DIR, X_data, y_data):
        for img in tqdm(os.listdir(DIR)):
            label = self.assign_label(img_type)
            path = os.path.join(DIR, img)
            
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (50, 50))
            
            X_data.append(np.array(img))
            y_data.append(label)

    def create_train_data(self, dataset_type): 
        X_data = []
        y_data = []
        main_path = os.path.join('/data', self.dataset_name) 

        data_dir = os.path.join(main_path, dataset_type)

        for class_folder in os.listdir(data_dir):
            class_dir = os.path.join(data_dir, class_folder)
            self.make_data(class_folder, class_dir, X_data, y_data)

        return np.array(X_data), np.array(y_data)

    def convert_to_gray(self, images):
        return np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images])

    def convert_to_binary(self, images):
        binary_images = images.copy()
        for i in range(images.shape[0]):
            _, binary_images[i,:] = cv2.threshold(images[i], 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary_images


class DummyComponent(TMComponent):
    
    def __init__(self, model_cls, model_config, data, **kwargs) -> None:
        super().__init__(model_cls=model_cls, model_config=model_config, **kwargs)
        self.data = data

    def preprocess(self, data: dict):
        # In this setup, we expect data to contain two keys, 'train' and 'test' with bools.
        # We will return the correct dataset based on the key
        
        assert "train" in data or "test" in data, "Data must contain 'train' or 'test' key"
        assert not (data["train"] and data["test"]), "Only one of 'train' or 'test' can be true"
        assert data["train"] or data["test"], "At least one of 'train' or 'test' must be true"
        
        if data["train"]:
            return dict(
                X=self.data['x_train'],
                Y=self.data['y_train'],
            )
        elif data["test"]:
            return dict(
                X=self.data['x_test'],
                Y=self.data['y_test'],
            )
        
# Create a component for each dataset, these are just empty classes to rename from DummyComponent to the actual dataset name
class SpectogramComponent(DummyComponent):
    pass
class DominantFreqComponent(DummyComponent):
    pass
class PSDComponent(DummyComponent):
    pass
class FFTComponent(DummyComponent):
    pass
class STDComponent(DummyComponent):
    pass


class TMCompositeEvaluationCallback(TMCompositeCallback):

    def __init__(self, data):
        super().__init__()
        self.best_acc = 0.0
        self.data = data

    def on_epoch_end(self, composite, epoch, logs=None):
        preds = composite.predict(data=self.data)
        acc = (preds == self.data["Y"]).mean()
        _LOGGER.info(f"Epoch {epoch} - Accuracy: {acc:.2f}")

if __name__ == "__main__":
    
    spectogram_data = SpectrogramProcessor("spectrogram_v2", use_cache=True).get()
    dominant_freq_data = SpectrogramProcessor("dominant_freq", use_cache=True).get()
    psd_data = SpectrogramProcessor("psd", use_cache=True).get()
    fft_data = SpectrogramProcessor("fft", use_cache=True).get()
    std_data = SpectrogramProcessor("std", use_cache=True).get()
    
    # Print the dataset shapes
    _LOGGER.info(f"Spectogram data shapes: {spectogram_data['x_train'].shape}, {spectogram_data['y_train'].shape}")
    _LOGGER.info(f"Dominant freq data shapes: {dominant_freq_data['x_train'].shape}, {dominant_freq_data['y_train'].shape}")
    _LOGGER.info(f"PSD data shapes: {psd_data['x_train'].shape}, {psd_data['y_train'].shape}")
    _LOGGER.info(f"FFT data shapes: {fft_data['x_train'].shape}, {fft_data['y_train'].shape}")
    _LOGGER.info(f"STD data shapes: {std_data['x_train'].shape}, {std_data['y_train'].shape}")    

    # General hyperparameters
    epochs = 100

    composite_model = TMComposite(
        components=[
            SpectogramComponent(TMClassifier, TMClassifierConfig(
                number_of_clauses=500,
                T=250,
                s=10,
                max_included_literals=32,
                weighted_clauses=True,
                patch_dim=(10, 10)
            ), spectogram_data, epochs=epochs),
            
            DominantFreqComponent(TMClassifier, TMClassifierConfig(
                number_of_clauses=500,
                T=250,
                s=10,
                max_included_literals=32,
                weighted_clauses=True,
                patch_dim=(10, 10)
            ), dominant_freq_data, epochs=epochs),
            
            PSDComponent(TMClassifier, TMClassifierConfig(
                number_of_clauses=500,
                T=250,
                s=10,
                max_included_literals=32,
                weighted_clauses=True,
                patch_dim=(10, 10)
            ), psd_data, epochs=epochs),
            
            FFTComponent(TMClassifier, TMClassifierConfig(
                number_of_clauses=500,
                T=250,
                s=10,
                max_included_literals=32,
                weighted_clauses=True,
                patch_dim=(10, 10)
            ), fft_data, epochs=epochs),
            
            STDComponent(TMClassifier, TMClassifierConfig(
                number_of_clauses=500,
                T=250,
                s=10,
                max_included_literals=32,
                weighted_clauses=True,
                patch_dim=(10, 10)
            ), std_data, epochs=epochs)
        ]
    )

    # Train the composite model
    composite_model.fit(
        data=dict(train=True, test=False),
        callbacks=[
            TMCompositeEvaluationCallback(data=dict(train=False, test=True))
        ]
    )
    
    preds = composite_model.predict(data=dict(train=False, test=True))

    y_true = spectogram_data["y_test"].flatten()
    for k, v in preds.items():
        acc = (v == y_true).mean()
        _LOGGER.info(f"{k} Accuracy: {acc:.2f}")