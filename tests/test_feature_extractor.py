# test_feature_extractor.py
import pytest
from feature_extractor import FeatureExtractor_CNN
import torch
import numpy as np

@pytest.fixture
def feature_extractor():
    return FeatureExtractor_CNN()

def test_init(feature_extractor):
    assert feature_extractor.model is not None
    assert feature_extractor.transform is not None

def test_preprocess_image(feature_extractor):
    # Assuming you have a sample image named "sample.jpg" in the current directory
    image_path = "/home/c3po/Documents/project/learning/amar-works/HeyBagh/data/image_0006.jpg"
    processed_image = feature_extractor.preprocess_image(image_path)
    assert isinstance(processed_image, torch.Tensor)
    assert processed_image.shape == torch.Size([1, 3, 224, 224])

def test_extract_features(feature_extractor):
    # Create a dummy tensor to simulate a preprocessed image
    dummy_tensor = torch.randn(1, 3, 224, 224).to(feature_extractor.device)
    features = feature_extractor.extract_features(dummy_tensor)
    assert isinstance(features, np.ndarray)
    # The exact shape of features depends on the model architecture
    assert len(features.shape) == 1

def test_cosine_similarity(feature_extractor):
    # Create dummy feature vectors
    features1 = np.random.rand(1000)
    features2 = np.random.rand(1000)
    similarity = feature_extractor.cosine_similarity(features1, features2)
    assert isinstance(similarity, float)
    assert 0 <= similarity <= 1

def test_cosine_similarity_images(feature_extractor):
    # Assuming you have two sample images named "sample1.jpg" and "sample2.jpg" in the current directory
    image1_path = "/home/c3po/Documents/project/learning/amar-works/HeyBagh/data/image_0006.jpg"
    image2_path = "/home/c3po/Documents/project/learning/amar-works/HeyBagh/data/image_0006.jpg"
    similarity = feature_extractor.cosine_similarity_images(image1_path, image2_path)
    assert 0 <= similarity <= 1

def test_preprocess_extract_feature(feature_extractor):
    # Assuming you have a sample image named "sample.jpg" in the current directory
    image_path = "/home/c3po/Documents/project/learning/amar-works/HeyBagh/data/image_0006.jpg"
    features = feature_extractor.preprocess_extract_feature(image_path)
    assert isinstance(features, np.ndarray)
    # The exact shape of features depends on the model architecture
    assert len(features.shape) == 1
