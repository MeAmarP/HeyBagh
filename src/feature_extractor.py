import torch
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image
from scipy.spatial import distance


class FeatureExtractor_CNN:
    def __init__(self):
        self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model.eval()  # Set model to evaluation mode

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        image_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
        return image_tensor

    def extract_features(self, image_tensor):
        with torch.no_grad():
            features = self.model(image_tensor).squeeze(0).cpu().numpy()
        return features

    def cosine_similarity(self, features1, features2):
        similarity = 1 - distance.cosine(features1, features2)
        return similarity

    def cosine_similarity_images(self, image1_path, image2_path):
        img1_tensor = self.preprocess_image(image1_path)
        img2_tensor = self.preprocess_image(image2_path)
        
        features1 = self.extract_features(img1_tensor)
        features2 = self.extract_features(img2_tensor)

        return self.cosine_similarity(features1, features2)


# Example usage
if __name__ == "__main__":
    extractor = FeatureExtractor_CNN()
    image1_path = "/home/c3po/Documents/project/learning/amar-works/HeyBagh/data/caltech-101/101_ObjectCategories/ant/image_0001.jpg"
    image2_path = "/home/c3po/Documents/project/learning/amar-works/HeyBagh/data/caltech-101/101_ObjectCategories/ant/image_0020.jpg"

    try:
        similarity_score = extractor.cosine_similarity_images(image1_path, image2_path)
        print(f"Cosine similarity between images: {similarity_score}")
    except Exception as e:
        print(f"Error: {e}")

