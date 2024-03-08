import os
from PIL import Image
from dotenv import load_dotenv, find_dotenv
import gradio as gr

from milvus_client import HeyBaghMilvusClient
from feature_extractor import FeatureExtractor_CNN
from utils import display_images

load_dotenv(find_dotenv())




def main():
    # connect to milvlus vector db
    hey_bagh_db_client = HeyBaghMilvusClient()

    # load collection
    hey_bagh_db_client.load_collection("heybagh_caltech101_imgs")

    # extract a feature of input image
    image_feat_extractor = FeatureExtractor_CNN()
    root_path_img_dataset = "/home/c3po/Documents/project/learning/amar-works/datasets/caltech-101/101_ObjectCategories"
    # image_path = "/home/c3po/Documents/project/learning/amar-works/datasets/caltech-101/101_ObjectCategories/brontosaurus/image_0006.jpg"

    # # DO image search
    # search_hits = hey_bagh_db_client.img_search(
    #     img_path=image_path, feature_extractor=image_feat_extractor
    # )

    # process output
    # output_img_paths = [(root_path_img_dataset + "/" + path) for path in search_hits[0]]

    # Show results in matplotlib
    # display_images(image_path, output_image_paths=output_img_paths)

    # User interface using Gradio to show results.
    def show_results(uploaded_image_file):
        input_image_path = uploaded_image_file.name
        input_image = Image.open(uploaded_image_file.name).resize((180,180))
            # DO image search
        search_hits = hey_bagh_db_client.img_search(
            img_path=input_image_path, feature_extractor=image_feat_extractor
        )
        result_image_paths = [(root_path_img_dataset + "/" + path) for path in search_hits[0]]
        result_images = [Image.open(path).resize((180,180)) for path in result_image_paths]

        return input_image, result_images
        # Define Gradio interface
    iface = gr.Interface(
        fn=show_results,
        inputs=["file"],
        outputs=[
            gr.Image(type="pil", label="Your Input Image", width=200),
            gr.Gallery(label="Result Images", min_width=200)
        ],
        title="Project HeyBagh",
        description="Upload a photo of object that you wish to search",
        theme=gr.themes.Soft())
    
    iface.launch()

        



if __name__ == "__main__":
    main()
