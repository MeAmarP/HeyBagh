# HeyBagh
Find and Discover Visually Similar Content - Images

---
![HeyBagh](https://github.com/MeAmarP/HeyBagh/blob/ce1f3925be0a35c4a33a63d8d2c4faef217ec5a9/assets/gradio_interface.png)


## Technicals
- **Feature Extraction / Image embeddings:**: 
  - Pretrained ResNet-18, to extract features from images. 
  - CNN adept at capturing the semantic meaning and visual characteristics of an image.
  - Vector of size 1000
- **Vector Database:**
  - [Milvus Benchmark](https://zilliz.com/vector-database-benchmark-tool?database=Milvus%2CWeaviateCloud%2CQdrantCloud&dataset=large&filter=none%2Clow%2Chigh)
  - Milvus utilizes advanced indexing techniques and optimizations to perform blazing-fast searches for similar vectors.
  - Most of the vector index types supported by Milvus use approximate nearest neighbors search (ANNS) algorithms. 
  - Well-structured documentation with Helpful practical examples.
  - configuration used for HeyBagh
    - Similarity Metric Type: COSINE
    - ANN Algo: IVF_FLAT - Quantization-based index, High-speed query and Higher Recall
    - Top_k: 4
- **User Interface:**
  - Interface built with Gradio. 
  - This interface allows users to easily upload images or provide image URLs and visually explore the retrieved results.


## Objective:
Build a image search engine using deep learning techniques to find visually similar images.

## Plan:
- [x] Image Data - We can use Caltech-101, for starters.
- [X] DB - Store images and extracted features/embeddings.
- [X] UI - Design UI, we can start with Gradio.

TODO:
- Data Pre-processing
  - ~~resize images to a uniform size~~
  - augmentation??
- Model Selection and Feature Extraction
- Benchmark all options below for SPEED and ACCURACY
  - ~~CNN~~
  - Siamese Networks?
  - Triplet Networks
  - Autoencoders
  - CLIP
- Indexing
  - ~~Select Vector DB~~
  - ~~to efficiently store and retrieve feature vectors associated with each image~~
    - ~~Approximate Nearest Neighbors (ANN) algorithms~~
- Query Processing  
  - ~~Use same pre-processing technique as used in indexing~~

- Deployment
- Diagrams
  - Use `mermaid.js` for flow of the application
