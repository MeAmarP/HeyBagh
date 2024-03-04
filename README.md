# HeyBagh
Find and Discover Visually Similar Content - Images

## Objective:
Build a image search engine using deep learning techniques to find visually similar images

## Plan:
- [x] Image Data - We can use Caltech-101, for starters.
- [X] DB - Store images and extracted features/embeddings.
- [ ] UI - Design UI, we can start with Gradio.
- [ ] Core - Define deep learning technique to be used for extracting user uploaded images.
- [ ] Metric - Define Metric for similarity.

TODO:
- Data Pre-processing
  - resize images to a uniform size
  - augmentation??
- Model Selection and Feature Extraction
- Benchmark all options below for SPEED and ACCURACY
  - CNN
  - Siamese Networks
  - Triplet Networks
  - Autoencoders
  - CLIP
- Indexing
  - Select Vector DB
  - to efficiently store and retrieve feature vectors associated with each image
    - Approximate Nearest Neighbors (ANN) algorithms
    - Locality-Sensitive Hashing (LSH)
    - tree-based methods like KD-trees or Ball-trees.
- Query Processing  
  - Use same pre-processing technique as used in indexing

- Deployment
- Diagrams
  - Use `mermaid.js` for flow of the application
