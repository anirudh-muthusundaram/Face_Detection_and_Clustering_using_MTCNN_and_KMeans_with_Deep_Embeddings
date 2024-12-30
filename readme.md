# Face Detection and Clustering using MTCNN and K-Means with Deep Embeddings

This model implements an automated pipeline for face detection, embedding generation, and clustering using deep learning and unsupervised machine learning techniques. The model utilizes MTCNN (Multi-task Cascaded Convolutional Neural Networks) for face detection, and InceptionResNetV1 to generate 512-dimensional facial embeddings based on the VGGFace2 dataset. The embeddings are subsequently clustered using K-Means Clustering to group similar faces. The pipeline also includes detailed evaluations using clustering metrics like Silhouette Score, Davies-Bouldin Score, and Calinski-Harabasz Score, along with visualizations such as t-SNE and PCA. The results demonstrate the effectiveness of the model in organizing large datasets of face images into meaningful clusters, with potential applications in facial recognition, surveillance, and dataset preprocessing.

## Getting Started

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/anirudh-muthusundaram/your-repo-name.git
   cd your-repo-name

2. **Create and Activate a Virtual Environment (Windows)**:
    - # For venv
    ```bash
   python -m venv face_env

3. **Create and Activate a Virtual Environment (MacOS)**:
    - # For venv
    ```bash
   source face_env/bin/activate

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt

3. **Download and and Add the Dataset**:
- Download and add the dataset from https://www.kaggle.com/datasets/fareselmenshawii/face-detection-dataset?resource=download-directory.
- Create a folder named images and all the data downloaded.

4. **Train, Test and Validate the Model:**:
- All of the above functions can be done on the train.ipynb file.
- Please use Jupyter Notebook to run the program. 

## **Dependencies**

The project depends on the following libraries:
Install these dependencies using the `requirements.txt` file

- **Python 3.8+**
- `numpy`
- `opencv-python`
- `facenet-pytorch`
- `torch`
- `scikit-learn`
- `tqdm`
- `seaborn`
- `matplotlib`

## Overview

1. **Step 1: Train the Clustering Model**:  
   To train the clustering model, use the train.ipynb notebook. This notebook detects faces in the training set, generates embeddings using InceptionResNetV1, and performs K-Means clustering to group similar faces. Upon completion, the following outputs will be generated:
   - cluster_centers.npy: Stores the cluster centers for K-Means.
   - train_assignments.json: Contains cluster assignments for the training images.
   Simply open and run the train.ipynb notebook in your Jupyter environment to complete this step.

2. **Step 2: Test the Clustering Model**:  
   After training, use the train code cell within the train.py script to apply the trained model on the validation dataset. This script will assign cluster labels to each validation image based on the cluster centers computed during training. The output of this step is:
   - test_assignments.json: Contains cluster assignments for the validation images.

3. **Step 3: Evaluate and Visualize Clustering**:  
   Once the training and testing steps are complete, the t-SNE and PCA visualizations included in the train.ipynb notebook can be used to analyze and evaluate the clustering results. These visualizations help to verify cluster separations and provide insights into the quality of the embeddings. Run the respective cells in the train.ipynb notebook to generate the plots and evaluate the clustering performance.

## Note

- Make sure you use and experiment with the latest version (e.g., _vX (future purposes)).
- Any and all contributions to this this model are welcome.
- This repository will be updated iteratively to improve the model.
 
## Features

- **Face Detection**: Uses MTCNN to detect faces in input images. Handles multiple faces per image (if applicable).
- **Embedding Generation**: Uses InceptionResNetV1 (pre-trained on VGGFace2) to generate 512-dimensional embeddings for each detected face.
- **Clustering**: Uses K-Means clustering to group faces into a specified number of clusters (K). Automatically saves cluster centers and assignments.
- **Visualizations**: Bar Plot: Displays the distribution of images across clusters. t-SNE: Reduces the embedding dimensions to 2D for visualization. PCA: Provides an alternative dimensionality reduction and clustering visualization.

## Performance Metrics

- **Silhouette Score**: Measures how similar an object is to its own cluster compared to other clusters. Higher scores (>0.5) indicate better-defined clusters.
- **Davies-Bouldin Score**: Evaluates the separation between clusters and their internal consistency. Lower scores indicate better clustering.
- **Calinski-Harabasz Score**: Measures the ratio of between-cluster variance to within-cluster variance. Higher scores indicate better-defined clusters.

## Known Issues and Improvements

- **Low Silhouette Score**: If the score is low, consider experimenting with different values of K using the Elbow Method or Silhouette Analysis.
- **Overlapping Clusters**: The t-SNE and PCA visualizations might show overlap between clusters. This could be addressed by: Improving embeddings with a more advanced model (e.g., ArcFace). Using alternative clustering methods like DBSCAN or Hierarchical Clustering.

## Future Improvements

- Integrate DBSCAN or Hierarchical Clustering for better handling of non-spherical clusters.
- Replace MTCNN with a faster and more accurate detector like RetinaFace.
- Use a more advanced embedding model like ArcFace or DINO.
- Implement additional preprocessing steps to improve face detection and alignment.