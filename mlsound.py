import os
import librosa
import numpy as np
from sklearn.cluster import KMeans

def extract_features(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    return mfcc_mean

def load_audio_data(directory):
    features = []
    audio_files = []    
    for file in os.listdir(directory):
        if file.endswith(".wav"):
            audio_file_path = os.path.join(directory, file)
            audio_files.append(file)
            mfcc_features = extract_features(audio_file_path)
            features.append(mfcc_features)
    return np.array(features), audio_files

def perform_clustering(features, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(features)
    return labels, kmeans

def main():
    # Optional: keep for command-line usage
    audio_directory = r"D:\ForInternshipProjects\mlsoundclassification\audio"
    features, audio_files = load_audio_data(audio_directory)
    n_clusters = 3
    labels, kmeans = perform_clustering(features, n_clusters)
    print(f"Audio files grouped into {n_clusters} clusters:\n")
    for cluster in range(n_clusters):
        print(f"Cluster {cluster}:")
        for i, label in enumerate(labels):
            if label == cluster:
                print(f"  - {audio_files[i]}")

if __name__ == "__main__":
    main()