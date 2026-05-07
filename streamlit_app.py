import os
from pathlib import Path

import streamlit as st
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import mlsound


def plot_pca(features, labels):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(features)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="viridis")
    ax.set_title("PCA of Audio Features")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    plt.colorbar(scatter, ax=ax)
    return fig


def main():
    st.title("Audio Clustering — mlsound")

    audio_dir = Path("audio")
    tmp_dir = None  # Track if we're using uploaded files
    
    if not audio_dir.exists():
        st.warning("No `audio` directory found in the repository root.")
        use_upload = st.checkbox("Upload audio files instead?")
        if not use_upload:
            return
        tmp_dir = Path("uploaded_audio")
        features, audio_files = [], []
    else:
        features, audio_files = mlsound.load_audio_data(str(audio_dir))

    if len(audio_files) == 0:
        st.info("No .wav files found. Please upload WAV files.")
        uploaded = st.file_uploader("Upload WAV files", type=["wav"], accept_multiple_files=True)
        if uploaded:
            tmp_dir = Path("uploaded_audio")
            tmp_dir.mkdir(exist_ok=True)
            for up in uploaded:
                dest = tmp_dir / up.name
                with open(dest, "wb") as f:
                    f.write(up.getbuffer())
            features, audio_files = mlsound.load_audio_data(str(tmp_dir))
        else:
            return

    n_clusters = st.sidebar.slider("Number of clusters", min_value=2, max_value=10, value=3)

    if st.button("Run clustering"):
        labels, _ = mlsound.perform_clustering(features, n_clusters)

        st.subheader("Cluster assignments")
        
        # Determine which directory to look for audio files
        source_dir = tmp_dir if tmp_dir and tmp_dir.exists() else audio_dir
        
        for cluster in range(n_clusters):
            st.markdown(f"**Cluster {cluster}**")
            cluster_files = []
            for i, label in enumerate(labels):
                if label == cluster:
                    cluster_files.append(audio_files[i])
                    st.write(f"- {audio_files[i]}")
                    
            # Play all audio files from this cluster
            for audio_file in cluster_files:
                file_path = source_dir / audio_file
                if file_path.exists():
                    with open(file_path, "rb") as f:
                        st.audio(f.read(), format="audio/wav")

        st.subheader("PCA visualization")
        fig = plot_pca(features, labels)
        st.pyplot(fig)


if __name__ == "__main__":
    main()