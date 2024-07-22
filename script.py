import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def load_image(image_path):
    """
    Charge une image et la convertit en espace de couleur RGB.
    
    Args:
        image_path (str): Chemin de l'image à charger.
    
    Returns:
        np.ndarray: Image en espace de couleur RGB.
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def restructure_image(image):
    """
    Restructure l'image en un tableau de pixels.
    
    Args:
        image (np.ndarray): Image à restructurer.
    
    Returns:
        np.ndarray: Tableau de pixels de l'image.
    """
    pixels = image.reshape((-1, 3))
    return pixels

def detect_main_colors(pixels, n_colors=3):
    """
    Utilise K-means pour détecter les couleurs principales d'une image.
    
    Args:
        pixels (np.ndarray): Tableau de pixels de l'image.
        n_colors (int): Nombre de couleurs principales à détecter.
    
    Returns:
        np.ndarray: Centres des clusters représentant les couleurs principales.
        np.ndarray: Labels des pixels correspondant aux clusters.
    """
    kmeans = KMeans(n_clusters=n_colors)
    kmeans.fit(pixels)
    return kmeans.cluster_centers_, kmeans.labels_

def plot_colors(centers):
    """
    Affiche les couleurs principales détectées.
    
    Args:
        centers (np.ndarray): Centres des clusters représentant les couleurs principales.
    """
    plt.figure(figsize=(8, 6))
    for i, color in enumerate(centers):
        plt.subplot(1, len(centers), i + 1)
        plt.imshow([[color / 255.0]])
        plt.axis('off')
    plt.show()

def main(image_path, n_colors=3):
    """
    Charge une image, détecte les couleurs principales, et les affiche.
    
    Args:
        image_path (str): Chemin de l'image à analyser.
        n_colors (int): Nombre de couleurs principales à détecter.
    """
    image = load_image(image_path)
    pixels = restructure_image(image)
    centers, labels = detect_main_colors(pixels, n_colors)
    plot_colors(centers)

# Utilisation
main('testimg.png', n_colors=2)
