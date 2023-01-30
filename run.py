import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.simplefilter("error", np.VisibleDeprecationWarning)
from facer import facer

# Load face images
path_to_images = "./cleaner"
images = facer.load_images(path_to_images)
#
# # Detect landmarks for each face
landmarks, faces = facer.detect_face_landmarks(images)
#
# # Use  the detected landmarks to create an average face
average_face_results = facer.create_average_face(faces, landmarks,
                                                 save_image=True,
                                                 return_intermediates=True)
#
average_face = average_face_results[0]
# These are sets of images
warped = average_face_results[1]
incremental = average_face_results[2]
imagesNorm = average_face_results[3]
# View the composite image
plt.imshow(average_face)
plt.show()
