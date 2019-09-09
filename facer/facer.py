import cv2
import dlib
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import math
import os
import glob

from facer.utils import similarityTransform, constrainPoint, calculateDelaunayTriangles, warpTriangle

# https://www.learnopencv.com/facial-landmark-detection/
# https://www.learnopencv.com/average-face-opencv-c-python-tutorial/
def load_face_detector():
    """Loads the dlib face detector"""
    return dlib.get_frontal_face_detector()

def load_landmark_predictor(predictor_path):
    return dlib.shape_predictor(predictor_path)

# Load the face detector and landmark predictor
PREDICTOR_PATH = "./dlib/shape_predictor_68_face_landmarks.dat"
detector = load_face_detector()
predictor = load_landmark_predictor(PREDICTOR_PATH)
print("Done, models loaded.")

def plot_face_rectangle(points, color="cyan", style="-", alpha=0.8):
    plt.plot([points.left(), points.left()], [points.bottom(), points.top()], style, color=color, alpha=alpha)
    plt.plot([points.left(), points.right()], [points.bottom(), points.bottom()], style, color=color, alpha=alpha)
    plt.plot([points.left(), points.right()], [points.top(), points.top()], style, color=color, alpha=alpha)
    plt.plot([points.right(), points.right()], [points.top(), points.bottom()], style, color=color, alpha=alpha)

def plot_face_landmarks(points, color="red", style=".", **kwargs):
    for point in points:
        try:
            x, y = point.x, point.y
        except:
            x, y = point[0], point[1]
        plt.plot(x, y, style, color=color, **kwargs)

def save_landmarks_to_disk(points, fp):
    txt = "\n".join(list(map(lambda p: f"{p.x}, {p.y}", (points))))
    with open(fp, "w") as outfile:
        outfile.write(txt)

def glob_image_files(root, extensions=["jpg", "jpeg", "png"]):
    """Returns a list of image files in `root`"""
    files = glob.glob(os.path.join(root, "*"))
    return [f for f in files if f.rsplit(".", 1)[-1] in extensions]

def load_images(root, verbose=True):
    """Returns list of image arrays
    :param root: (str) Directory containing face images
    :param verbose: (bool) Toggle verbosity
    :output images: (dict) Dict of OpenCV image arrays, key is filename
    """
    files = sorted(glob_image_files(root))
    num_files = len(files)
    if verbose:
        print(f"\nFound {num_files} in '{root}'.")
        N = max(round(0.10 * num_files), 1)

    # Load the images
    images = {}
    for n, file in enumerate(files):
        if verbose and n % N == 0:
            print(f"({n + 1} / {num_files}): {file}")

        image = cv2.imread(file)[..., ::-1]
        image = np.float32(image) / 255.0
        images[file] = image
    return images

def load_face_landmarks(root, verbose=True):
    """Load face landmarks created by `detect_face_landmarks()`
    :param root: (str) Path to folder containing CSV landmark files
    :param verbose: (bool) Toggle verbosity
    :output landmarks: (list)
    """
    #List all files in the directory and read points from text files one by one
    all_paths = glob.glob(root.strip("/") + "/*_landmarks*")
    print(all_paths)
    landmarks = []
    for fn in all_paths:
        points = []
        with open(fn) as file:
            for line in file:
                x, y = line.split(", ")
                points.append((int(x), int(y)))

        # Store array of points
        landmarks.append(points)
    return landmarks

def detect_face_landmarks(images,
                          save_landmarks=True,
                          max_faces=1,
                          verbose=True,
                          print_freq=0.10):
    """Detect and save the face landmarks for each image
    :param images: (dict) Dict of image files and arrays from `load_images()`.
    :param save_landmarks: (bool) Save landmarks to .CSV
    :param max_faces: (int) Skip images with too many faces found.
    :param verbose: (bool) Toggle verbosity
    :param print_freq: (float) How often do you want print statements?
    :output landmarks: (list) 68 landmarks for each found face
    :output faces: (list) List of the detected face images
    """
    num_images = len(images.keys())
    if verbose:
        print(f"\nStarting face landmark detection...")
        print(f"Processing {num_images} images.")
        N = max(round(print_freq * num_images), 1)

    # Look for face landmarks in each image
    num_skips = 0
    all_landmarks, all_faces = [], []
    for n, (file, image) in enumerate(images.items()):
        if verbose and n % N == 0:
            print(f"({n + 1} / {num_images}): {file}")

        # Try to detect a face in the image
        imageForDlib = dlib.load_rgb_image(file) # Kludge for now
        found_faces = detector(imageForDlib, 1)

        # Only save landmarks when num_faces = 1
        if len(found_faces) == 0 or len(found_faces) > max_faces:
            num_skips += 1
            continue

        # Find landmarks, save to CSV
        for num, face in enumerate(found_faces):
            landmarks = predictor(imageForDlib, face)
            if not landmarks:
                continue

            # Add this image to be averaged later
            all_faces.append(image)

            # Convert landmarks to list of (x, y) tuples
            lm = [(point.x, point.y) for point in landmarks.parts()]
            all_landmarks.append(lm)

            # Save landmarks as a CSV file (optional)
            if save_landmarks:
                fp = file.rsplit(".", 1)[0] + f"_landmarks_{num}.csv"
                save_landmarks_to_disk(landmarks.parts(), fp=fp)

    if verbose:
        print(f"Skipped {100 * (num_skips / num_images):.1f}% of images.")
    return all_landmarks, all_faces

def create_average_face(faces,
                        landmarks,
                        output_dims=(600, 600),
                        save_image=True,
                        output_file="average_face.jpg",
                        return_intermediates=False,
                        verbose=True,
                        print_freq=0.05):
    """Combine the faces into an average face"""
    if verbose:
        print(f"\nStarting face averaging for {len(faces)} faces.")
    msg = "Number of landmark sets != number of images."
    assert len(faces) == len(landmarks), msg

    # Eye corners
    num_images = len(faces)
    n = len(landmarks[0])
    w, h = output_dims
    eyecornerDst = [(np.int(0.3 * w), np.int(h / 3)),
                    (np.int(0.7 * w), np.int(h / 3))]
    imagesNorm, pointsNorm = [], []

    # Add boundary points for delaunay triangulation
    boundaryPts = np.array([(0,0),
                            (w / 2, 0),
                            (w - 1, 0),
                            (w - 1, h / 2),
                            (w - 1, h - 1),
                            (w / 2, h - 1),
                            (0, h - 1),
                            (0, h / 2)])

    # Initialize location of average points to 0s
    pointsAvg = np.array([(0, 0)] * (len(landmarks[0]) + len(boundaryPts)), np.float32())

    # Warp images and trasnform landmarks to output coordinate system,
    # and find average of transformed landmarks.
    warped, incremental = [], []
    N = max(round(print_freq * num_images), 1)
    for i in range(0, num_images):
        if verbose and i % N == 0:
            print(f"Image {i + 1} / {num_images}")

        # Corners of the eye in input image
        points1 = landmarks[i]
        eyecornerSrc  = [landmarks[i][36], landmarks[i][45]]

        # Compute similarity transform
        tform = similarityTransform(eyecornerSrc, eyecornerDst)

        # Apply similarity transformation
        img_affine = cv2.warpAffine(faces[i], tform, (w, h)).get()

        # Apply similarity transform on points
        points2 = np.reshape(np.array(points1), (68, 1, 2))
        points = cv2.transform(points2, tform).get()
        points = np.float32(np.reshape(points, (68, 2)))

        # Append boundary points. Will be used in Delaunay Triangulation
        points = np.append(points, boundaryPts, axis=0)

        # Calculate location of average landmark points.
        pointsAvg = pointsAvg + points / num_images
        pointsNorm.append(points)
        imagesNorm.append(img_affine)

        # Delaunay triangulation
        rect = (0, 0, w, h);
        dt = calculateDelaunayTriangles(rect, np.array(pointsAvg))

        # Warp input images to average image landmarks
        output = np.zeros((h, w, 3), np.float32())
        for i in range(0, len(imagesNorm)):
            img = np.zeros((h, w, 3), np.float32())
            # Transform triangles one by one
            for j in range(0, len(dt)):
                tin, tout = [], []

                for k in range(0, 3):
                    pIn = pointsNorm[i][dt[j][k]]
                    pIn = constrainPoint(pIn, w, h)

                    pOut = pointsAvg[dt[j][k]]
                    pOut = constrainPoint(pOut, w, h)

                    tin.append(pIn)
                    tout.append(pOut)
                img = warpTriangle(imagesNorm[i], img, tin, tout)
            incremental.append((output + img) / (i + 1))


            # Add image intensities for averaging
            output = output + img

        # Divide by num_images to get average
        output = output / num_images

        warped.append(img_affine)
    incremental = incremental[-num_images:]
    print('Done.')

    # Save the output image to disk
    if save_image:
        cv2.imwrite(output_file, 255 * output[..., ::-1])
    if return_intermediates: # For animated GIFs
        return output, warped, incremental, imagesNorm
    return output

def create_average_face_from_directory(dir_in,
                                       dir_out,
                                       filename,
                                       save_image=True,
                                       **kwargs):
    verbose = kwargs.get('verbose', True)
    if verbose:
        print(f"Directory: {dir_in}")
    images = load_images(dir_in, verbose=verbose)
    if len(images) == 0:
        if verbose:
            print(f"Couldn't find any images in: '{dir_in}'.")
        return

    # Detect landmarks for each face
    landmarks, faces = detect_face_landmarks(images, verbose=verbose)

    # Use  the detected landmarks to create an average face
    fn = f"average_face_{filename}.jpg"
    fp = os.path.join(dir_out, fn).replace(" ", "_")
    average_face = create_average_face(faces,
                                       landmarks,
                                       output_file=fp,
                                       save_image=True)

    # Save a labeled version of the average face
    if save_image:
        save_labeled_face_image(average_face, filename, dir_out)
    return average_face

def save_labeled_face_image(image, name, dir_out="./", label=""):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)

    # Add a title
    kwargs = {"fontsize": 20, "fontweight": "heavy",
              "color": "gray", "alpha":0.9}
    title = f"{name} average face"
    ax.set_title(title, **kwargs)

    # Touch up the image
    ax.set(**{"xlabel": '', "ylabel": '', "xticks": [], "yticks": []})
    plt.tight_layout()
    kwargs = {'fontsize': 17, 'color': 'black', 'weight': 'heavy',
              'alpha': 0.6, 'ha': 'right'}
    x, y = image.shape[:2]
    x, y = 0.98 * x, 0.97 * y
    ax.text(x, y, label, **kwargs)

    # Save the image
    fp = os.path.join(dir_out, f"average_face_{name}_labeled.png")
    fp = fp.replace(" ", "_")
    fig.savefig(fp, dpi=300)
    return

def create_animated_gif(path_to_images, save_gif=True, verbose=True):
    """Create an animated face average GIF from a directory of images"""
    def save_to_file(gif, fn=None, fps=None, verbose=True):
        fn = "animation" if not fn else fn
        if not fps:
            fps = (1e3 / gif._interval)
        if verbose:
            print(fn)
        gif.save(f'./{fn}.gif', writer='imagemagick', fps=fps)
        if verbose:
            print("Done saving the GIF.")

    # Load the images
    images = load_images(path_to_images, verbose=verbose)

    # Detect face landmarks
    landmarks, faces = detect_face_landmarks(images, verbose=verbose)

    # Create the average face and interim images
    averaged, warped, incremental, raw = create_average_face(faces,
                                                             landmarks,
                                                             return_intermediates=True,
                                                             save_image=False,
                                                             verbose=verbose)

    # Create the animation
    def tight(**kwargs):
        plt.tight_layout(**kwargs)

    # Make the plot
    fig, axs = plt.subplots(1, 2, figsize=(6, 3.7))
    lines = []
    titles = ["Individual", "Averaged"]
    kwargs = {"fontsize": 18, "color": "white", "alpha": 1.0, "weight": "heavy"}
    for ax, title in zip(axs, titles):
        lines.append(ax.imshow(np.zeros_like(warped[0])))
        ax.axis('off')
        ax.set_title(title, fontsize=18)
    k = dict(pad=0, w_pad=0, h_pad=0) # Layout values
    tight(**k)

    # initialization function: plot the background of each frame
    def init():
        for line in lines:
            line.set_data(np.zeros_like(raw[0]))
        tight(**k)
        return lines

    # animation function. This is called sequentially
    def animate(i):
        num_raw = len(raw)
        i = min(num_raw - 1, i)
        orig, average = lines
        orig.set_data(raw[i])
        average.set_data(incremental[-num_raw:][i])
        tight(**k)
        return lines

    # Call the animator
    anim = animation.FuncAnimation(fig,
                                   animate,
                                   init_func=init,
                                   frames=round(1.5 * len(raw)),
                                   interval=500,
                                   blit=True)

    # Save the animation
    if save_gif:
        save_to_file(anim)
    return anim, averaged
