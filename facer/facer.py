import glob
import os
from typing import cast

import cv2
import dlib
import matplotlib.pyplot as plt
import numpy as np
from cv2.typing import MatLike
from matplotlib.animation import FuncAnimation
from numpy.typing import NDArray

from facer.typing import Detector, Points, Predictor, Rectangle
from facer.utils import (
    calculateDelaunayTriangles,
    constrainPoint,
    similarityTransform,
    warpTriangle,
)


# https://www.learnopencv.com/facial-landmark-detection/
# https://www.learnopencv.com/average-face-opencv-c-python-tutorial/
def load_face_detector() -> Detector:
    """Loads the dlib face detector"""
    return dlib.get_frontal_face_detector()  # type: ignore[attr-defined]


def load_landmark_predictor(predictor_path: str) -> Predictor:
    return dlib.shape_predictor(predictor_path)  # type: ignore[attr-defined]


def load_rgb_image(filename: str) -> NDArray[np.uint8]:
    """Takes a path and returns a numpy array (RGB) containing the image"""
    return dlib.load_rgb_image(filename)  # type: ignore[attr-defined]


# Load the face detector and landmark predictor
PREDICTOR_PATH = "./model/shape_predictor_68_face_landmarks.dat"
detector = load_face_detector()
predictor = load_landmark_predictor(PREDICTOR_PATH)
print("Done, models loaded.")


def plot_face_rectangle(
    rect: Rectangle, color: str = "cyan", style: str = "-", alpha: float = 0.8
) -> None:
    plt.plot(
        [rect.left(), rect.left()],
        [rect.bottom(), rect.top()],
        style,
        color=color,
        alpha=alpha,
    )
    plt.plot(
        [rect.left(), rect.right()],
        [rect.bottom(), rect.bottom()],
        style,
        color=color,
        alpha=alpha,
    )
    plt.plot(
        [rect.left(), rect.right()],
        [rect.top(), rect.top()],
        style,
        color=color,
        alpha=alpha,
    )
    plt.plot(
        [rect.right(), rect.right()],
        [rect.top(), rect.bottom()],
        style,
        color=color,
        alpha=alpha,
    )


def plot_face_landmarks(
    points: Points, color: str = "red", style: str = ".", **kwargs
) -> None:
    for point in points:
        try:
            x, y = point.x, point.y
        except Exception:
            x, y = point[0], point[1]
        plt.plot(x, y, style, color=color, **kwargs)
    plt.gca().invert_yaxis()


def save_landmarks_to_disk(points: Points, fp: str) -> None:
    txt = "\n".join(list(map(lambda p: f"{p.x}, {p.y}", (points))))
    with open(fp, "w") as outfile:
        outfile.write(txt)


def glob_image_files(root: str, extensions: list[str] | None = None) -> list[str]:
    """Returns a list of image files in `root`"""
    if extensions is None:
        extensions = ["jpg", "jpeg", "png"]
    files = glob.glob(os.path.join(root, "*"))
    return [f for f in files if f.rsplit(".", 1)[-1].lower() in extensions]


def load_images(root: str, verbose: bool = True) -> dict[str, NDArray[np.float32]]:
    """Returns a dictionary of image arrays
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
        image = image.astype(np.float32) / 255.0
        images[file] = image
    return images


PointT = tuple[int, int]


def load_face_landmarks(root: str, verbose: bool = True) -> list[list[PointT]]:
    """Load face landmarks created by `detect_face_landmarks()`
    :param root: Path to folder containing CSV landmark files
    :param verbose: Toggle verbosity
    :output landmarks: List of landmarks for each face.
    """
    # List all files in the directory and read points from text files one by one
    all_paths = glob.glob(root.strip("/") + "/*_landmarks*")
    print(all_paths)
    landmarks: list[list[PointT]] = []
    for fn in all_paths:
        points: list[PointT] = []
        with open(fn) as file:
            for line in file:
                x, y = line.split(", ")
                points.append((int(x), int(y)))

        # Store array of points
        landmarks.append(points)
    return landmarks


def detect_face_landmarks(
    images: dict[str, NDArray[np.float32]],
    save_landmarks: bool = True,
    max_faces: int = 1,
    verbose: bool = True,
    print_freq: float = 0.10,
) -> tuple[list[list[PointT]], list[MatLike]]:
    """Detect and save the face landmarks for each image
    :param images: Dict of image files and arrays from `load_images()`.
    :param save_landmarks: Save landmarks to .CSV
    :param max_faces: Skip images with too many faces found.
    :param verbose: Toggle verbosity
    :param print_freq: How often do you want print statements?
    :output landmarks: 68 landmarks for each found face
    :output faces: List of the detected face images
    """
    num_images = len(images.keys())
    if verbose:
        print("\nStarting face landmark detection...")
        print(f"Processing {num_images} images.")
        N = max(round(print_freq * num_images), 1)

    # Look for face landmarks in each image
    num_skips = 0
    all_landmarks: list[list[PointT]] = []
    all_faces: list[np.ndarray] = []
    for n, (file, image) in enumerate(images.items()):
        if verbose and n % N == 0:
            print(f"({n + 1} / {num_images}): {file}")

        # Try to detect a face in the image
        imageForDlib = load_rgb_image(file)
        found_faces = detector(imageForDlib)

        # Only save landmarks when num_faces = 1
        if len(found_faces) == 0 or len(found_faces) > max_faces:
            num_skips += 1
            continue

        # Find landmarks, save to CSV
        for num, face in enumerate(found_faces):
            face.bottom()
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


def create_average_face(
    faces: list[MatLike],
    landmarks: list[list[PointT]],
    output_dims: tuple[int, int] = (600, 600),
    save_image: bool = True,
    output_file: str = "average_face.jpg",
    return_intermediates: bool = False,
    verbose: bool = True,
    print_freq: float = 0.05,
) -> tuple[
    MatLike | None,
    list[MatLike] | None,
    list[MatLike] | None,
    list[MatLike] | None,
]:
    """Combine the faces into an average face"""
    if verbose:
        print(f"\nStarting face averaging for {len(faces)} faces.")
    msg = "Number of landmark sets != number of images."
    assert len(faces) == len(landmarks), msg

    # Eye corners
    num_images = len(faces)
    n = len(landmarks[0])
    w, h = output_dims
    eyecornerDst = [(int(0.3 * w), int(h / 3)), (int(0.7 * w), int(h / 3))]
    imagesNorm: list[MatLike] = []
    pointsNorm: list[np.ndarray] = []

    # Add boundary points for delaunay triangulation
    boundaryPts = np.array(
        [
            (0, 0),
            (w / 2, 0),
            (w - 1, 0),
            (w - 1, h / 2),
            (w - 1, h - 1),
            (w / 2, h - 1),
            (0, h - 1),
            (0, h / 2),
        ]
    )

    # Initialize location of average points to 0s
    pointsAvg = np.array([(0, 0)] * (n + len(boundaryPts)), np.float32())

    # Warp images and transform landmarks to output coordinate system,
    # and find average of transformed landmarks.
    output: MatLike | None = None
    warped: list[MatLike] = []
    incremental: list[MatLike] = []
    N = max(round(print_freq * num_images), 1)
    for i in range(0, num_images):
        if verbose and i % N == 0:
            print(f"Image {i + 1} / {num_images}")

        # Corners of the eye in input image
        points1 = landmarks[i]
        eyecornerSrc = [landmarks[i][36], landmarks[i][45]]

        # Compute similarity transform
        tform = similarityTransform(eyecornerSrc, eyecornerDst)

        # Apply similarity transformation
        img_affine = cast(cv2.UMat, cv2.warpAffine(faces[i], tform, (w, h))).get()

        # Apply similarity transform on points
        points2 = np.reshape(np.array(points1), (68, 1, 2))
        points = cast(cv2.UMat, cv2.transform(points2, tform)).get()
        points = np.reshape(points, (68, 2)).astype(np.float32)

        # Append boundary points. Will be used in Delaunay Triangulation
        points = np.append(points, boundaryPts, axis=0)

        # Calculate location of average landmark points.
        pointsAvg = pointsAvg + points / num_images
        pointsNorm.append(points)
        imagesNorm.append(img_affine)

        # To see the dotted landmarks, use this:
        # plot_face_landmarks(pointsAvg)
        # Delaunay triangulation
        rect = (0, 0, w, h)
        dt = calculateDelaunayTriangles(rect, np.array(pointsAvg, dtype=np.float32))

        # Warp input images to average image landmarks
        output = np.zeros((h, w, 3), np.float32)
        for i in range(0, len(imagesNorm)):
            img: MatLike = np.zeros((h, w, 3), np.float32)
            # Transform triangles one by one
            for j in range(0, len(dt)):
                tin: list[PointT] = []
                tout: list[PointT] = []
                for k in range(0, 3):
                    pIn = constrainPoint(pointsNorm[i][dt[j][k]], w, h)
                    pOut = constrainPoint(pointsAvg[dt[j][k]], w, h)

                    tin.append(pIn)
                    tout.append(pOut)
                img = warpTriangle(imagesNorm[i], img, tin, tout)
            if return_intermediates:
                incremental.append((output + img) / (i + 1))

            # Add image intensities for averaging
            output = output + img

        # Divide by num_images to get average
        output = output / num_images

        if return_intermediates:
            warped.append(img_affine)
    incremental = incremental[-num_images:]
    print("Done.")

    # Save the output image to disk
    if save_image:
        assert output is not None
        cv2.imwrite(output_file, 255 * output[..., ::-1])
    if return_intermediates:  # For animated GIFs
        return output, warped, incremental, imagesNorm
    return output, None, None, None


def create_average_face_from_directory(
    dir_in: str,
    dir_out: str,
    filename: str,
    save_image: bool = True,
    verbose: bool = True,
    return_intermediates: bool = False,
) -> np.ndarray | None:
    if verbose:
        print(f"Directory: {dir_in}")
    images = load_images(dir_in, verbose=verbose)
    if len(images) == 0:
        if verbose:
            print(f"Couldn't find any images in: '{dir_in}'.")
        return None

    # Detect landmarks for each face
    landmarks, faces = detect_face_landmarks(images, verbose=verbose)

    # Use  the detected landmarks to create an average face
    fn = f"average_face_{filename}.jpg"
    fp = os.path.join(dir_out, fn).replace(" ", "_")
    average_face, _, _, _ = create_average_face(
        faces,
        landmarks,
        output_file=fp,
        save_image=True,
        return_intermediates=return_intermediates,
    )

    # Save a labeled version of the average face
    assert average_face is not None
    if save_image:
        save_labeled_face_image(average_face, filename, dir_out)
    return average_face


def save_labeled_face_image(
    image: NDArray, name: str, dir_out: str = "./", label: str = ""
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)

    # Add a title
    title = f"{name} average face"
    ax.set_title(title, fontsize=20, fontweight="heavy", color="gray", alpha=0.9)

    # Touch up the image
    ax.set(**{"xlabel": "", "ylabel": "", "xticks": [], "yticks": []})
    plt.tight_layout()

    x = image.shape[0] * 0.98
    y = image.shape[1] * 0.97
    ax.text(
        x, y, label, fontsize=17, color="black", weight="heavy", alpha=0.6, ha="right"
    )

    # Save the image
    fp = os.path.join(dir_out, f"average_face_{name}_labeled.png")
    fp = fp.replace(" ", "_")
    fig.savefig(fp, dpi=300)
    return


def create_animated_gif(
    path_to_images: str, save_gif: bool = True, verbose: bool = True
) -> tuple[FuncAnimation, MatLike]:
    """Create an animated face average GIF from a directory of images"""

    def save_to_file(
        gif: FuncAnimation,
        fn: str | None = None,
        fps: int | None = None,
        verbose: bool = True,
    ) -> None:
        fn = "animation" if not fn else fn
        if fps is None and hasattr(gif, "_interval"):
            fps = 1000 // getattr(gif, "_interval")
        elif fps is None:
            fps = 2
        if verbose:
            print(fn)

        gif.save(f"./{fn}.mov", writer="pillow", fps=fps)
        if verbose:
            print("Done saving the GIF.")

    # Load the images
    images = load_images(path_to_images, verbose=verbose)

    # Detect face landmarks
    landmarks, faces = detect_face_landmarks(images, verbose=verbose)

    # Create the average face and interim images
    averaged, warped, incremental, raw = create_average_face(
        faces, landmarks, return_intermediates=True, save_image=False, verbose=verbose
    )
    assert averaged is not None
    assert warped is not None
    assert incremental is not None
    assert raw is not None

    # Create the animation
    def tight(**kwargs):
        plt.tight_layout(**kwargs)

    # Make the plot
    fig, axs = plt.subplots(1, 2, figsize=(6, 3.7))
    lines = []
    titles = ["Individual", "Averaged"]
    for ax, title in zip(axs, titles):
        lines.append(ax.imshow(np.zeros_like(warped[0])))
        ax.axis("off")
        ax.set_title(title, fontsize=18)
    k = dict(pad=0, w_pad=0, h_pad=0)  # Layout values
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
    anim = FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=round(1.5 * len(raw)),
        interval=500,
        blit=True,
    )

    # Save the animation
    if save_gif:
        save_to_file(anim)
    return anim, averaged
