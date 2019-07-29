import cv2
import dlib
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import glob

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
    files = glob_image_files(root)
    num_files = len(files)
    if verbose:
        print(f"Found {num_files} in '{root}'.")
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

def detect_face_landmarks(images, save_landmarks=True, max_faces=1, verbose=True):
    """Detect and save the face landmarks for each image
    :param images: (dict) Dict of image files and arrays from `load_images()`.
    :param save_landmarks: (bool) Save landmarks to .CSV
    :param max_faces: (int) Skip images with too many faces found.
    :param verbose: (bool) Toggle verbosity
    :output landmarks: (list) 68 landmarks for each found face
    :output faces: (list) List of the detected face images
    """
    num_images = len(images.keys())
    if verbose:
        print(f"Starting face landmark detection...")
        print(f"Processing {num_images}.")
        N = max(round(0.10 * num_images), 1)

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
        face = found_faces[0] # Kludge for now, just take first face
        all_faces.append(image)
        landmarks = predictor(imageForDlib, face)
        if not landmarks:
            num_skips += 1
            continue

        # Save landmarks as a CSV file (optional)
        fp = file.rsplit(".", 1)[0] + "_landmarks.csv"
        if save_landmarks:
            save_landmarks_to_disk(landmarks.parts(), fp=fp)

        # Convert landmarks to list of (x, y) tuples
        lm = [(point.x, point.y) for point in landmarks.parts()]
        all_landmarks.append(lm)

    if verbose:
        print(f"Skipped {100 * (num_skips / num_images):.1f}% of images.")
    return all_landmarks, all_faces

#   --------------------------------------------------------   #
#            IMAGE ALIGNMENT AND AVERAGING FUNCTIONS           #
#   --------------------------------------------------------   #

# Compute similarity transform given two sets of two points.
# OpenCV requires 3 pairs of corresponding points.
# We are faking the third one.
def similarityTransform(inPoints, outPoints) :
    s60 = math.sin(60*math.pi/180);
    c60 = math.cos(60*math.pi/180);
    inPts = np.copy(inPoints).tolist();
    outPts = np.copy(outPoints).tolist();

    xin = c60*(inPts[0][0] - inPts[1][0]) - s60*(inPts[0][1] - inPts[1][1]) + inPts[1][0];
    yin = s60*(inPts[0][0] - inPts[1][0]) + c60*(inPts[0][1] - inPts[1][1]) + inPts[1][1];
    inPts.append([np.int(xin), np.int(yin)]);

    xout = c60*(outPts[0][0] - outPts[1][0]) - s60*(outPts[0][1] - outPts[1][1]) + outPts[1][0];
    yout = s60*(outPts[0][0] - outPts[1][0]) + c60*(outPts[0][1] - outPts[1][1]) + outPts[1][1];
    outPts.append([np.int(xout), np.int(yout)]);

    tform = cv2.estimateAffinePartial2D(np.array([inPts]), np.array([outPts]))
    return cv2.UMat(tform[0])

# Check if a point is inside a rectangle
def rectContains(rect, point):
    if point[0] < rect[0]:
        return False
    if point[1] < rect[1]:
        return False
    if point[0] > rect[2]:
        return False
    if point[1] > rect[3]:
        return False
    return True

def constrainPoint(p, w, h):
    p = (min(max(p[0], 0), w - 1 ),
         min(max(p[1], 0), h - 1 ))
    return p

# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size):
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None,
                         flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)
    return dst

def calculateDelaunayTriangles(rect, points):
    # Insert points into subdiv
    subdiv = cv2.Subdiv2D(rect)
    for p in points:
        subdiv.insert((p[0], p[1]))

    # List of triangles. Each triangle is a list of 3 points ( 6 numbers )
    triangleList = subdiv.getTriangleList()

    # Find the indices of triangles in the points array
    delaunayTri = []
    for t in triangleList:
        pt = []
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
            ind = []
            for j in range(0, 3):
                for k in range(0, len(points)):
                    if(abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
                        ind.append(k)
            if len(ind) == 3:
                delaunayTri.append((ind[0], ind[1], ind[2]))
    return delaunayTri

# Warps and alpha blends triangular regions from img1 and img2 to img
def warpTriangle(img1, img2, t1, t2):
    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by left top corner of the respective rectangles
    t1Rect, t2Rect, t2RectInt = [], [], []
    for i in range(0, 3):
        t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        t2RectInt.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    size = (r2[2], r2[3])
    img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)
    img2Rect = img2Rect * mask

    # Copy triangular region of the rectangular patch to the output image
    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * ((1.0, 1.0, 1.0) - mask)
    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] + img2Rect
    return img2

def create_average_face(faces, landmarks,
                        output_dims=(800, 800),
                        save_image=True,
                        output_file="average_face.jpg",
                        verbose=True):
    """Combine the faces into an average face"""
    if verbose:
        print(f"Starting face averaging for {len(faces)} faces.")
    msg = "Number of landmark sets != number of images."
    assert len(faces) == len(landmarks), msg

    # Eye corners
    num_images = len(faces)
    n = len(landmarks[0])
    w, h = output_dims
    eyecornerDst = [(np.int(0.3 * w), np.int(h / 3)), (np.int(0.7 * w), np.int(h / 3))]
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
    N = max(round(0.10 * num_images), 1)
    for i in range(0, num_images):
        if verbose and i % N == 0:
            print(f"Image {i} / {num_images}")

        # Corners of the eye in input image
        points1 = landmarks[i]
        eyecornerSrc  = [landmarks[i][36], landmarks[i][45]]

        # Compute similarity transform
        tform = similarityTransform(eyecornerSrc, eyecornerDst)

        # Apply similarity transformation
        img = cv2.warpAffine(faces[i], tform, (w, h)).get()

        # Apply similarity transform on points
        points2 = np.reshape(np.array(points1), (68, 1, 2))
        points = cv2.transform(points2, tform).get()
        points = np.float32(np.reshape(points, (68, 2)))

        # Append boundary points. Will be used in Delaunay Triangulation
        points = np.append(points, boundaryPts, axis=0)

        # Calculate location of average landmark points.
        pointsAvg = pointsAvg + points / num_images
        pointsNorm.append(points)
        imagesNorm.append(img)

        # Delaunay triangulation
        rect = (0, 0, w, h);
        dt = calculateDelaunayTriangles(rect, np.array(pointsAvg))

        # Warp input images to average image landmarks
        output = np.zeros((h,w,3), np.float32())
        for i in range(0, len(imagesNorm)):
            img = np.zeros((h,w,3), np.float32())
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

            # Add image intensities for averaging
            output = output + img

        # Divide by num_images to get average
        output = output / num_images
    print('Done.')

    # Save the output image to disk
    if save_image:
        cv2.imwrite(output_file, 255 * output[..., ::-1])

    return output
