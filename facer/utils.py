import cv2
import numpy as np
import math

#   --------------------------------------------------------   #
#            IMAGE ALIGNMENT AND AVERAGING FUNCTIONS           #
#   --------------------------------------------------------   #


# Compute similarity transform given two sets of two points.
# OpenCV requires 3 pairs of corresponding points.
# We are faking the third one.
def similarityTransform(inPoints, outPoints):
    s60 = math.sin(60 * math.pi / 180)
    c60 = math.cos(60 * math.pi / 180)
    inPts = np.copy(inPoints).tolist()
    outPts = np.copy(outPoints).tolist()

    xin = (
        c60 * (inPts[0][0] - inPts[1][0])
        - s60 * (inPts[0][1] - inPts[1][1])
        + inPts[1][0]
    )
    yin = (
        s60 * (inPts[0][0] - inPts[1][0])
        + c60 * (inPts[0][1] - inPts[1][1])
        + inPts[1][1]
    )
    inPts.append([int(xin), int(yin)])

    xout = (
        c60 * (outPts[0][0] - outPts[1][0])
        - s60 * (outPts[0][1] - outPts[1][1])
        + outPts[1][0]
    )
    yout = (
        s60 * (outPts[0][0] - outPts[1][0])
        + c60 * (outPts[0][1] - outPts[1][1])
        + outPts[1][1]
    )
    outPts.append([int(xout), int(yout)])

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
    p = (min(max(p[0], 0), w - 1), min(max(p[1], 0), h - 1))
    return p


# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size):
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(
        src,
        warpMat,
        (size[0], size[1]),
        None,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
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
        if (
            rectContains(rect, pt1)
            and rectContains(rect, pt2)
            and rectContains(rect, pt3)
        ):
            ind = []
            for j in range(0, 3):
                for k in range(0, len(points)):
                    # Hyperparameter
                    if (
                        abs(pt[j][0] - points[k][0]) < 0.001
                        and abs(pt[j][1] - points[k][1]) < 0.001
                    ):
                        ind.append(k)
            if len(ind) == 3:
                delaunayTri.append((ind[0], ind[1], ind[2]))
            elif len(ind) == 4:
                delaunayTri.append((ind[0], ind[1], ind[2]))
                delaunayTri.append((ind[1], ind[2], ind[3]))
                # May need to use this too?
                # delaunayTri.append((ind[2], ind[3], ind[0]))
                # delaunayTri.append((ind[3], ind[0], ind[1]))
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
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1] : r1[1] + r1[3], r1[0] : r1[0] + r1[2]]
    size = (r2[2], r2[3])
    img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)
    img2Rect = img2Rect * mask

    # Copy triangular region of the rectangular patch to the output image
    img2[r2[1] : r2[1] + r2[3], r2[0] : r2[0] + r2[2]] = img2[
        r2[1] : r2[1] + r2[3], r2[0] : r2[0] + r2[2]
    ] * ((1.0, 1.0, 1.0) - mask)
    img2[r2[1] : r2[1] + r2[3], r2[0] : r2[0] + r2[2]] = (
        img2[r2[1] : r2[1] + r2[3], r2[0] : r2[0] + r2[2]] + img2Rect
    )
    return img2
