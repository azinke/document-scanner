"""Documemt Scanner."""
import cv2 as cv
import numpy as np
import copy
import matplotlib.pyplot as plt
import argparse
import sys

    # w:480, h:512
IMAGE_WIDTH: int = 720
IMAGE_HEIHGT: int = 768


class Processor(object):
    """Processor.

    This class describe each stage in the Image processinf pipeline.
    The default behavior of the stage is to have pass the input image as
    output when no specific operation is defined. It's then acting like a
    buffer
    """

    def __init__(self, image) -> None:
        """Processor constructor.
        Parameter:
            image: The input image of the stage
        """
        self.original_image = image
        self.input_image = copy.deepcopy(image)
        self.output_image = copy.deepcopy(image)

    def show(self) -> None:
        """Display the output image."""
        cv.imshow(str(self.__class__), self.output_image)

    def process(self):
        """Process the input image.
        
        Return by default the same image as the input image
        """
        return self.output_image


class Resizer(Processor):
    """Resizer.

    Resize the input image.
    """

    WIDTH: int = IMAGE_WIDTH
    HEIHGT: int = IMAGE_HEIHGT

    def process(self):
        """Convert the input image into grayscale."""
        self.output_image = cv.resize(
            self.input_image,
            (self.WIDTH, self.HEIHGT),
        )
        return self.output_image


class Grayscale(Processor):
    """Grayscale.
    
    Process an image into grayscale
    """

    def process(self):
        """Convert the input image into grayscale."""
        self.output_image = cv.cvtColor(self.input_image, cv.COLOR_BGR2GRAY)
        return self.output_image


class Blur(Processor):
    """Blur.

    Blur the image to reduice noise
    """

    def process(self):
        """Aplly a Gaussion blur to the input image."""
        self.output_image = cv.GaussianBlur(self.input_image, (7, 7), 0)
        return self.output_image


class Threshold(Processor):
    """Threshold.

    Perform the thresholding of the image
    """

    MIN_THRESHOLD: int = 50
    MAX_PIXEL_VALUE: int = 255
    BLOCK_SIZE: int = 11
    CONSTANT: int = 3

    def process(self):
        """Binarization of the image"""
        self.output_image = cv.adaptiveThreshold(
            self.input_image,
            # self.MIN_THRESHOLD,
            self.MAX_PIXEL_VALUE,
            cv.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv.THRESH_BINARY_INV,
            self.BLOCK_SIZE,
            self.CONSTANT,
        )
        return self.output_image


class Morphology(Processor):
    """Morphology.

    Try to erode background from the image
    """

    def process(self):
        """Perform erosion and dilation operations on the image."""
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        # np.ones((5, 5), np.uint8)
        # self.output_image = cv.morphologyEx(self.output_image, cv.MORPH_OPEN, kernel, iterations=1)
        self.output_image = cv.morphologyEx(self.output_image, cv.MORPH_GRADIENT, kernel, iterations=1)
        self.output_image = cv.morphologyEx(self.output_image, cv.MORPH_CLOSE, kernel, iterations=3)
        return self.output_image


class EdgeDetector(Processor):
    """Edges detection.

    Perform edges detection on the input image
    """

    MIN_THRESHOLD: int = 25
    MAX_THRESHOLD: int = 200

    def process(self):
        """Detect edges from image"""
        self.output_image = cv.Canny(
            self.input_image,
            self.MIN_THRESHOLD,
            self.MAX_THRESHOLD,
        )
        return self.output_image


class ContourDetector(Processor):
    """Contour detection."""

    def process(self):
        """Detect edges from image"""
        contours, hierachy = cv.findContours(
            self.input_image,
            cv.RETR_LIST,
            cv.CHAIN_APPROX_SIMPLE,
        )
        contours = sorted(contours, key=cv.contourArea, reverse=True)
        target = None
        for c in contours:
            path = cv.arcLength(c, True)
            approx = cv.approxPolyDP(c, 0.1*path, True)
            if len(approx) == 4:
                # print(approx)
                target = approx
                break
        if target is not None:
            cv.drawContours(
                self.original_image,
                [target],
                0,
                (255, 255, 255),
                5
            )
        self.output_image = copy.deepcopy(self.original_image)
        return (self.output_image, target)


class CornerDetector(Processor):
    """Corner detector."""

    def process(self):
        """Detect the corners of the document from the image."""
        self.output_image = cv.cornerHarris(self.input_image, 2, 3, 0.04)
        self.output_image = cv.dilate(self.output_image, kernel=None, iterations=2)
        return self.output_image


class LineDetector(Processor):
    """Line detector."""

    def process(self):
        """Detect the borders of the document from the image."""
        lines = cv.HoughLinesP(self.input_image, 1, np.pi/180, 100, 10, 150, 400)
        for x1,y1,x2,y2 in lines[0]:
            cv.line(self.output_image,(x1,y1),(x2,y2),(0,255,0),2)
        cv.line(self.output_image, (0,0), (100, 100), (0, 0, 255), 4)
        return self.output_image


class Scanner(object):
    """Handle document scan."""

    WIDTH: int = IMAGE_WIDTH
    HEIHGT: int = IMAGE_HEIHGT

    def __init__(self) -> None:
        """Constructor."""
        self.pipeline = [
            Resizer,
            Grayscale,
            Blur,
            Threshold,
            Morphology,
            EdgeDetector,
            # ContourDetector,
            # CornerDetector,
            # LineDetector,
        ]

    def run(self, image_filename: str) -> None:
        """Execute the pipeline"""
        input_img = cv.imread(image_filename)
        for processor in self.pipeline:
            handler = processor(input_img)
            input_img = handler.process()
            handler.show()
        _, corners = ContourDetector(input_img).process()
        # Flaten the array
        #
        # The output of the corner detector is an array of 2-level nested items
        # eg.: [ [[x0, y0]],  [[x1, y1]], [[x2, y2]], [[x3, y3]] ]
        # Flatenning would allow to have just a 4x2 matrix. Each row representing
        # the corners
        corners = np.array([l[0] for l in corners])

        """
        Processing the image
        """
        image = cv.imread(image_filename)
        # Resize the image
        resizer = Resizer(image)
        image = resizer.process()
        # Produce grayscale image
        grayscaler = Grayscale(image)
        image = grayscaler.process()
        # Extract the document
        document = self.extract_document(image, corners)
        # Image binarization
        document = cv.adaptiveThreshold(
            document,
            255,
            cv.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv.THRESH_BINARY,
            11, 7,
        )
        cv.imshow("Document", document)

    def reorder(self, corners) -> np.array:
        """Reorder the corners.

        There is actually no guarantee that the corners would appear in the
        order that would allow them to be used for perspective transformation.
        So, they need to be re-ordered
        """
        ordered_corners = np.zeros((4, 2), dtype=np.float32)
        sums = corners.sum(1)
        ordered_corners[0] = corners[np.argmin(sums)]
        ordered_corners[2] = corners[np.argmax(sums)]
        diffs = np.diff(corners, axis=1)
        ordered_corners[1] = corners[np.argmin(diffs)]
        ordered_corners[3] = corners[np.argmax(diffs)]
        return ordered_corners

    def extract_document(self, image, corners):
        """"Extract the document given its corners."""
        # Re-order corners
        corners = self.reorder(corners)
        output_corners = np.float32([
            [0, 0],
            [self.WIDTH, 0],
            [self.WIDTH, self.HEIHGT],
            [0, self.HEIHGT],
        ])
        # Transformation matrix
        TF = cv.getPerspectiveTransform(corners, output_corners)
        return cv.warpPerspective(image, TF, (self.WIDTH, self.HEIHGT))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Scanner",
        description="Extract a document from an image."
    )
    parser.add_argument(
        "--file", "-f",
        help="Path to the file to process",
        type=str,
    )
    parser.add_argument(
        "--directory", "-d",
        help="Path to a directory containing all the pictures to process",
        type=str,
    )
    args = parser.parse_args()
    if args.file:
        scanner = Scanner()
        scanner.run(args.file)
        cv.waitKey()
        sys.exit(0)

    parser.print_help()
