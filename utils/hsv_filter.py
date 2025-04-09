import cv2
import numpy as np

class HSVFilter:
    def __init__(self, lower_bound, upper_bound):
        """
        Initializes the HSV filter with specified lower and upper HSV bounds.

        Parameters:
        - lower_bound: List or tuple of 3 ints (H, S, V) for the lower threshold.
        - upper_bound: List or tuple of 3 ints (H, S, V) for the upper threshold.
        """
        self.lower_bound = np.array(lower_bound, dtype=np.uint8)
        self.upper_bound = np.array(upper_bound, dtype=np.uint8)

    def apply(self, frame):
        """
        Applies HSV filtering to the input frame.

        Parameters:
        - frame: The input image/frame in BGR color space.

        Returns:
        - A binary mask where white pixels represent areas within the HSV range.
        """
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv_frame, self.lower_bound, self.upper_bound)


# Example usage
import cv2

cap = cv2.VideoCapture(0)
# Example: Filtering out a blue object
blue_filter = HSVFilter(lower_bound=(100, 150, 50), upper_bound=(140, 255, 255))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    mask = blue_filter.apply(frame)
    cv2.imshow("Blue Object Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
