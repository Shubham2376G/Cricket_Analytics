import cv2

class BackgroundSubtractor:
    def __init__(self, history=1000, varThreshold=50, detectShadows=True):
        """
        Initializes the BackgroundSubtractor using MOG2 algorithm.

        Parameters:
        - history: Length of the history.
        - varThreshold: Threshold on the squared Mahalanobis distance to decide
                        whether it is well described by the background model.
        - detectShadows: If True, the algorithm detects shadows.
        """
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=varThreshold,
            detectShadows=detectShadows
        )

    def apply(self, frame):
        """
        Applies the background subtraction on the provided frame.

        Parameters:
        - frame: The input image/frame (grayscale or color).

        Returns:
        - The foreground mask.
        """
        return self.bg_subtractor.apply(frame)



#Example usage
import cv2

cap = cv2.VideoCapture(0)
bg_subtractor = BackgroundSubtractor()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    fg_mask = bg_subtractor.apply(frame)
    cv2.imshow("Foreground Mask", fg_mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
