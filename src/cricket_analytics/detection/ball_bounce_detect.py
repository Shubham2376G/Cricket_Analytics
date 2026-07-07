import cv2
import numpy as np

class CricketBallTracker:
    def __init__(self, video_path, output_path, skip_frames=100):
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (self.width, self.height))
        
        self.backgroundObject = cv2.createBackgroundSubtractorKNN(detectShadows=False)
        self.skip_frames = skip_frames
        self.frame_no = 0
        self.ball_cen = []
        self.bounce_points = []

    def detect_ball(self, img):
        """Detects the ball and returns its coordinates."""
        fgmask = self.backgroundObject.apply(img)
        _, thresh1 = cv2.threshold(fgmask, 10, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_balls = []

        for contour in contours:
            area = cv2.contourArea(contour)
            _, radius = cv2.minEnclosingCircle(contour)
            circularity = area / (np.pi * (radius ** 2))
            x, y, w, h = cv2.boundingRect(contour)
            cx, cy = x + w // 2, y + h // 2
            aspect_ratio = w / h

            if 30 < area < 900 and circularity > 0.6:
                roi_gray = cv2.cvtColor(img[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
                white_pixel_ratio = np.count_nonzero(roi_gray) / (w * h)

                if white_pixel_ratio > 0.3 and 290 < cy < 623 and 738 < cx < 1149:
                    detected_balls.append((cx, cy))

        return detected_balls

    def detect_bounce(self):
        """Detects the bounce point based on trajectory change."""
        if len(self.ball_cen) < 3:
            return None

        last_3_points = self.ball_cen[-3:]
        frame_1, frame_2, frame_3 = last_3_points

        if frame_2[1] > frame_1[1] and frame_2[1] > frame_3[1]:
            bb1, bb2 = frame_2
            if 915 < bb1 < 1194 and 311 < bb2 < 497:
                return (bb1, bb2)

        return None

    def draw_trajectory(self, img):
        """Draws the trajectory and bounce points on the frame."""
        for center in self.ball_cen:
            cv2.circle(img, center, 6, (0, 255, 0), -1)  # Green for trajectory

        for bp in self.bounce_points:
            cv2.circle(img, bp, 7, (255, 0, 0), 2)  # Blue for bounce points
            cv2.putText(img, "Bounce Frame", (bp[0] + 10, bp[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    def process_video(self):
        """Processes the entire video to detect ball trajectory and bounce points."""
        while True:
            success, img = self.cap.read()
            if not success:
                break

            self.frame_no += 1
            if self.frame_no <= self.skip_frames:
                continue

            detected_balls = self.detect_ball(img)
            if detected_balls:
                self.ball_cen.extend(detected_balls)

            bounce = self.detect_bounce()
            if bounce and bounce not in self.bounce_points:
                self.bounce_points.append(bounce)

            self.draw_trajectory(img)
            self.out.write(img)

            cv2.imshow("Processed Video", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    video_path = "Rasikh Dar - Right arm fast medium.mp4"
    output_path = "output_video_rcb.mp4"

    tracker = CricketBallTracker(video_path, output_path)
    tracker.process_video()