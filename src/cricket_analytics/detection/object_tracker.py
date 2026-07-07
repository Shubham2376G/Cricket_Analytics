import cv2
from background_subtractor import BackgroundSubtractor
from contour_detector import ContourDetector
from hsv_filter import HSVFilter

class YellowObjectTracker:
    def __init__(self, video_path, min_area=5, min_circularity=0.4, draw_contours=True, draw_centroids=True):
        self.cap = cv2.VideoCapture(video_path)
        self.bg_subtractor = BackgroundSubtractor()
        self.contour_detector = ContourDetector(min_area=min_area, min_circularity=min_circularity)
        self.hsv_filter = HSVFilter(lower_bound=[10, 100, 100], upper_bound=[90, 255, 255])
        self.draw_contours = draw_contours
        self.draw_centroids = draw_centroids

    def process_frame(self, frame):
        fg_mask = self.bg_subtractor.apply(frame)
        
        mask_yellow = self.hsv_filter.apply(frame)
        moving_yellow_mask = cv2.bitwise_and(fg_mask, mask_yellow)
        
        contours, centroids = self.contour_detector.detect(moving_yellow_mask)
        
        if self.draw_contours:
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Area: {int(cv2.contourArea(contour))}, Circ: {self.contour_detector.min_circularity:.2f}",
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if self.draw_centroids:
            for cx, cy in centroids:
                cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)
        
        return frame

    def run(self,fr):
        while True:
            fr=fr+1
            ret, frame = self.cap.read()
            if not ret:
                break
            processed_frame = self.process_frame(frame)
            if fr>1:
            
              
                
                cv2.imshow('Yellow Moving Object Detection', processed_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage with optional drawing parameters
    tracker = YellowObjectTracker(
        video_path="front_ball1.mp4", 
        min_area=5, 
        min_circularity=0.4,
        draw_contours=True,  # Set to False to disable contour drawing
        draw_centroids=True,  # Set to False to disable centroid drawing
        # fr=0
    )
    tracker.run()
