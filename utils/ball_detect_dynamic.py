import cv2
import numpy as np

ball_cen = []

def scale_frame(frame, scale=0.20):
    height, width = frame.shape[:2]
    new_width = int(width * scale)
    new_height = int(height * scale)
    return cv2.resize(frame, (new_width, new_height))

def detect_white_objects_in_video(frame, scale=0.75):
    global ball_cen  # Use the global list to track ball centers

    # Scale the frame for display
    scaled_frame = scale_frame(frame, scale)

    # Convert the frame from BGR to HSV color space
    hsv_frame = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2HSV)

    # Define the range for white color in HSV
    lower_white = np.array([0, 0, 200])  # Slightly higher minimum brightness
    upper_white = np.array([180, 40, 255])  # Slightly lower maximum saturation

    # Create a mask for white objects
    white_mask = cv2.inRange(hsv_frame, lower_white, upper_white)

    # Find contours of the white objects
    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours and bounding boxes on the frame
    for contour in contours:
       
        x, y, w, h = cv2.boundingRect(contour)
        bounding_box_area = w * h
        area = cv2.contourArea(contour)
        _, radius = cv2.minEnclosingCircle(contour)
        circularity = area / (np.pi * (radius ** 2))
        p = (x + w)
        q = (y + h)
        cx = (x + p) // 2
        cy = (y + q) // 2
        aspect_ratio = w / h

        if 5 < area < 100 and circularity > 0.4:
            white_pixel_count = cv2.contourArea(contour)

            # Calculate the white pixel ratio
            white_pixel_ratio = white_pixel_count / bounding_box_area

            if white_pixel_ratio > 0.2:
                # Draw the bounding box
                cv2.rectangle(scaled_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                ball_cen.append((cx, cy))

    # Display the results
    # out.write(frame) 
    cv2.imshow("Original Frame with Detections", scaled_frame)
    cv2.imshow("White Mask", white_mask)

    # Wait for a key press
    cv2.waitKey(1)

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot fetch the frame.")
            break

        frame_id += 1
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter('dynamic_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
       
        

        # Convert to HSV for pitch detection
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 150])
        upper_white = np.array([180, 50, 255])
        white_mask = cv2.inRange(hsv_frame, lower_white, upper_white)
        contours, _ = cv2.findContours(white_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        pitch_detected = False

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            aspect_ratio = w / h

            if area > 10000 and 0.9 <= aspect_ratio <= 1.1:
                pitch_detected = True
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Pitch", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # If pitch is detected (event = 1), run the secondary detection
        if pitch_detected:
            print("Event: 1 (Pitch detected)")
            detect_white_objects_in_video(frame, scale=0.75)
        else:
            print("Event: 0 (Pitch not detected)")
        
        
        out.write(frame)
        # cv2.imshow("Frame", frame)

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    cv2.destroyAllWindows()



#Example usage

video_path = "video3.mp4"  # Path to the video file
# Call the main function
process_video(video_path)