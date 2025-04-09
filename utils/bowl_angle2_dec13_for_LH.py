import cv2
import json
import numpy as np
import math


def is_point_in_polygon(point, polygon):
    x, y = point
    return cv2.pointPolygonTest(np.array(polygon, np.int32), (x, y), False) >= 0

def draw_court_region(frame, court_polygon):
    cv2.polylines(frame, [np.array(court_polygon, np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

def draw_bbox(frame, bbox):
    if bbox and len(bbox) == 4:
        x, y, w, h = bbox
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 255), 2)

def draw_foot_keypoints(frame, keypoints, keypoint_scores, court_polygon):
    foot_indices = [15, 16]  # Indices for left and right feet
    for idx in foot_indices:
        if idx < len(keypoints) and idx < len(keypoint_scores):
            x, y = keypoints[idx]
            if is_point_in_polygon((int(x), int(y)), court_polygon):
                confidence = round(keypoint_scores[idx], 2)
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)  # Green circle for foot keypoints
                cv2.putText(frame, str(confidence), (int(x) + 10, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

def draw_hand_keypoints(frame, keypoints, keypoint_scores, court_polygon):
    #hand_indices = [10, 8, 6, 5]  # Indices for right hand keypoints
    hand_indices = [9,7,5,6 ]  # Indices for left hand keypoints
    for idx in range(len(hand_indices) - 1):  # Iterate through hand keypoints
        start_idx = hand_indices[idx]
        end_idx = hand_indices[idx + 1]

        if start_idx < len(keypoints) and end_idx < len(keypoints) and \
           start_idx < len(keypoint_scores) and end_idx < len(keypoint_scores):
            x1, y1 = keypoints[start_idx]
            x2, y2 = keypoints[end_idx]

            # Check if both points are inside the court polygon
            if is_point_in_polygon((int(x1), int(y1)), court_polygon) and \
               is_point_in_polygon((int(x2), int(y2)), court_polygon):
                confidence = round(keypoint_scores[start_idx], 2)
                
                # Draw line between consecutive keypoints
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                
                # Draw circles for each keypoint and add confidence score
                cv2.circle(frame, (int(x1), int(y1)), 5, (255, 0, 0), -1)  # Blue circle for hand keypoints
                cv2.putText(frame, str(confidence), (int(x1) + 10, int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                # Similarly for the second point
                confidence = round(keypoint_scores[end_idx], 2)
                cv2.circle(frame, (int(x2), int(y2)), 5, (255, 0, 0), -1)  # Blue circle for hand keypoints
                cv2.putText(frame, str(confidence), (int(x2) + 10, int(y2) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

def calculate_angle(p1, p2, p3):
    """
    Calculate the angle between three points (p1, p2, p3)
    p1, p2, p3: (x, y) tuples
    """
    
    p1p2 = np.array(p1) - np.array(p2)
    p2p3 = np.array(p3) - np.array(p2)
    cos_angle = np.dot(p1p2, p2p3) / (np.linalg.norm(p1p2) * np.linalg.norm(p2p3))
    angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    return angle_deg
    

def draw_angle_arc(frame, p1, p2, p3, angle_deg):
    """
    Draw an arc showing the angle formed by points p1, p2, p3 on the frame.
    p1, p2, p3: (x, y) tuples
    angle_rad: angle in radians
    """
    # Calculate center of the angle (at point p2)
    center = (int(p2[0]), int(p2[1]))

    # Get the radius (distance from p2 to p1 or p2 to p3)
    #radius = int(np.linalg.norm(np.array([p2[0] - p1[0], p2[1] - p1[1]])))
    radius = 20  # Adjust radius as needed
    # Calculate the start and end points for the arc
    #start_angle = 0
    #end_angle = np.degrees(angle_rad)
    start_angle = int(np.degrees(np.arctan2(p1[1]-p2[1], p1[0]-p2[0])))
    end_angle = int(np.degrees(np.arctan2(p3[1]-p2[1], p3[0]-p2[0])))

# Draw the lines connecting the key joints
    
    # Use cv2.ellipse to draw the arc
    cv2.ellipse(frame, center, (radius, radius), 0, start_angle, end_angle, (0, 255, 0), 2)
     # Annotate the angle value
    text_position = (center[0] + 40, center[1] - 10)
    cv2.putText(
        frame, f"{angle_deg:.1f}degrees", text_position,
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1
    )


def detect_bowler(frame, keypoints, keypoint_scores, court_polygon,bowler_frame_count):
    #bowler_frame_count= 0
   # hand_indices = [10, 8, 6, 5]  # Indices for 10th, 8th, 6th, and 5th keypoints
    hand_indices = [9,7,5,6]  # Indices for left hand keypoints
    if all(idx < len(keypoints) and idx < len(keypoint_scores) for idx in hand_indices):
        p1 = (keypoints[hand_indices[0]][0], keypoints[hand_indices[0]][1])  # 10th keypoint
        p2 = (keypoints[hand_indices[1]][0], keypoints[hand_indices[1]][1])  # 8th keypoint
        p3 = (keypoints[hand_indices[2]][0], keypoints[hand_indices[2]][1])  # 6th keypoint
        p4 = (keypoints[hand_indices[3]][0], keypoints[hand_indices[3]][1])  # 5th keypoint
        
        # Check y-value conditions (10th < 8th, 8th < 6th, 6th < 5th)
        if p1[1] < p2[1] and p2[1] < p3[1] and p3[1] < p4[1]:
            # Calculate the angle
            angle = calculate_angle(p1, p2, p3)
                    # Draw the angle arc on the frame
            draw_angle_arc(frame, p1, p2, p3, angle)

            # Check if the angle is close to 180 degrees (straight line)
            if 165 < angle < 180:
                # Draw angle on frame and mark the player as bowler
                cv2.putText(frame, 'Bowler', (int(p2[0] - 10), int(p2[1] - 25)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 0, 255), 2)
                cv2.line(frame, (int(p2[0]), int(p2[1])), (int(p3[0]), int(p3[1])), (0, 0, 255), 2)
                cv2.line(frame, (int(p3[0]), int(p3[1])), (int(p4[0]), int(p4[1])), (0, 0, 255), 2)
                                # Call draw_angle_arc to visualize the angle on the frame
                draw_angle_arc(frame, p1, p2, p3,angle)
                if cv2.pointPolygonTest(np.array(court_polygon), p2, False) >= 0:
                    # Save the frame if the bowler is inside the court polygon
                    frame_filename = f"{output_folder}/bowler_{bowler_frame_count}_frame.jpg"
                    cv2.imwrite(frame_filename, frame)
                    #bowler_frame_count +=1
                    return True
                return True
    return False

def detect_releasing_frame(frame, keypoints, keypoint_scores, court_polygon):
   # hand_indices = [10, 8, 6, 5]  # Indices for 10th, 8th, 6th, and 5th keypoints
    hand_indices = [9,7,5,6 ]  # Indices for left hand keypoints
    if all(idx < len(keypoints) and idx < len(keypoint_scores) for idx in hand_indices):
        p1 = (keypoints[hand_indices[0]][0], keypoints[hand_indices[0]][1])  # 10th keypoint
        p2 = (keypoints[hand_indices[1]][0], keypoints[hand_indices[1]][1])  # 8th keypoint
        p3 = (keypoints[hand_indices[2]][0], keypoints[hand_indices[2]][1])  # 6th keypoint
        p4 = (keypoints[hand_indices[3]][0], keypoints[hand_indices[3]][1])  # 5th keypoint

        # Check if the 10th keypoint has the lowest Y value
        if p1[1] < p2[1] and p1[1] < p3[1] and p1[1] < p4[1]:
            # Calculate the angle between the 10th, 8th, and 6th keypoints (for vertical alignment)
            angle_10_8_6 = calculate_angle(p1, p2, p3)
            angle_8_6_5 = calculate_angle(p2, p3, p4)

            # Check if the angle between the points is close to 90 degrees (vertical line)
            if abs(angle_10_8_6 - 180) < 15 and abs(angle_8_6_5 - 180) < 15:
                # Draw angle and mark the frame as releasing
                cv2.putText(frame, 'Releasing', (int(p2[0] - 10), int(p2[1] - 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 0, 255), 2)
                cv2.line(frame, (int(p2[0]), int(p2[1])), (int(p3[0]), int(p3[1])), (0, 0, 255), 2)
                cv2.line(frame, (int(p3[0]), int(p3[1])), (int(p4[0]), int(p4[1])), (0, 0, 255), 2)
                return True
    return False


def simulate_from_json(video_path, json_path, output_path, output_json_path, court_polygon, output_folder):
    with open(json_path, 'r') as f:
        data = json.load(f)

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    frame_index = 0
    keypoints_data = []
    bowler_frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        draw_court_region(frame, court_polygon)

        if frame_index < len(data):
            frame_data = data[frame_index]
            instances = frame_data['instances']

            for instance in instances:
                keypoints = instance.get('keypoints', [])
                keypoint_scores = instance.get('keypoint_scores', [])

                if keypoints and keypoint_scores:
                    draw_bbox(frame, instance.get('bbox', []))
                    draw_foot_keypoints(frame, keypoints, keypoint_scores, court_polygon)
                    draw_hand_keypoints(frame, keypoints, keypoint_scores, court_polygon)
                  
                    # Detect if the player is a bowler
                    if detect_bowler(frame, keypoints, keypoint_scores, court_polygon,bowler_frame_count):
                        bowler_frame_count += 1
                        #output_bowler_frame_path = f"{output_folder}/bowler_frame_{bowler_frame_count}.jpg"
                        #cv2.imwrite(output_bowler_frame_path, frame)
                                            # Example of calculating and drawing the angle arc for keypoints (e.g., wrist, elbow, shoulder)
                        
        out.write(frame)
        frame_index += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    with open(output_json_path, 'w') as f:
        json.dump(data, f, indent=4)


# Example Usage
video_path = r"D:\IIT_Madras_Task\Pavani_cricket\CRICKET DATA COLLECTION\bowlers_vid\Swapnil Singh - Left arm Orthodox - Trim.mp4"
json_path = r"D:\IIT_Madras_Task\mmpose\bowlerstatic_cric2\results_Swapnil Singh - Left arm Orthodox - Trim.json"
output_path = r"D:\IIT_Madras_Task\Pavani_cricket\CRICKET DATA COLLECTION\HAWKEYE_OUTPUTS\SwapnilSingh_output_keysimu_player.avi"
output_json_path = r'D:\IIT_Madras_Task\mmpose\playershawkeye2\outputforSwapnilSingh_hand&foot_keypoints.json'
#court_polygon = [(739, 342), (1216, 342), (1216, 820), (739, 820)]#rakshidar
court_polygon = [(476, 178), (794, 178), (794, 460), (476, 460)]#SwapnilSingh
#court_polygon = [(476, 174), (797, 174), (797, 455), (476, 455)]#Mohit
#court_polygon = [(714, 255), (1186, 255), (1186, 657), (714, 657)]#SuyushSharma

output_folder = r"D:\IIT_Madras_Task\Pavani_cricket\CRICKET DATA COLLECTION\HAWKEYE_OUTPUTS\SwapnilSingh_bowler_det"
simulate_from_json(video_path, json_path, output_path, output_json_path, court_polygon, output_folder)


