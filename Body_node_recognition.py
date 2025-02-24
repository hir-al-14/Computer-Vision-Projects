import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)  # Using camera index 0

if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Error: Ignoring empty camera frame")
            continue

        # Convert BGR to RGB and flip the image horizontally for a mirror effect
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        
        # Process the image to detect pose landmarks
        results = pose.process(image)

        # Convert image back to BGR for OpenCV to display it
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            h, w, z = image.shape

            # Iterate over all points (you can specify which points you want)
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                confidence = landmark.visibility

                # Show the index and confidence of the landmark
                cv2.putText(image, f"{idx} ({confidence:.2f})", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the result
        cv2.imshow("Mediapipe Pose", image)

        # Break the loop if the 'Esc' key is pressed
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
