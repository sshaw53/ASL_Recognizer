"""
A game program uses hand tracking to 
determine ASL letters.

@author: Sierra Shaw

"""

import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2

BLUE = (0, 0, 255)

# Library Constants
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkPoints = mp.solutions.hands.HandLandmark
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
DrawingUtil = mp.solutions.drawing_utils
     
class ASL:
    def __init__(self):

        # Create the hand detector
        base_options = BaseOptions(model_asset_path='data/hand_landmarker.task')
        options = HandLandmarkerOptions(base_options=base_options,
                                                num_hands=1)
        self.detector = HandLandmarker.create_from_options(options)

        # Load video
        self.video = cv2.VideoCapture(0)

        # Letter
        self.letter = ''
    
    def draw_landmarks_on_hand(self, image, detection_result):
        """
        Draws all the landmarks on the hand
        Args:
            image (Image): Image to draw on
            detection_result (HandLandmarkerResult): HandLandmarker detection results
        """
        # Get a list of the landmarks
        hand_landmarks_list = detection_result.hand_landmarks
        
        # Loop through the detected hands to visualize.
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]

            # Save the landmarks into a NormalizedLandmarkList
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])

            # Draw the landmarks on the hand
            DrawingUtil.draw_landmarks(image,
                                       hand_landmarks_proto,
                                       solutions.hands.HAND_CONNECTIONS,
                                       solutions.drawing_styles.get_default_hand_landmarks_style(),
                                       solutions.drawing_styles.get_default_hand_connections_style())

    def run(self):
        """
        Main game loop. Runs until the 
        user presses "q".
        """    
        
        while self.video.isOpened():
            # Get the current frame
            frame = self.video.read()[1]

            # Convert it to an RGB image
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # The image comes mirrored - flip it
            image = cv2.flip(image, 1)

            # Draw letter onto screen
            cv2.putText(image, str(self.letter), (50, 50), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=BLUE, thickness=2)

            # Convert the image to a readable format and find the hands
            to_detect = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            results = self.detector.detect(to_detect)

            # Draw the hand landmarks
            self.draw_landmarks_on_hand(image, results)

            # Change the color of the frame back
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imshow('ASL Recognizer', image)

            # Record the hand locations if the user presses 'p' key
            if cv2.waitKey(15) == ord('p'):
                for landmark in results.hand_landmarks:
                    for i in range (21):
                        print(str(i) + " " + str(landmark[i]))

            # Break the loop if the user presses 'q'
            if cv2.waitKey(15) & 0xFF == ord('q'):
                break
                    
        self.video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":        
    g = ASL()
    g.run()