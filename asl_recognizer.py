"""
A game program uses hand tracking to 
determine ASL letters.

@author: Sierra Shaw

"""

import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2
import pandas as pd

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

        self.data = {}

        self.data.clear()

        self.currenttrack_letter = 'a'
    
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

    def copy_info(self, index, landmark_info):
        landmark_names = ["WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP", "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP",
                            "INDEX_FINGER_TIP","MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP", "RING_FINGER_MCP",
                            "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP", "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"]
        if landmark_names[index] + "_X" not in self.data and landmark_names[index] + "_Y" not in self.data:
            self.data[landmark_names[index] + "_X"] = []
            self.data[landmark_names[index] + "_Y"] = []

        if landmark_info == None:
            self.data[landmark_names[index] + "_X"].append(None)
            self.data[landmark_names[index] + "_Y"].append(None)
        else:
            self.data[landmark_names[index] + "_X"].append(landmark_info.x)
            self.data[landmark_names[index] + "_Y"].append(landmark_info.y)

        # Fixing the varying len of array issue
        largest_len = 0
        for key in self.data:
            if key != "Letter":
                if len(self.data[key]) > largest_len:
                    largest_len = len(self.data[key])
            
        for key in self.data:
            if key != 'Letter':
                if len(self.data[key]) < largest_len:
                    for i in range (largest_len - len(self.data[key])):
                        self.data[key].append(None)
        
        self.data["Letter"] = self.currenttrack_letter

    
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

            # Draw the rectangle size on the screen
            cv2.rectangle(image, (400,150), (900,650), BLUE, 4)

            # Convert the image to a readable format and find the hands
            to_detect = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            results = self.detector.detect(to_detect)

            # Draw the hand landmarks
            self.draw_landmarks_on_hand(image, results)

            # Change the color of the frame back
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imshow('ASL Recognizer', image)

            key_pressed = cv2.waitKey(15) & 0xFF

            # Record the hand locations if the user presses 'p' key
            if key_pressed == ord('p'):
                # landmark_names = ["WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP", "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP",
                #             "INDEX_FINGER_TIP","MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP", "RING_FINGER_MCP",
                #             "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP", "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"]
                # for i in range (21):
                #     if landmark_names[i] in results.hand_landmarks:
                #         self.copy_info(i, results.hand_landmarks[smth][i])
                #     else:
                #         self.copy_info(i, None)

                for landmark in results.hand_landmarks:
                    for i in range (21):
                        self.copy_info(i, landmark[i])
                # for landmark in results.hand_landmarks:
                #     print(landmark)
                
                #print(self.data)
            
            # Changes to the next letter for data capturing
            if key_pressed == ord('n'):
                self.currenttrack_letter
                code = ord(self.currenttrack_letter)
                self.currenttrack_letter = chr(code + 1)

            # Break the loop if the user presses 'q'
            if key_pressed == ord('q'):
                # Turn the dict data into a dataframe
                data_frame = pd.DataFrame.from_dict(self.data)
                data_frame.to_csv('data/landmark_locations.csv', index=False, mode='a')
                #data_frame.to_csv('data/landmark_locations', index=False)
                break

        self.video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":        
    g = ASL()
    g.run()