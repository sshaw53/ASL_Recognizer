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
from sklearn.neighbors import KNeighborsClassifier

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

        # Train + Create the model
        df = pd.read_csv("data/landmark_locations.csv")
        features = ['WRIST_X', 'WRIST_Y', 'THUMB_CMC_X', 'THUMB_CMC_Y', 'THUMB_MCP_X', 'THUMB_MCP_Y', 'THUMB_IP_X', 'THUMB_IP_Y', 'THUMB_TIP_X',
            'THUMB_TIP_Y', 'INDEX_FINGER_MCP_X', 'INDEX_FINGER_MCP_Y', 'INDEX_FINGER_PIP_X', 'INDEX_FINGER_PIP_Y', 'INDEX_FINGER_DIP_X', 'INDEX_FINGER_DIP_Y', 
            'INDEX_FINGER_TIP_X', 'INDEX_FINGER_TIP_Y', 'MIDDLE_FINGER_MCP_X', 'MIDDLE_FINGER_MCP_Y', 'MIDDLE_FINGER_PIP_X', 'MIDDLE_FINGER_PIP_Y', 
            'MIDDLE_FINGER_DIP_X', 'MIDDLE_FINGER_DIP_Y', 'MIDDLE_FINGER_TIP_X', 'MIDDLE_FINGER_TIP_Y', 'RING_FINGER_MCP_X', 'RING_FINGER_MCP_Y', 
            'RING_FINGER_PIP_X', 'RING_FINGER_PIP_Y', 'RING_FINGER_DIP_X', 'RING_FINGER_DIP_Y', 'RING_FINGER_TIP_X', 'RING_FINGER_TIP_Y', 'PINKY_MCP_X', 
            'PINKY_MCP_Y', 'PINKY_PIP_X', 'PINKY_PIP_Y', 'PINKY_DIP_X', 'PINKY_DIP_Y', 'PINKY_TIP_X', 'PINKY_TIP_Y']
        X = df[features]
        y = df["Letter"]

        self.model = KNeighborsClassifier(n_neighbors=13)
        self.model = self.model.fit(X, y)
    
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
                            "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP", "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"]
        
        if landmark_names[index] + "_X" not in self.data and landmark_names[index] + "_Y" not in self.data:
            self.data[landmark_names[index] + "_X"] = []
            self.data[landmark_names[index] + "_Y"] = []

        self.data[landmark_names[index] + "_X"].append(landmark_info.x)
        self.data[landmark_names[index] + "_Y"].append(landmark_info.y)
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
            
            # Getting the coordinates of the hand landmarks at a given moment
            to_test = []
            if len(results.hand_landmarks[0]) == 21:
                for i in range (21):
                    to_test.append(results.hand_landmarks[0][i].x)
                    to_test.append(results.hand_landmarks[0][i].y)

            # Put in the hand coordinates as the features and save the label into self.letter
            self.letter = self.model.predict([to_test])

            key_pressed = cv2.waitKey(15) & 0xFF

            # FOR DATA COLLECTING
            # Record the hand locations if the user presses 'p' key
            if key_pressed == ord('p'):
                for i in range (21):
                    self.copy_info(i, results.hand_landmarks[0][i])
                                
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
                break

        self.video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":        
    g = ASL()
    g.run()