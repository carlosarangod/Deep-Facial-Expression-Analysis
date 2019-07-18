# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 20:00:20 2019

@author: carlo
"""

import dlib
import numpy as np
import cv2
import Face_ROI_Dlib_generator as frg
from keras.models import load_model

def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
 
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

# Initialize the face frontal detector from Dlib
face_detector = dlib.get_frontal_face_detector()
#Initialize the facial landmarks predictor from Dlib
face_predictor = dlib.shape_predictor("Face_landmarks/shape_predictor_68_face_landmarks.dat")

#Activate the webcam using OpenCV
camera = cv2.VideoCapture(0)

# Flag to start the face detection part
detect_flag = True
# Flag to start the face landmark preciction part
predict_flag = False
# class that contains the result of the face_detector
predicted_rectangle =  dlib.rectangle
# Flag to announce if this is the first face detection ever
first_det_flag = False

# Counter until we go back to face detection
detection_count = 0
detection_count_reset = 15
eye_left = np.zeros([6,2,15])
eye_right = np.zeros([6,2,15])
eyeb_left = np.zeros([5,2,15])
eyeb_right = np.zeros([5,2,15])
nose = np.zeros([5,2,15])
mouth = np.zeros([12,2,15])

while(True):
    # Capture frame-by-frame
    ret, frame = camera.read()
    frame = cv2.flip( frame, 1 )
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Face Detection cycle
    if detect_flag:    
        detections = face_detector(frame, 1)
        if not detections:
            continue
        ## Choose first detected face
        shape = detections[0]
        (init_det_x, init_det_y, face_width, face_height) = rect_to_bb(shape)
#        Paint detected face rectangle
        cv2.rectangle(frame, (init_det_x, init_det_y), (init_det_x + face_width, init_det_y + face_height), (0, 255, 255), 2)
        detect_flag = False
        predict_flag = True
#        Update detected face location
        predicted_rectangle =  dlib.rectangle(init_det_x,init_det_y,shape.right(),shape.bottom())
        reset_shape = True
        old_face_det_x = 0
        old_face_det_y = 0
    # Face Detection cycle
    if predict_flag:
#        Face landmark prediction
        facelan = face_predictor(gray, predicted_rectangle)   
        land_coor = frg.shape_to_np(facelan)
        # Visualize the landmarks
        for (x, y) in land_coor:
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
        
#        Update face location based on the predicted face landmarks
        (new_face_det_x,new_face_det_y) = np.mean(land_coor, axis = 0)
        if reset_shape:
            reset_shape = False
            face_det_deriv_x = 0
            face_det_deriv_y = 0
        else:
            face_det_deriv_x = new_face_det_x - old_face_det_x
            face_det_deriv_y = new_face_det_y - old_face_det_y
        face_box_x = np.rint(init_det_x + face_det_deriv_x).astype(int)
        face_box_y = np.rint(init_det_y + face_det_deriv_y).astype(int)
        face_box_w = np.rint(init_det_x + face_width + face_det_deriv_x).astype(int)
        face_box_h = np.rint(init_det_y + face_height + face_det_deriv_y).astype(int)
        predicted_shape =  dlib.rectangle(face_box_x, face_box_y, face_box_w, face_box_h)
        (x, y, w, h) = rect_to_bb(predicted_rectangle)
        cv2.rectangle(frame, (face_box_x, face_box_y), (face_box_w, face_box_h), (0, 255, 0), 2)
        old_face_det_x = new_face_det_x
        old_face_det_y = new_face_det_y
        
        eye_left[:,:,detection_count] = land_coor[36:42,:]
        eye_right[:,:,detection_count] = land_coor[42:48,:]
        eyeb_left[:,:,detection_count] = land_coor[17:22,:]
        eyeb_right[:,:,detection_count] = land_coor[22:27,:]
        nose[:,:,detection_count] = land_coor[31:36,:]
        mouth[:,:,detection_count] = land_coor[48:60,:]
    
    eye_left_mean = np.mean(eye_left[:,:,detection_count], axis = 0).T
    eye_right_mean = np.mean(eye_right[:,:,detection_count], axis = 0).T
    eyeb_left_mean = np.mean(eyeb_left[:,:,detection_count], axis = 0).T
    eyeb_right_mean = np.mean(eyeb_right[:,:,detection_count], axis = 0).T
    nose_mean = np.mean(nose[:,:,detection_count], axis = 0).T
    mouth_mean = np.mean(mouth[:,:,detection_count], axis = 0).T    
    
    detection_count = detection_count + 1
    # Create face bounding boxes
    if detection_count == (detection_count_reset) and not(first_det_flag) :
        first_det_flag = True
        
        temp_eyeleft = np.reshape(np.median(eye_left[:,:,0:15],axis=2),(eye_left.shape[0],eye_left.shape[1],1))
        temp_eyeright = np.reshape(np.median(eye_right[:,:,0:15],axis=2),(eye_left.shape[0],eye_left.shape[1],1))
        temp_mouth = np.reshape(np.vstack((np.median(mouth[:,:,0:15],axis=2),np.median(nose[-1,:,0:15],axis=1))),(mouth.shape[0]+1,mouth.shape[1],1))
                
        eye_feat = np.squeeze(np.vstack((temp_eyeleft,temp_eyeright)))     
        lbrow_ROI_Corners = frg.Feature_box(eye_feat,eyeb_left,'eyebrow',[0,0,2,2])
        rbrow_ROI_Corners = frg.Feature_box(eye_feat,eyeb_right,'eyebrow',[0,0,2,2])
        leye_ROI_Corners = frg.Feature_box(eye_feat,temp_eyeleft,'eye',[1,1,3,3])
        reye_ROI_Corners = frg.Feature_box(eye_feat,temp_eyeright,'eye',[1,1,3,3])
        nose_ROI_Corners = frg.Feature_box(eye_feat,nose,'nose',[0.5,0.5,5,20])
        mouth_ROI_Corners = frg.Feature_box(eye_feat,temp_mouth,'mouth',[0,0,1,1])
        
        
        face_pts2 = np.float32([eye_left_mean,eye_right_mean, eyeb_left_mean,
                           eyeb_right_mean, nose_mean,mouth_mean])
    #    Every 15 frames we restart the face detection cycle
    if detection_count == detection_count_reset:
        detection_count = 0
        detect_flag = True
        predict_flag = False
        
#    Update face bounding boxes
    if first_det_flag:
        face_pts1 = face_pts2
        face_pts2 = np.float32([eye_left_mean,eye_right_mean, eyeb_left_mean,
                           eyeb_right_mean, nose_mean,mouth_mean])
        face_pts3 = face_pts2 - face_pts1
        
        for i in range(0,2):
            leye_ROI_Corners[:,i] = leye_ROI_Corners[:,i] + face_pts3[0,i]
            reye_ROI_Corners[:,i] = reye_ROI_Corners[:,i] + face_pts3[1,i]
            lbrow_ROI_Corners[:,i] = lbrow_ROI_Corners[:,i] + face_pts3[2,i]
            rbrow_ROI_Corners[:,i] = rbrow_ROI_Corners[:,i] + face_pts3[3,i]
            nose_ROI_Corners[:,i] = nose_ROI_Corners[:,i] + face_pts3[4,i]
            mouth_ROI_Corners[:,i] = mouth_ROI_Corners[:,i] + face_pts3[5,i]
        
#        Visualize bounding boxes
        len_corner = leye_ROI_Corners.shape[0]
        for i in range(0,len_corner-1):
            p1 = tuple(lbrow_ROI_Corners[i,:].astype(int))
            p2 = tuple(lbrow_ROI_Corners[i+1,:].astype(int))
            cv2.line(frame, p1,p2, (255, 255, 0))   
            p1 = tuple(rbrow_ROI_Corners[i,:].astype(int))
            p2 = tuple(rbrow_ROI_Corners[i+1,:].astype(int))
            cv2.line(frame, p1,p2, (255, 255, 0))   
            p1 = tuple(leye_ROI_Corners[i,:].astype(int))
            p2 = tuple(leye_ROI_Corners[i+1,:].astype(int))
            cv2.line(frame, p1,p2, (255, 255, 0))   
            p1 = tuple(reye_ROI_Corners[i,:].astype(int))
            p2 = tuple(reye_ROI_Corners[i+1,:].astype(int))
            cv2.line(frame, p1,p2, (255, 255, 0))   
            p1 = tuple(mouth_ROI_Corners[i,:].astype(int))
            p2 = tuple(mouth_ROI_Corners[i+1,:].astype(int))
            cv2.line(frame, p1,p2, (255, 255, 0))
            p1 = tuple(nose_ROI_Corners[i,:].astype(int))
            p2 = tuple(nose_ROI_Corners[i+1,:].astype(int))
            cv2.line(frame, p1,p2, (255, 0, 255))
   
    # Display the resulting frame
    cv2.imshow('frame',frame)
    # Stops the video if the user press the 'q' button on the keyboard.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
camera.release()
cv2.destroyAllWindows()