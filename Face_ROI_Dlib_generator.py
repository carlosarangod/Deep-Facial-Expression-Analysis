# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 14:40:11 2019

@author: carlo
"""

import numpy as np

def Facial_feature_axis(feat_landmarks):
    num = feat_landmarks.shape[0]
    C = feat_landmarks[:,0][np.newaxis]
    Y = feat_landmarks[:,1][np.newaxis]
    A = np.concatenate((C.T,np.ones((num,1),dtype = float)),axis=1)
    m, c = np.linalg.lstsq(A, Y.T)[0]

    return m, c

def Facial_feature_box(feat_points, axis_eq, box_scale):
    ## Order of the coordinates left-right-up-down
    m, c = axis_eq
    feat_ang = np.arctan(m)
    center = np.mean(feat_points,axis=0)
    Dist_ver = np.sqrt(np.sum((feat_points[2,0:2]-center[0:2])**2))
    Dist_hor = np.sqrt(np.sum((feat_points[0,0:2]-center[0:2])**2))
    box_in_sc = 0.5
    x_left = feat_points[0,0] - box_scale[0]*Dist_hor*box_in_sc*np.cos(feat_ang); 
    y_left = feat_points[0,1] - box_scale[0]*Dist_hor*box_in_sc*np.sin(feat_ang); 
    b_left = y_left+x_left*(1/m);
    x_right= feat_points[1,0] + box_scale[1]*Dist_hor*box_in_sc*np.cos(feat_ang); 
    y_right = feat_points[1,1] + box_scale[1]*Dist_hor*box_in_sc*np.sin(feat_ang); 
    b_right = y_right+x_right*(1/m);
        
    x_up = feat_points[2,0] - box_scale[2]*Dist_ver*box_in_sc*np.cos(np.pi/2-feat_ang); 
    y_up = feat_points[2,1] - box_scale[2]*Dist_ver*box_in_sc*np.sin(np.pi/2-feat_ang); 
    b_up = y_up+x_up*(-m);
    x_down= feat_points[3,0] + box_scale[3]*Dist_ver*box_in_sc*np.cos(np.pi/2-feat_ang); 
    y_down = feat_points[3,1] + box_scale[3]*Dist_ver*box_in_sc*np.sin(np.pi/2-feat_ang); 
    b_down = y_down+x_down*(-m);
    
    Corners = np.zeros((5,2))

    Corners[0,0] = (b_left-b_up)/(m+1/m);
    Corners[0,1] =  m*Corners[0,0]+b_up;
    Corners[1,0] = (b_right-b_up)/(m+1/m);
    Corners[1,1] = m*Corners[1,0]+b_up;
    Corners[2,0] = (b_right-b_down)/(m+1/m);
    Corners[2,1] = m*Corners[2,0]+b_down;
    Corners[3,0] = (b_left-b_down)/(m+1/m);
    Corners[3,1] = m*Corners[3,0]+b_down;
    Corners[4,:] = Corners[0,:];
    return Corners
    
def Face_box(eyes_feat,eyeb_left,eyeb_right,Contour):  
    eye_m, eye_c = Facial_feature_axis(eyes_feat) 
    border_up = np.mean(np.vstack((eyeb_left[[1,2,3],:,0],eyeb_right[[1,2,3],:,0])), axis = 0)
    border_down = np.mean(np.median(Contour[[7,8,9],:,0:20],axis=0),axis=1)
    border_left = np.mean(Contour[[0,1,2],:,0], axis = 0)
    border_right = np.mean(Contour[[14,15,16],:,0], axis = 0)
    face_border = np.vstack((border_left,border_right,border_up,border_down))
    Face_ROI_Corners = Facial_feature_box(face_border, [eye_m,eye_c], [0,0,1,0])
    return Face_ROI_Corners

def Feature_box(Eq_feat,face_feat,feat_type,box_scale):  
#    eye_m, eye_c = Facial_feature_axis(eye_feat, eye_right[0,0,:], eye_left[0,3,:]) 
    Eq_m, Eq_c = Facial_feature_axis(Eq_feat) 
    if feat_type == 'eye':
        border_up = np.mean(face_feat[1:3,:,0], axis = 0)
        border_down = np.mean(face_feat[4:6,:,0], axis = 0)
        border_left = face_feat[0,:,0]
        border_right = face_feat[3,:,0]
    elif feat_type == 'eyebrow':
        border_up = np.mean(face_feat[1:3,:,0], axis = 0)
        border_down = np.mean(face_feat[[0,4],:,0], axis = 0)
        border_left = face_feat[0,:,0]
        border_right = face_feat[4,:,0]
    elif feat_type == 'nose':
        border_up = np.mean(face_feat[[0,4],:,0], axis = 0)
        border_down = face_feat[2,:,0]
        border_left = face_feat[0,:,0]
        border_right = face_feat[4,:,0]
    elif feat_type == 'mouth':
        border_up = np.mean(face_feat[2:5,:,0], axis = 0)
        border_down = np.mean(face_feat[8:11,:,0], axis = 0)
        border_left = face_feat[0,:,0]
        border_right = face_feat[6,:,0]
    elif feat_type == 'cormouth':
        border_up = face_feat[12,:,0]
        border_down = np.mean(face_feat[8:11,:,0], axis = 0)
        border_left = face_feat[0,:,0]
        border_right = face_feat[6,:,0]
    
    feat_borders = np.vstack((border_left,border_right,border_up,border_down))
    Feat_ROI_Corners = Facial_feature_box(feat_borders, [Eq_m,Eq_c], box_scale)
    return Feat_ROI_Corners

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

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
 
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
 
	# return the list of (x, y)-coordinates
	return coords