from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from math import ceil, exp, pi
from demo_ui import Ui_Dialog
from PyQt5.QtGui import QPixmap

import sys, demo_ui
import cv2
import os
import numpy as np
import glob

from PIL import Image
from torchvision import transforms
from torchsummary import summary
from torchvision.models import vgg19_bn
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.setup_control()

        # Image Calibration
        self.image_path = {}
        self.pics = []
        self.draw_pics = []
        self.found_corners = []
        self.found_chessboards = []
        self.object_points = []
        self.undist_pics = []
        self.h = None
        self.w = None        
        self.ui.spinBox.setRange(1,15)
        self.rvecs = None
        self.tvecs = None
        self.mtx = None
        self.dist = None
        self.newcameramtx = None
        self.roi = None
        self.computed = False
        self.csize = 512

        # Augmented Reality
        self.on_board_imgs = []

        # Stereo Disparity Map
        self.L_image_path = {}
        self.L_image = []
        self.R_image_path = {}
        self.R_image = []
        self.L_image_draw = []
        self.R_image_draw = []

        # VGG
        self.VGG_image_path = None
        self.load = False


    def setup_control(self):
        
        # Image Calibration
        self.ui.Load_folder.clicked.connect(self.open_folder)
        self.ui.Findcorner.clicked.connect(self.find_corner)
        self.ui.FindIntrinsic.clicked.connect(self.Intrinsic_Matrix_result)
        self.ui.Findextrinsic.clicked.connect(self.Extrinsic_Matrix)
        self.ui.Finddistortion.clicked.connect(self.Distortion_Matrix)
        self.ui.ShowResult.clicked.connect(self.Show_Result)

        # Augmented Reality
        self.ui.Showwordsonboard.clicked.connect(self.Show_Words_on_Board)
        self.ui.Showwordsvertically.clicked.connect(self.Show_Words_Vertical)

        # Stereo Disparity Map
        self.ui.LoadImage_L.clicked.connect(self.open_file_L)
        self.ui.LoadImage_R.clicked.connect(self.open_file_R)
        self.ui.StereoDisparityMap.clicked.connect(self.Stereo_Disparity_Map)

        # SIFT
        self.ui.LoadImage_1.clicked.connect(self.open_file_L)
        self.ui.LoadImage_2.clicked.connect(self.open_file_R)
        self.ui.Keypoints.clicked.connect(self.SIFT_Keypoints)
        self.ui.MatchedKeypoints.clicked.connect(self.Matched_Keypoints)

        # VGG19
        self.ui.Augmented.clicked.connect(self.Show_Aug_Images)
        self.ui.ModelStructure.clicked.connect(self.Model_Structure)
        self.ui.ACC_LOSS.clicked.connect(self.Show_ACC_LOSS)
        self.ui.LoadImage_VGG.clicked.connect(self.open_file_VGG)
        self.ui.Inference.clicked.connect(self.Inference)
        

    def open_folder(self):
        folder_path  = QFileDialog.getExistingDirectory(self,
                  "Open folder",
                  "./")                 # start path
        for f in glob.glob(os.path.join(folder_path, '*.bmp')):
            no = f.split('\\')[-1].split('.')[0]
            self.image_path[str(no)] = cv2.imdecode(np.fromfile(f,dtype=np.uint8),-1)


    def open_file_L(self):
        filename, filetype = QFileDialog.getOpenFileName(self,
                  "Open file",
                  "./")                 # start path
        self.L_image_path = cv2.imdecode(np.fromfile(filename,dtype=np.uint8),-1)


    def open_file_R(self):
        filename, filetype = QFileDialog.getOpenFileName(self,
                  "Open file",
                  "./")                 # start path
        self.R_image_path = cv2.imdecode(np.fromfile(filename,dtype=np.uint8),-1)


    def open_file_VGG(self):
        filename, filetype = QFileDialog.getOpenFileName(self,
                  "Open file",
                  "./")                 # start path
        self.VGG_image_path = filename
        scaled_pixmap = QPixmap(filename).scaled(128, 128)
        self.ui.INF.setPixmap(scaled_pixmap)
        self.ui.Predict.setText("Predict = ")
        

    def read_images_calibration(self):
        self.pics = []
        self.draw_pics = []
        for i in range(1, len(self.image_path)+1):
            pic = self.image_path[str(i)]
            self.h, self.w, _ = pic.shape
            self.pics.append(pic.copy())
            self.draw_pics.append(pic.copy())


    def read_image_L(self):
        self.L_image = self.L_image_path[1]


    def read_image_R(self):
        self.R_image = self.R_image_path[1]


    def find_corner(self):
        self.read_images_calibration()
        self.found_chessboards = []

        for image in self.draw_pics:
            print(image.shape)
            # Convert the image to grayscale
            gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray_img, (11, 8), None)

            if ret:
                # Refine the corner positions
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                cv2.cornerSubPix(gray_img, corners, winSize=(5, 5), zeroZone=(-1, -1), criteria=criteria)

                # Draw the corners on the image
                image_with_corners = image.copy()
                cv2.drawChessboardCorners(image_with_corners, (11, 8), corners, ret)
                self.found_chessboards.append(image_with_corners)
            else:
                print("Chessboard not found.")

        for pic in self.found_chessboards:
            image = cv2.resize(pic, (self.csize, self.csize))
            cv2.imshow('Chessboard', image)
            cv2.waitKey(500)
        cv2.destroyAllWindows()


    def Intrinsic_Matrix(self):
        self.read_images_calibration()
        objp = np.zeros((11 * 8, 3), np.float32)
        objp[:,:2] = np.mgrid[0 : 11, 0 : 8].T.reshape(-1,2)

        self.found_corners = []
        self.object_points = []
        self.found_chessboards = []

        for image in self.pics:
            grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(grayimg, (11, 8), None)
            if ret:
                self.found_chessboards.append(ret)
                self.found_corners.append(corners)
                self.object_points.append(objp)
            else:
                print("FAIL")

        # Calculate the matrix
        ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(self.object_points, self.found_corners, (self.h, self.w), None, None)


    def Undisort(self):
        self.newcameramtx, self.roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (self.w, self.h), 1, (self.w, self.h))
        for pic in self.pics:
            img = cv2.undistort(pic, self.mtx, self.dist, None, self.newcameramtx)
            x, y, w, h = self.roi
            img = img[y:y+h, x:x+w]
            pic = pic[y:y+h, x:x+w]
            self.undist_pics.append(cv2.hconcat([pic, img]))


    def Intrinsic_Matrix_result(self):
        if not self.computed:
            self.Intrinsic_Matrix()
            self.computed = True
        print('Intrinsic:')
        print(self.mtx)


    def Extrinsic_Matrix(self):
        if not self.computed:
            self.Intrinsic_Matrix()
            self.computed = True
        num = (self.ui.spinBox.value())
        res , _ = cv2.Rodrigues(self.rvecs[num])
        res = np.concatenate((res, self.tvecs[num]), axis=1)
        print('Extrinsic:')
        print(res)

    
    def Distortion_Matrix(self):
        if not self.computed:
            self.Intrinsic_Matrix()
            self.computed = True
        print('Distortion')
        print(self.dist)

    
    def Show_Result(self):
        if not self.computed:
            self.Intrinsic_Matrix()
            self.computed = True
        
        self.Undisort()
        for pic in self.undist_pics:
            image = cv2.resize(pic, (2 * self.csize, self.csize))
            cv2.imshow('Chessboard', image)
            cv2.waitKey(500)
        cv2.destroyAllWindows()


    def Show_Words_on_Board(self):
        self.Intrinsic_Matrix()
        self.on_board_imgs = []
        txt_file = cv2.FileStorage('Dataset/Q2_Image/Q2_lib/alphabet_lib_onboard.txt', cv2.FILE_STORAGE_READ)
        content = self.ui.textEdit_2.toPlainText()
        for num in range(len(self.pics)):
            self.on_board_imgs.append(self.pics[num])
            for i in range(len(content)):
                ch = txt_file.getNode(content[i]).mat()
                for line in range(ch.shape[0]):
                    for pi in range(ch[line].shape[0]):
                        ch[line][pi][0] = ch[line][pi][0] + (7 - 3 * (i % 3))
                        ch[line][pi][1] = ch[line][pi][1] + (5 - 3 * (i // 3))
                    lpts, _ = cv2.projectPoints(ch[line].astype(np.float32), self.rvecs[num], self.tvecs[num], self.mtx, self.dist)
                    cv2.line(self.on_board_imgs[num], tuple(lpts[0].astype(int).ravel()), tuple(lpts[1].astype(int).ravel()), (0, 0, 255), 5)

        for pic in self.on_board_imgs:
            image = cv2.resize(pic, (self.csize, self.csize))
            cv2.imshow('On Board', image)
            cv2.waitKey(1000)
        cv2.destroyAllWindows()


    def Show_Words_Vertical(self):
        self.Intrinsic_Matrix()
        self.on_board_imgs = []
        txt_file = cv2.FileStorage('Dataset/Q2_Image/Q2_lib/alphabet_lib_vertical.txt', cv2.FILE_STORAGE_READ)
        content = self.ui.textEdit_2.toPlainText()
        for num in range(len(self.pics)):
            self.on_board_imgs.append(self.pics[num])
            for i in range(len(content)):
                ch = txt_file.getNode(content[i]).mat()
                for line in range(ch.shape[0]):
                    for pi in range(ch[line].shape[0]):
                        ch[line][pi][0] = ch[line][pi][0] + (7 - 3 * (i % 3))
                        ch[line][pi][1] = ch[line][pi][1] + (5 - 3 * (i // 3))
                    lpts, _ = cv2.projectPoints(ch[line].astype(np.float32), self.rvecs[num], self.tvecs[num], self.mtx, self.dist)
                    cv2.line(self.on_board_imgs[num], tuple(lpts[0].astype(int).ravel()), tuple(lpts[1].astype(int).ravel()), (0, 0, 255), 5)

        for pic in self.on_board_imgs:
            image = cv2.resize(pic, (self.csize, self.csize))
            cv2.imshow('On Board', image)
            cv2.waitKey(1000)
        cv2.destroyAllWindows()


    def Stereo_Disparity_Map(self):
        image_L_gray = cv2.cvtColor(self.L_image_path, cv2.COLOR_BGR2GRAY)
        image_R_gray = cv2.cvtColor(self.R_image_path, cv2.COLOR_BGR2GRAY)
        stereo = cv2.StereoBM_create(numDisparities=256, blockSize=5)
        disparity = stereo.compute(image_L_gray, image_R_gray)
        disparity_norm = (disparity - abs(disparity.min())) / (disparity.max() - disparity.min())
        
        L = self.L_image_path
        R = self.R_image_path
        L = cv2.resize(L, (960, 540))
        R = cv2.resize(R, (960, 540))
        disparity_norm_i = cv2.resize(disparity_norm, (960, 540))

        cv2.namedWindow('imageL')
        cv2.moveWindow('imageL', 0, 0)
        cv2.imshow('imageL', L)
        cv2.namedWindow('imageR')
        cv2.moveWindow('imageR', 512, 0)
        cv2.imshow('imageR', R)
        cv2.namedWindow('Disparity')
        cv2.moveWindow('Disparity', 1024, 0)
        cv2.imshow('Disparity', disparity_norm_i)

        def draw_dot(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                img = R.copy()
                img = cv2.circle(img , (int(x - disparity[y][x] / 4019.284 * 279.184) , y), 0, (0, 255, 0), 30)
                disparity_value = disparity[y*2, x*2]
                if(disparity_value < 0): disparity_value = "Fail"
                print(f"({x*2}, {y*2}), dis:{disparity_value}")
                cv2.imshow('imageR', img)

        cv2.setMouseCallback('imageL', draw_dot)
        cv2.waitKey()
        cv2.destroyAllWindows()

    
    def Keypoint(self, image):
        # Load the image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Use SIFT to detect keypoints
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray, None)

        # Draw keypoints
        output_image = np.copy(image)
        output_image = cv2.drawKeypoints(gray, keypoints, output_image, flags=None, color=(0, 255, 0))

        return keypoints, descriptors, output_image, gray


    def SIFT_Keypoints(self):
        _, _, output_image, _ = self.Keypoint(self.L_image_path)
        output_image = cv2.resize(output_image, (1024, 1024))

        cv2.namedWindow('Keypoint')
        cv2.moveWindow('Keypoint', 0, 0)
        cv2.imshow('Keypoint', output_image)
        cv2.waitKey()
        cv2.destroyAllWindows()

    
    def Matched_Keypoints(self):
        keypoints_L, descriptors_L, _, gray_L = self.Keypoint(self.L_image_path)
        keypoints_R, descriptors_R, _, gray_R = self.Keypoint(self.R_image_path)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors_L, descriptors_R, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        matchesMask = [[0, 0] for _ in range(len(matches))]
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.75 * n.distance:
                matchesMask[i] = [1, 0]

        output_image = np.zeros((max(gray_L.shape[0], gray_R.shape[0]), gray_L.shape[1] + gray_R.shape[1], 3), dtype=np.uint8)
        output_image = cv2.UMat(output_image)

        cv2.drawMatchesKnn( gray_L, 
                            keypoints_L, 
                            gray_R, 
                            keypoints_R, 
                            matches1to2=matches, 
                            matchesMask=matchesMask,
                            outImg = output_image,
                            flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        output_image = output_image.get()
        output_image = cv2.resize(output_image, (1024, 512))

        cv2.namedWindow('Matching Result')
        cv2.moveWindow('Matching Result', 0, 0)
        cv2.imshow("Matching Result", output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows() 

    
    def Show_Aug_Images(self):
        folder_path = "Dataset/Q5_image/Q5_1/"
        transforms_train = transforms.Compose({
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
        })
        path = []
        fig, axs = plt.subplots(3, 3, figsize=(8, 8))

        for f in glob.glob(os.path.join(folder_path, '*.png')):
            path.append(f)

        for i, image_filename in enumerate(path):
            row = i // 3
            col = i % 3
            img = Image.open(image_filename)
            img = transforms_train(img)
            axs[row, col].imshow(img)
            axs[row, col].set_title(f"{os.path.split(image_filename)[1].split('.')[0]}")
            axs[row, col].axis('off')  

        plt.subplots_adjust(wspace=0.2, hspace=0.2)
        plt.gcf().canvas.set_window_title("Show Augmented Images")
        plt.show()

    
    def Model_Structure(self):
        model = vgg19_bn(num_classes=10)
        summary(model, (3, 32, 32), device='cpu')

    
    def Show_ACC_LOSS(self):
        Acc = Image.open('Acc_pre.png')
        Loss = Image.open('Loss_pre.png')
        fig, axs = plt.subplots(2, 1, figsize=(6, 8))
        axs[0].imshow(Acc)
        axs[0].set_title("Accuracy")
        axs[0].axis('off')  
        axs[1].imshow(Loss)
        axs[1].set_title("Loss")
        axs[1].axis('off')  
        plt.gcf().canvas.set_window_title("Show Accuracy and Loss")
        plt.show()
    

    def Inference(self):
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        model = vgg19_bn(num_classes=10)
        model.classifier[6] = nn.Linear(4096, 10)
        checkpoint = torch.load("Best_pre.pth", map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)
        image_path = self.VGG_image_path

        model.eval()
        with torch.set_grad_enabled(False):
            img = Image.open(image_path)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])(img)
            image = img.unsqueeze(0)
            output = model(image)
            probas = F.softmax(output, dim=1).detach().numpy()[0]
            _, predicted = torch.max(output, 1)
        Answer = "Predicted = " + str(classes[predicted[0]])
        self.ui.Predict.setText(Answer)

        plt.figure(figsize=(6, 6))
        plt.bar(classes, probas)
        plt.xticks(rotation=45)  
        plt.yticks(np.arange(0, 1.1, 0.1))  
        plt.xlabel('Class', labelpad=10)
        plt.ylabel('Accuracy')
        plt.title('Probability of each class')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())