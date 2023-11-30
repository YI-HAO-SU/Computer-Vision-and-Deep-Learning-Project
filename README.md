# Computer-Vision-and-Deep-Learning-Project
The proposed system encompasses a comprehensive computer vision application with multifaceted capabilities, including camera calibration, augmented reality implementation, generation of stereo disparity maps, and the utilization of Scale-Invariant Feature Transform (SIFT). Additionally, the system involves deep learning tasks, specifically the training of a CIFAR-10 classifier. For this purpose, the well-established VGG19 architecture, enhanced with Batch Normalization (BN), is employed as the underlying framework for robust model training and classification tasks.

The image data and weights are uploaded to Google Drive.

Image data: https://drive.google.com/file/d/1Re9EJSVUzbtnLcoFVaolEZF3iW4f3OEW/view?usp=drive_link

Weight: https://drive.google.com/file/d/1OXZSUsvjLR_eJMJ2F5aZ8ibZsjUVbtJ-/view?usp=drive_link


The UI implies functions. 

![圖片](https://github.com/YeeHaoSu/Computer-Vision-and-Deep-Learning-Project/assets/90921571/764360a6-869e-4e45-8a17-11d414cbdd4b)


## 1. The first part of the Project is to do the below functions.

![圖片](https://github.com/YeeHaoSu/Computer-Vision-and-Deep-Learning-Project/assets/90921571/8c1bb688-3463-4ff2-820d-637dbfcde66b)

### 1.1 Corner detection 

![圖片](https://github.com/YeeHaoSu/Computer-Vision-and-Deep-Learning-Project/assets/90921571/2488c3fb-a49d-4e17-a6d2-977bf1dbf5eb) ![圖片](https://github.com/YeeHaoSu/Computer-Vision-and-Deep-Learning-Project/assets/90921571/50d52cf9-de53-4454-89e9-d36f6936966b)

### 1.2 Find the intrinsic matrix 

Find the intrinsic matrix:

![圖片](https://github.com/YeeHaoSu/Computer-Vision-and-Deep-Learning-Project/assets/90921571/6c7c35b1-6d4c-4040-99b7-4819da4f028a) ![圖片](https://github.com/YeeHaoSu/Computer-Vision-and-Deep-Learning-Project/assets/90921571/82c2be82-1a67-4d79-b35e-1e135efeeec0)

### 1.3 Find the extrinsic matrix 

Find the extrinsic matrix of the chessboard for each of the 15 images, respectively:

![圖片](https://github.com/YeeHaoSu/Computer-Vision-and-Deep-Learning-Project/assets/90921571/69095c9b-b5a9-41f7-8c24-57dc7dd1330d) ![圖片](https://github.com/YeeHaoSu/Computer-Vision-and-Deep-Learning-Project/assets/90921571/3450351d-a09f-4687-9f68-0ffb17839b66)

### 1.4 Find the distortion matrix

Find the distortion matrix:

![圖片](https://github.com/YeeHaoSu/Computer-Vision-and-Deep-Learning-Project/assets/90921571/fb2afd83-d4e9-4c2a-9464-f033c996d0fe) ![圖片](https://github.com/YeeHaoSu/Computer-Vision-and-Deep-Learning-Project/assets/90921571/ad93d4cc-e1fb-4156-af1e-361b91d8952c)

### 1.5 Show the undistorted result

Undistort the chessboard images.

![圖片](https://github.com/YeeHaoSu/Computer-Vision-and-Deep-Learning-Project/assets/90921571/17826eb0-7f87-4361-8b2b-a58158b15dc6) ![圖片](https://github.com/YeeHaoSu/Computer-Vision-and-Deep-Learning-Project/assets/90921571/ff287833-0545-4b2e-be6e-99d1401fb077)

## 2. The Second part of the Project is to do the below functions and implement augmented reality.

![圖片](https://github.com/YeeHaoSu/Computer-Vision-and-Deep-Learning-Project/assets/90921571/51485abe-ff95-4546-b284-85d7dd3263ac)

### 2.1 Show words on the board

Show a Word (e.g. CAMERA) on the chessboard 

![圖片](https://github.com/YeeHaoSu/Computer-Vision-and-Deep-Learning-Project/assets/90921571/dac58a0e-40c1-43cd-9e5f-537187b9f5b4)


### 2.2 Show words vertically

Show a Word (e.g. CAMERA) vertically on the chessboard 

![圖片](https://github.com/YeeHaoSu/Computer-Vision-and-Deep-Learning-Project/assets/90921571/1dd8c346-214b-408e-9366-91dff45a332f)

### 3. The Third part of the Project is to do the below functions and implement the Stereo Disparity Map.

### 3.1 Stereo Disparity Map

Given: a pair of images, imL.png and imR.png (have been rectified). Find the disparity map/image based on Left and Right stereo images.

![圖片](https://github.com/YeeHaoSu/Computer-Vision-and-Deep-Learning-Project/assets/90921571/ce9e3c0b-ae5c-4e2f-9a10-4d0410d03e5c)

![圖片](https://github.com/YeeHaoSu/Computer-Vision-and-Deep-Learning-Project/assets/90921571/0dfb2f38-2987-4b4f-9c39-2d2d01a6eddd)

### 3.2 Checking the Disparity Value

Given: a pair of images, imL.png and imR.png, and a disparity map from 3.1. Click on the left image and draw the corresponding dot on the right image.

![圖片](https://github.com/YeeHaoSu/Computer-Vision-and-Deep-Learning-Project/assets/90921571/1ac26578-8cff-4cd3-9287-b02e6dfb83e7)

### 4. SIFT

4.1 Keypoints

Click the button “4.1 Keypoints” to show Keypoints.

![圖片](https://github.com/YeeHaoSu/Computer-Vision-and-Deep-Learning-Project/assets/90921571/fe8e1359-1db8-449d-8408-654dbddef492)


4.2 Matched Keypoints

Click the button “4.2 Matched Keypoints” to show Matched Keypoints.

![圖片](https://github.com/YeeHaoSu/Computer-Vision-and-Deep-Learning-Project/assets/90921571/d4e44e73-dc03-4f21-8cd7-c7db83292de6)

### 5. Training a CIFAR10 Classifier Using VGG19 with BN

![圖片](https://github.com/YeeHaoSu/Computer-Vision-and-Deep-Learning-Project/assets/90921571/0f11ec4c-7cae-4669-8c24-5d03e7fa09d6)

5.1 Load CIFAR10 and show 9 Augmented Images with Labels. 

Upon clicking the "1. Show Augmentation Images" button, load nine images from the "/Q5_image/Q5_1/" folder. Apply data augmentation to these images and display the resulting nine augmented images along with their corresponding labels in a new window.

![圖片](https://github.com/YeeHaoSu/Computer-Vision-and-Deep-Learning-Project/assets/90921571/7d42a114-0e9f-46ac-aca0-65136254ee2b) ![圖片](https://github.com/YeeHaoSu/Computer-Vision-and-Deep-Learning-Project/assets/90921571/e6e38b80-f14e-4939-b4c0-c89a6a8469e9)

5.2 Load Model and Show Model Structure. 

Click the button “2. Show Model Structure”. Run the function to show the structure in the terminal.

![圖片](https://github.com/YeeHaoSu/Computer-Vision-and-Deep-Learning-Project/assets/90921571/fc5c4003-7fb4-4ff3-b471-ccde99c57fe9) ![圖片](https://github.com/YeeHaoSu/Computer-Vision-and-Deep-Learning-Project/assets/90921571/4c1b505a-b438-4a7f-83b3-5a552750852c)

5.3 Show Training/Validating Accuracy and Loss.

![圖片](https://github.com/YeeHaoSu/Computer-Vision-and-Deep-Learning-Project/assets/90921571/5f3ab4f7-7084-41bb-8f78-441ebab9ae53) ![圖片](https://github.com/YeeHaoSu/Computer-Vision-and-Deep-Learning-Project/assets/90921571/b40de886-ac5e-44a4-b770-28dd36fbabfa)

5.4 Use the Model with the Highest Validation Accuracy to Run Inference and show the Predicted Distribution and Class Label.

Load the model trained at home by clicking on the "Load Image" button, which prompts a new file selection dialog. Display the predicted class label on the GUI. Additionally, present the probability distribution of model predictions through a histogram in a separate window.

![圖片](https://github.com/YeeHaoSu/Computer-Vision-and-Deep-Learning-Project/assets/90921571/0af7e7ef-1669-4647-8c8c-8e54cf69b989)


