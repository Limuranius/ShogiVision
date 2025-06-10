# ShogiVision
Program that can recognize image/video of shogi board and convert it to digital formats (KIF, SFEN)

Works with following inputs:
- Photo
- Video
- Camera
- E-book in PDF format

https://github.com/user-attachments/assets/d198b818-07ff-40cd-bc38-98f070df82a6

### How to use
Download latest release of ShogiVision, install it and use it. 

Or install packages using ```pip install -r requirements.txt``` and then run ```tools/ShogiVision/main.py```. You might also want to make inference run faster by installing ```onnxruntime-gpu``` and [torch with CUDA](https://pytorch.org/get-started/locally/)

### About
ShogiVision uses YOLO to detect corners of board and then custom convolutional neural network to recognize figure and its direction inside every cell of shogi board. It can work with videos, images and pdf books. All you need to do is specify what it works with. 

There are four main components used by ShogiVision:
- Image source. Could be either photo, video or camera
- Corner detector. Thing that tries to find 4 corners of shogi board on image
- Inventory detector (optional). This component tries to find inventories of each player on image.
- Memorizer. This component only works with video and camera. It keeps track of the game and stores history of moves by comparing current frame of video/camera with previous ones.

More about each element [here](./Elements/README.md)

### Training
If you're not satisfied with current models you can train your own. To train models you must have datasets stored in ```dataset``` folder. You can use datasets created by me [here](#resources) or build your own datasets. There are two models you can train. 
- YOLO segmentation model to detect board corners. By default its dataset needs to be stored at ```datasets/board_segmentation``` in YOLO dataset format. Run ```ShogiNeuralNetwork/train_board_yolo_segmentation.py``` to train model
- ONNX two-output (figure and direction) classification model. Dataset needs to be stored at ```datasets/figure_direction_classification```. Run ```train_mixed_classifier.py``` to train model
  
All trained models are saved to ```models``` folder. If you installed ShogiVision via installer then replace models stored in ```ShogiVision/_internal/models``` with new ones.

### Using your own dataset
If you want to train classification model based on your own data:
1. Launch ```tools/create dataset/main.py```
2. Drag images into window and select true cell values for each image
3. New dataset will be created at ```datasets/```.

### TODO
- Fast Video2kif (WIP)
- Train segmentation model that can detect each piece on image instead of splitting image in 81 cells
- Make working inventory detector for camera

### Resources
- [Models](https://drive.google.com/drive/folders/1QTWss5RQerwVI-kkQVF-ml3MvJ0GjDcT?usp=sharing)
- [Datasets](https://drive.google.com/drive/folders/1HrZ2PqalGUQhEsnZh24-DDBiXgweO7rn?usp=sharing)
- [Videos](https://drive.google.com/drive/folders/18i83vt4UiXAwscvO0VYH0MkUnKvQFGvb?usp=sharing)
- [Images](https://drive.google.com/drive/folders/1lKfzcnO9T8nDU2GhFhHPv1JOf7HEbrxP?usp=sharing)
