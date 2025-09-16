#from vision import face, photo, QRcode
#from speech import Speech
#from control import dog_control
from vision.camera_manager  import  CameraManager
from vision.qr_code_detector  import  QRCodeDetector
from vision.face_detector  import  FaceDetector
from vision.photo_detector  import  PhotoDetector
from vision.hand_pose import HandPoseEstimator
import threading

def main():
    # 创建摄像头管理器
    cam_manager = CameraManager()
    cam_manager.start()
    
    # 创建QR检测器
    qr_detector = QRCodeDetector(cam_manager)
    face_detector=FaceDetector(cam_manager)
    photo_detector=FaceDetector(cam_manager)
    # 创建手部姿态估计器
    hand_estimator = HandPoseEstimator(
        cam_manager,
#        model_path="./assets/handpose_x/weights/ReXNetV1-size-256-wingloss102-0.122.pth", #模型更大
#        model='ReXNetV1',
        model_path="./assets/handpose_x/weights/squeezenet1_1-size-256-loss-0.0732.pth",   #轻量模型
        model='squeezenet1_1',
        num_classes=42,
        GPUS='0',
        img_size=(256, 256)
    )


    # 为每个检测器创建单独的线程
    qr_thread = threading.Thread(target=qr_detector.run_detection, name="QR_Detector")
    face_thread = threading.Thread(target=face_detector.run_detection, name="Face_Detector")
    photo_thread = threading.Thread(target=photo_detector.run, name="Photo_Detector")
    hand_thread = threading.Thread(target=hand_estimator.run_estimation, 
                                  kwargs={'camera_id': 0, 'vis': True}, 
                                  name="Hand_Pose_Estimator")
    
    # 启动所有线程
    qr_thread.start()
    face_thread.start()
    photo_thread.start()
    hand_thread.start()
   
if __name__ == '__main__':
    main()