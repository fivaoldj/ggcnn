import sys
import rospy
import numpy as np
import torch
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class GGCNNGraspDetector:
    """Класс для детекции точек захвата с использованием GGCNN на глубинных изображениях."""
    
    def __init__(self):
        """Инициализация детектора, загрузка модели и настройка подписчика ROS."""
        # Настройка устройства (GPU/CPU)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Игнорирование предупреждений PyTorch
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        
        # Загрузка модели GGCNN
        model_path = '../ggcnn/ggcnn_weights_cornell/ggcnn_epoch_23_cornell'
        try:
            self.model = torch.load(model_path, map_location=self.device)
            self.model.eval()
            torch.nn.Module.dump_patches = True
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

        self.bridge = CvBridge()  # Конвертер ROS Image ↔ OpenCV
        self.depth_image = None   # Текущее глубинное изображение
        self.grasp_params = None  # Параметры последнего обнаруженного захвата
        
        # Подписка на топик с глубинным изображением
        rospy.Subscriber('/camera_3_depth/depth/image_raw', Image, self.depth_image_callback)
        rospy.loginfo("GGCNNGraspDetector initialized successfully")

    def depth_image_callback(self, data):
        """Обработка входящего глубинного изображения из ROS.
        
        Args:
            data (sensor_msgs.msg.Image): Входящее сообщение с глубинным изображением.
        """
        try:
            depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
            
            if depth_image is None or depth_image.size == 0:
                rospy.logwarn("Received empty depth image!")
                return
            
            depth_image = self.preprocess_depth_image(depth_image)
            self.grasp_params = self.process_with_ggcnn(depth_image)

        except CvBridgeError as e:
            rospy.logerr(f"Image conversion error: {e}")
        except Exception as e:
            rospy.logerr(f"Depth image processing error: {e}")

    def preprocess_depth_image(self, depth_image):
        """Нормализация глубинного изображения.
        
        Args:
            depth_image (numpy.ndarray): Входное глубинное изображение.
            
        Returns:
            numpy.ndarray: Нормализованное изображение.
        """
        depth_image = np.nan_to_num(depth_image, nan=0.0, posinf=0.0, neginf=0.0)
        min_depth = np.min(depth_image)
        max_depth = np.max(depth_image)
        return (depth_image - min_depth) / (max_depth - min_depth)

    def process_with_ggcnn(self, depth_image, k=40):
        """Обработка изображения с помощью GGCNN и поиск оптимальных точек захвата.
        
        Args:
            depth_image (numpy.ndarray): Нормализованное глубинное изображение.
            k (int): Количество топ-точек для усреднения.
            
        Returns:
            dict: Параметры захвата {x, y, z, theta, width} или None при ошибке.
        """
        try:
            depth_tensor = torch.tensor(depth_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                q_img, angle_img, width_img, pos_img = self.model(depth_tensor)
                
            q_img = q_img.cpu().squeeze().numpy()
            angle_img = angle_img.cpu().squeeze().numpy()
            width_img = width_img.cpu().squeeze().numpy()
            pos_img = pos_img.cpu().squeeze().numpy()

            # Поиск и усреднение топ-k точек
            top_k_indices = np.argpartition(q_img.flatten(), -k)[-k:]
            top_k_indices_2d = np.unravel_index(top_k_indices, q_img.shape)
            
            grasp_y = np.mean(top_k_indices_2d[0]).astype(int)
            grasp_x = np.mean(top_k_indices_2d[1]).astype(int)
            grasp_theta = np.mean(angle_img[top_k_indices_2d])
            grasp_width = width_img[grasp_y, grasp_x]
            grasp_z = pos_img[grasp_y, grasp_x]

            # Визуализация результатов
            self._visualize_results(depth_image, q_img, angle_img, grasp_x, grasp_y)

            return {
                "x": grasp_x,
                "y": grasp_y,
                "z": grasp_z,
                "theta": grasp_theta,
                "width": grasp_width,
            }
        except Exception as e:
            rospy.logerr(f"Model processing error: {e}")
            return None

    def _visualize_results(self, depth_image, q_img, angle_img, grasp_x, grasp_y):
        """Визуализация промежуточных результатов обработки.
        
        Args:
            depth_image (numpy.ndarray): Нормализованное глубинное изображение.
            q_img (numpy.ndarray): Карта качества захвата.
            angle_img (numpy.ndarray): Карта углов захвата.
            grasp_x (int): X-координата точки захвата.
            grasp_y (int): Y-координата точки захвата.
        """
        depth_colored = cv2.applyColorMap((depth_image * 255).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.circle(depth_colored, (grasp_x, grasp_y), 5, (0, 255, 0), -1)
        cv2.imshow("Grasp Point", depth_colored)
        cv2.imshow("Quality of grasp", q_img)
        cv2.imshow("Angle map", angle_img)
        cv2.waitKey(1)