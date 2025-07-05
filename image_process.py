import sys
import rospy
import numpy as np
import torch
import cv2
from sklearn.decomposition import PCA
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class GGCNNGraspDetector:
    """Класс для детекции точек захвата с использованием GGCNN."""
    
    def __init__(self, q_threshold=0.5, use_rgb=True):
        """Инициализация детектора.
        
        Args:
            q_threshold (float): Порог для отбора точек захвата (по умолчанию 0.5)
            use_rgb (bool): Использовать RGB изображение для визуализации (если True)
        """
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.q_threshold = q_threshold
        self.use_rgb = use_rgb
        print(f"Using device: {self.device}, q_threshold: {q_threshold}, use_rgb: {use_rgb}")
        
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

        self.bridge = CvBridge()
        self.depth_image = None
        self.rgb_image = None  # Для хранения RGB изображения
        self.grasp_params = None
        
        # Подписки на топики
        rospy.Subscriber('/camera_3_depth/depth/image_raw', Image, self.depth_image_callback)
        if self.use_rgb:
            rospy.Subscriber('/camera_3/rgb/image_raw', Image, self.rgb_image_callback)
        rospy.loginfo("GGCNNGraspDetector initialized successfully")

    def depth_image_callback(self, data):
        """Обработка глубинного изображения."""
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
            
            if self.depth_image is None or self.depth_image.size == 0:
                rospy.logwarn("Received empty depth image!")
                return
            
            self.depth_image = self.preprocess_depth_image(self.depth_image)
            self.grasp_params = self.process_with_ggcnn(self.depth_image)

        except CvBridgeError as e:
            rospy.logerr(f"Depth image conversion error: {e}")
        except Exception as e:
            rospy.logerr(f"Depth image processing error: {e}")

    def rgb_image_callback(self, data):
        """Обработка RGB изображения (если use_rgb=True)."""
        try:
            self.rgb_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"RGB image conversion error: {e}")

    def preprocess_depth_image(self, depth_image):
        """Нормализация глубинного изображения."""
        depth_image = np.nan_to_num(depth_image, nan=0.0, posinf=0.0, neginf=0.0)
        min_depth = np.min(depth_image)
        max_depth = np.max(depth_image)
        return (depth_image - min_depth) / (max_depth - min_depth)

    def process_with_ggcnn(self, depth_image):
        """Основная обработка изображения."""
        try:
            depth_tensor = torch.tensor(depth_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                q_img, _, _, pos_img = self.model(depth_tensor)
                
            q_img = q_img.cpu().squeeze().numpy()
            pos_img = pos_img.cpu().squeeze().numpy()

            top_points = self._get_top_grasp_points(q_img)
            
            if len(top_points) == 0:
                rospy.logwarn(f"No points with q >= {self.q_threshold} found!")
                return None

            center, angle = self._calculate_grasp_orientation(top_points)
            grasp_z = pos_img[int(center[1]), int(center[0])]

            # Визуализация на цветном или grayscale изображении
            vis_image = self._prepare_visualization_image()
            self._visualize_results(vis_image, q_img, center, angle, top_points)

            return {
                "x": int(center[0]),
                "y": int(center[1]),
                "z": float(grasp_z),
                "theta": float(angle),
                "width": self._calculate_grasp_width(top_points)
            }
        except Exception as e:
            rospy.logerr(f"Model processing error: {e}")
            return None

    def _prepare_visualization_image(self):
        """Подготовка изображения для визуализации."""
        if self.use_rgb and self.rgb_image is not None:
            # Используем RGB изображение, если доступно
            return self.rgb_image.copy()
        else:
            # Иначе используем grayscale глубинной карты
            return cv2.cvtColor((self.depth_image * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    def _get_top_grasp_points(self, q_img):
        """Отбор точек по порогу."""
        y_coords, x_coords = np.where(q_img >= self.q_threshold)
        return np.column_stack((x_coords, y_coords))

    def _calculate_grasp_orientation(self, points):
        """Расчет ориентации захвата."""
        center = np.mean(points, axis=0)
        
        if len(points) > 1:
            pca = PCA(n_components=2)
            pca.fit(points)
            main_component = pca.components_[0]
            angle = np.arctan2(main_component[1], main_component[0])
        else:
            angle = 0.0
        
        return center, angle

    def _calculate_grasp_width(self, points):
        """Расчет ширины захвата."""
        if len(points) < 2:
            return 0.0
        pca = PCA(n_components=2)
        pca.fit(points)
        return 2 * np.sqrt(pca.explained_variance_[0])

    def _visualize_results(self, vis_image, q_img, center, angle, points):
        """Визуализация результатов."""
        # Рисуем все точки захвата
        for pt in points:
            cv2.circle(vis_image, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)
        
        # Рисуем центр масс
        cv2.circle(vis_image, (int(center[0]), int(center[1])), 5, (0, 0, 255), -1)
        
        # Рисуем линию ориентации
        line_length = 50
        end_x = int(center[0] + line_length * np.cos(angle))
        end_y = int(center[1] + line_length * np.sin(angle))
        cv2.line(vis_image, (int(center[0]), int(center[1])), (end_x, end_y), (255, 0, 0), 2)
        
        # Добавляем информацию
        cv2.putText(vis_image, f"Q: {self.q_threshold}", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(vis_image, f"Points: {len(points)}", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Отображаем результаты
        cv2.imshow("Grasp Visualization", vis_image)
        cv2.imshow("Quality Map", q_img)
        cv2.waitKey(1)