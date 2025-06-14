import sys
import rospy
import numpy as np
import torch
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class GGCNNGraspDetector:
    def __init__(self):
        # Для обработки глубинной карты будем использовать графический процессор
        self.device = torch.device("cuda")  # Используем GPU
        
        # Игнорировать предупреждения
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        
        # Проверка CUDA
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Загрузка модели
        model_path = '../ggcnn/ggcnn_weights_cornell/ggcnn_epoch_23_cornell'
        try:
            self.model = torch.load(model_path, map_location=self.device)
            self.model.eval()  # Переключить в режим оценки
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        # Импортируем модель нейросети
        torch.nn.Module.dump_patches = True
        
        # self.model = torch.load('../ggcnn/ggcnn_weights_cornell/ggcnn_epoch_23_cornell')

        # Инициализируем CvBridge для конвертации ROS Image в OpenCV
        self.bridge = CvBridge()

        # Подписываемся на топик с глубинным изображением
        rospy.Subscriber('/camera_3_depth/depth/image_raw', Image, self.depth_image_callback)

        # Переменные для хранения данных
        self.depth_image = None
        self.grasp_params = None  # Параметры захвата

        # Сообщим, что инициализация прошла успешно
        rospy.loginfo("GGCNNGraspDetector class was initialized")

    def depth_image_callback(self, data):
        """
        Callback для получения глубинного изображения и его обработки.
        """
        try:
            # Конвертируем сообщение ROS Image в OpenCV формат
            depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
            
            # Проверяем, что изображение корректное
            if depth_image is None or depth_image.size == 0:
                rospy.logwarn("Получено пустое глубинное изображение!")
                return
            
            # Нормализуем глубинное изображение
            depth_image = self.preprocess_depth_image(depth_image)
            
            # Передаём глубинное изображение в нейросеть
            self.grasp_params = self.process_with_ggcnn(depth_image)

        except CvBridgeError as e:
            rospy.logerr(f"Ошибка конвертации изображения: {e}")
        except Exception as e:
            rospy.logerr(f"Ошибка при обработке глубинного изображения: {e}")

    def preprocess_depth_image(self, depth_image):
        """
        Нормализация глубинного изображения.
        """
        # Заменяем NaN на 0
        depth_image = np.nan_to_num(depth_image, nan=0.0, posinf=0.0, neginf=0.0)

        # Нормализуем изображение
        min_depth = np.min(depth_image)
        max_depth = np.max(depth_image)
        depth_image = (depth_image - min_depth) / (max_depth - min_depth)

        return depth_image

    def process_with_ggcnn(self, depth_image, k=40):
        """
        Возвращает усреднённые (x, y) и theta по k точкам с максимальным q_img.
        """
        # Преобразование в тензор
        depth_tensor = torch.tensor(depth_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Прогон через модель
        try:
            with torch.no_grad():
                q_img, angle_img, width_img, pos_img = self.model(depth_tensor)
        except Exception as e:
            rospy.logerr(f"Ошибка модели: {e}")
            return None

        # Конвертация в numpy
        q_img = q_img.cpu().squeeze().numpy()
        angle_img = angle_img.cpu().squeeze().numpy()
        width_img = width_img.cpu().squeeze().numpy()
        pos_img = pos_img.cpu().squeeze().numpy()

        # Находим k точек с максимальным q_img
        top_k_indices = np.argpartition(q_img.flatten(), -k)[-k:]  # Индексы топ-k значений
        top_k_indices_2d = np.unravel_index(top_k_indices, q_img.shape)  # (y_coords, x_coords)

        # Усреднение координат
        grasp_y = np.mean(top_k_indices_2d[0]).astype(int)  # Средний y
        grasp_x = np.mean(top_k_indices_2d[1]).astype(int)  # Средний x

        # Усреднение угла theta по тем же k точкам
        grasp_theta = np.mean(angle_img[top_k_indices_2d])

        # Параметры для усреднённой точки (или можно усреднить и их)
        grasp_width = width_img[grasp_y, grasp_x]
        grasp_z = pos_img[grasp_y, grasp_x]

        # Визуализация
        depth_colored = cv2.applyColorMap((depth_image * 255).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.circle(depth_colored, (grasp_x, grasp_y), 5, (0, 255, 0), -1)
        cv2.imshow("Grasp Point", depth_colored)
        cv2.imshow("Quality of grasp", q_img)
        cv2.imshow("Angle map", angle_img)
        cv2.waitKey(1)

        return {
            "x": grasp_x,
            "y": grasp_y,
            "z": grasp_z,
            "theta": grasp_theta,
            "width": grasp_width,
        }
