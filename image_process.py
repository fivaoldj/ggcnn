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

        # Импортируем модель нейросети
        torch.nn.Module.dump_patches = True
        self.model = torch.load('../ggcnn/ggcnn_weights_cornell/ggcnn_epoch_23_cornell')

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
            
            # Нормализуем глубинное изображение и обрезаем центральную часть
            depth_image = self.preprocess_depth_image(depth_image)
            
            # Передаём глубинное изображение в нейросеть
            self.grasp_params = self.process_with_ggcnn(depth_image)

            # Выводим результаты захвата
            # rospy.loginfo(f"Grasp Params: {self.grasp_params}")
        except CvBridgeError as e:
            rospy.logerr(f"Ошибка конвертации изображения: {e}")
        except Exception as e:
            rospy.logerr(f"Ошибка при обработке глубинного изображения: {e}")

    def preprocess_depth_image(self, depth_image):
        """
        Нормализация глубинного изображения и обрезка центральной области.
        """
        # Заменяем NaN на 0
        depth_image = np.nan_to_num(depth_image, nan=0.0, posinf=0.0, neginf=0.0)

        # Обрезаем центральную часть изображения
        height, width = depth_image.shape
        center_x, center_y = width // 2, height // 2
        crop_width, crop_height = 640, 360  # Обрезаем до размера 640x360
        crop_left = center_x - crop_width // 2
        crop_right = center_x + crop_width // 2
        crop_top = center_y - crop_height // 2
        crop_bottom = center_y + crop_height // 2

        depth_image_cropped = depth_image[crop_top:crop_bottom, crop_left:crop_right]

        # Нормализуем изображение
        min_depth = np.min(depth_image_cropped)
        max_depth = np.max(depth_image_cropped)
        depth_image_cropped = (depth_image_cropped - min_depth) / (max_depth - min_depth)

        # Отображение глубинного изображения для отладки
        cv2.imshow("Cropped Depth Image", depth_image_cropped)
        cv2.waitKey(1)  # Ждём 1 миллисекунду для обновления окна

        return depth_image_cropped

    def process_with_ggcnn(self, depth_image):
        """
        Передаёт глубинную карту в нейросеть GGCNN и возвращает параметры захвата.
        """
        # Преобразуем глубинное изображение в формат тензора
        depth_tensor = torch.tensor(depth_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

        # Прогон через модель
        try:
            with torch.no_grad():
                output = self.model(depth_tensor)
        except Exception as e:
            rospy.logerr(f"Ошибка при работе модели: {e}")
            return None

        # Извлекаем каналы (предполагается, что модель возвращает 4 канала)
        q_img, angle_img, width_img, pos_img = output

        # Преобразуем тензоры в numpy
        q_img = q_img.cpu().squeeze().numpy()
        angle_img = angle_img.cpu().squeeze().numpy()
        width_img = width_img.cpu().squeeze().numpy()
        pos_img = pos_img.cpu().squeeze().numpy()

        # Ищем лучшую точку захвата
        max_q_idx = np.unravel_index(np.argmax(q_img), q_img.shape)
        grasp_x, grasp_y = max_q_idx[1], max_q_idx[0]
        grasp_theta = angle_img[max_q_idx]
        grasp_width = width_img[max_q_idx]
        grasp_z = pos_img[max_q_idx]

        return {
            "x": grasp_x,
            "y": grasp_y,
            "z": grasp_z,
            "theta": grasp_theta,
            "width": grasp_width,
        }
