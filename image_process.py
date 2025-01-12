import sys
import rospy
import numpy as np
import torch
import cv2
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2

class GGCNNGraspDetector:
    def __init__(self):
        # Для обработки нейросетью глубинной карты будем использовать графическйи процессор
        self.device = torch.device("cuda")  # Используйте GPU

        # Импортируем модель нейросети
        torch.nn.Module.dump_patches = True
        self.model = torch.load('../ggcnn/ggcnn_weights_cornell/ggcnn_epoch_23_cornell')

        # Подпишемся на топик, который будет вызывать метод pointcloud_callback класса
        rospy.Subscriber('/camera_3_stereo/points2', PointCloud2, self.pointcloud_callback)
        
        # Переменные для хранения данных
        self.depth_image = None
        self.grasp_params = None  # Здесь будут сохраняться параметры захвата

        # Сообщим, что инициализация прошла успешно
        rospy.loginfo("GGCNNGraspDetector class was ititialized")

    def pointcloud_callback(self, data):
        """
        Callback для получения данных из PointCloud2 и преобразования их в глубинную карту.
        """
        try:
            # Преобразование PointCloud2 в numpy-массив (только x, y, z)
            points = pc2.read_points(data, field_names=("x", "y", "z"), skip_nans=True)
            point_array = np.array(list(points))

            # Проверяем, что облако точек не пустое
            if point_array.shape[0] == 0:
                rospy.logwarn("PointCloud2 пустое!")
                return

            # Проверка на наличие NaN или Inf
            if np.any(np.isnan(point_array)) or np.any(np.isinf(point_array)):
                rospy.logwarn("Point array содержит NaN или Inf!")
                return

            # Преобразование в глубинную карту
            depth_image = self.pointcloud_to_depth_map(point_array)
            
            # Передача глубинной карты в нейросеть
            self.grasp_params = self.process_with_ggcnn(depth_image)

            # Вывод результатов
            # rospy.loginfo(f"Grasp Params: {self.grasp_params}")
        except Exception as e:
            rospy.logerr(f"Ошибка при обработке PointCloud2: {e}")

    def pointcloud_to_depth_map(self, point_array):
        """
        Преобразует облако точек в глубинную карту и обрезает её относительно центра.
        """
        # Настройка параметров камеры
        IMAGE_WIDTH = 1280
        IMAGE_HEIGHT = 720
        FX = 639.997649  # Фокусное расстояние по X
        FY = 639.997649  # Фокусное расстояние по Y
        CX = IMAGE_WIDTH / 2
        CY = IMAGE_HEIGHT / 2

        # Глубинная карта
        depth_image = np.full((IMAGE_HEIGHT, IMAGE_WIDTH), np.nan, dtype=np.float32)

        # Цикл по точкам
        for idx, point in enumerate(point_array):
            try:
                if len(point) != 3:
                    rospy.logwarn(f"Точка {idx} имеет неверное количество значений: {point}")
                    continue
                x, y, z = point
                if z > 0:  # Отбрасываем точки за камерой
                    u = int((x * FX / z) + CX)
                    v = int((y * FY / z) + CY)
                    if 0 <= u < IMAGE_WIDTH and 0 <= v < IMAGE_HEIGHT:
                        depth_image[v, u] = z
            except Exception as e:
                rospy.logerr(f"Ошибка при обработке точки {idx}: {e}")
                continue

        # Заполняем пробелы методом интерполяции
        depth_image = self.fill_nan_gaps(depth_image)

        # Обрезаем центральную часть изображения
        center_x, center_y = IMAGE_WIDTH // 2, IMAGE_HEIGHT // 2
        crop_width, crop_height = 640, 360  # Обрезаем до центральной области 640x360

        # Рассчитываем границы для обрезки
        crop_left = center_x - crop_width // 2
        crop_right = center_x + crop_width // 2
        crop_top = center_y - crop_height // 2
        crop_bottom = center_y + crop_height // 2

        # Обрезаем изображение
        depth_image_cropped = depth_image[crop_top:crop_bottom, crop_left:crop_right]

        cv2.imshow("Cropped Depth Image", depth_image_cropped)
        cv2.waitKey(1)  # Ждем 1 миллисекунду, чтобы обновить окно

        return depth_image_cropped

    def fill_nan_gaps(self, depth_image):
        """
        Заполняет пробелы (NaN) в глубинной карте методом интерполяции.
        """
        mask = np.isnan(depth_image)
        depth_image[mask] = 0
        depth_image = cv2.inpaint(depth_image.astype(np.float32), mask.astype(np.uint8), inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        return depth_image

    def process_with_ggcnn(self, depth_image):
        """
        Передает глубинную карту в нейросеть GGCNN и возвращает параметры захвата.
        """
        # Преобразование глубинной карты в формат тензора
        depth_image = (depth_image - np.nanmin(depth_image)) / (np.nanmax(depth_image) - np.nanmin(depth_image))
        depth_tensor = torch.tensor(depth_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

        # Прогон через модель
        try:
            with torch.no_grad():
                output = self.model(depth_tensor)
        except Exception as e:
            rospy.logerr(f"Error during model inference: {e}")

        # Используем 4 канала, если они присутствуют
        q_img, angle_img, width_img, pos_img = output
        
        # Преобразование тензоров в numpy
        q_img = q_img.cpu().squeeze().numpy()
        angle_img = angle_img.cpu().squeeze().numpy()
        width_img = width_img.cpu().squeeze().numpy()
        pos_img = pos_img.cpu().squeeze().numpy()  # Добавлен вывод для позиции захвата

        # Поиск лучшей точки захвата
        max_q_idx = np.unravel_index(np.argmax(q_img), q_img.shape)
        grasp_x, grasp_y = max_q_idx[1], max_q_idx[0]
        grasp_theta = angle_img[max_q_idx]
        grasp_width = width_img[max_q_idx]

        # Позиция захвата (если она нужна)
        grasp_z = pos_img[max_q_idx]  # Возьмём значение позиции из pos_img

        return {
            "x": grasp_x,
            "y": grasp_y,
            "z": grasp_z,  # Теперь позиция захвата тоже учитывается
            "theta": grasp_theta,
            "width": grasp_width,
        }