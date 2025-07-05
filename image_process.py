import rospy
import numpy as np
import cv2
from sklearn.decomposition import PCA
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class GGCNNGraspDetector:
    def __init__(self):
        self.bridge = CvBridge()
        rospy.Subscriber('/camera_3_depth/depth/image_raw', Image, self.depth_image_callback)
        self.grasp_params = None
        
        # Параметры камеры (настройте под вашу камеру)
        self.fx = 640.0  # Фокусное расстояние по X
        self.fy = 640.0  # Фокусное расстояние по Y
        self.cx = 960 // 2  # Центр изображения по X
        self.cy = 540 // 2  # Центр изображения по Y
        
        # Параметры обработки глубины
        self.min_depth = 0.3  # Минимальная глубина (м)
        self.max_depth = 1.2   # Максимальная глубина (м)
        self.min_points = 500  # Минимальное количество точек для объекта
        
        rospy.loginfo("Geometric Grasp Detector initialized")

    def depth_image_callback(self, data):
        try:
            # Конвертация ROS Image -> OpenCV
            depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
            
            if depth_image is None or depth_image.size == 0:
                rospy.logwarn("Empty depth image received!")
                return
            
            # Предобработка depth map
            processed_depth = self.preprocess_depth_image(depth_image)
            
            # Определение параметров захвата
            self.grasp_params = self.process_geometric(processed_depth)

        except Exception as e:
            rospy.logerr(f"Processing error: {e}")

    def preprocess_depth_image(self, depth_image):
        # Конвертируем в float32 и заменяем нули на максимальную глубину
        depth_image = depth_image.astype(np.float32)
        invalid_mask = (depth_image == 0) | np.isnan(depth_image)
        depth_image[invalid_mask] = self.max_depth * 1000  # мм
        
        # Медианный фильтр для уменьшения шума
        filtered = cv2.medianBlur(depth_image, 5)
        
        # Нормализация для визуализации
        return filtered

    def process_geometric(self, depth_image):
        # Получаем 3D точки рабочей области
        points = self.depth_to_pointcloud(depth_image)
        
        if len(points) < self.min_points:
            rospy.logwarn(f"No object detected (only {len(points)} points)")
            return None

        # Кластеризация DBSCAN для отделения объекта от поверхности
        cluster_points = self.extract_object_points(points)
        if len(cluster_points) < self.min_points:
            rospy.logwarn(f"Object too small ({len(cluster_points)} points)")
            return None

        # Вычисляем OBB
        center, axes, dimensions = self.compute_obb(cluster_points)
        
        # Определяем точку захвата (5 см над центром верхней грани)
        grasp_point_3d = center + axes[2] * (dimensions[2]/2 + 0.05)
        
        # Проекция в 2D
        grasp_x, grasp_y = self.project_to_image(grasp_point_3d)
        
        # Вычисляем угол захвата
        angle = np.arctan2(axes[0][1], axes[0][0])
        
        # Визуализация
        self.visualize(depth_image, grasp_x, grasp_y, cluster_points)

        return {
            "x": grasp_x,
            "y": grasp_y,
            "z": grasp_point_3d[2],
            "theta": angle,
            "width": min(dimensions[0], dimensions[1]) / 1000.0
        }

    def depth_to_pointcloud(self, depth_image):
        height, width = depth_image.shape
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        
        # Маскируем только рабочую область
        mask = (depth_image > self.min_depth*1000) & (depth_image < self.max_depth*1000)
        z = depth_image[mask] / 1000.0  # Переводим мм в метры
        u = u[mask]
        v = v[mask]
        
        # Пересчет в 3D координаты
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        
        return np.column_stack((x, y, z))

    def extract_object_points(self, points):
        # Вычитаем плоскость стола (RANSAC)
        from sklearn.linear_model import RANSACRegressor
        model = RANSACRegressor()
        model.fit(points[:,:2], points[:,2])
        
        # Удаляем точки, принадлежащие плоскости
        dist = np.abs(points[:,2] - model.predict(points[:,:2]))
        return points[dist > 0.02]  # 2 см порог для объекта

    def compute_obb(self, points):
        pca = PCA(n_components=3)
        pca.fit(points)
        center = np.mean(points, axis=0)
        axes = pca.components_
        projected = pca.transform(points)
        dimensions = np.max(projected, axis=0) - np.min(projected, axis=0)
        return center, axes, dimensions

    def project_to_image(self, point_3d):
        x = int((point_3d[0] * self.fx / point_3d[2]) + self.cx)
        y = int((point_3d[1] * self.fy / point_3d[2]) + self.cy)
        return x, y

    def visualize(self, depth_image, x, y, points=None):
        # Нормализация глубины для визуализации
        vis = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)  # Конвертируем в цветное
        
        # Рисуем точку захвата (зеленый круг)
        cv2.circle(vis, (x, y), 10, (0, 255, 0), -1)
        
        if points is not None and len(points) > 0:
            # Вычисляем PCA и OBB для точек объекта
            center, axes, dimensions = self.compute_obb(points)
            
            # Проекция центра в 2D
            center_2d = self.project_to_image(center)
            
            # Масштабируем оси для визуализации (20 см)
            scale = 0.2
            axis1_end = center + axes[0] * scale
            axis2_end = center + axes[1] * scale
            axis3_end = center + axes[2] * scale
            
            # Проекция концов осей в 2D
            axis1_end_2d = self.project_to_image(axis1_end)
            axis2_end_2d = self.project_to_image(axis2_end)
            axis3_end_2d = self.project_to_image(axis3_end)
            
            # Рисуем главные компоненты (оси PCA)
            cv2.line(vis, center_2d, axis1_end_2d, (255, 0, 0), 2)  # Первая компонента (красная)
            cv2.line(vis, center_2d, axis2_end_2d, (0, 255, 0), 2)  # Вторая компонента (зеленая)
            cv2.line(vis, center_2d, axis3_end_2d, (0, 0, 255), 2)  # Третья компонента (синяя)
            
            # Рисуем OBB (ориентированный ограничивающий прямоугольник)
            # Создаем вершины OBB в 3D
            half_dims = dimensions / 2
            corners = []
            for i in [-1, 1]:
                for j in [-1, 1]:
                    for k in [-1, 1]:
                        corner = center + axes[0] * i * half_dims[0] + \
                                 axes[1] * j * half_dims[1] + \
                                 axes[2] * k * half_dims[2]
                        corners.append(corner)
            
            # Проекция вершин в 2D
            corners_2d = [self.project_to_image(c) for c in corners]
            
            # Рисуем ребра OBB
            edges = [
                (0,1), (0,2), (0,4),
                (1,3), (1,5),
                (2,3), (2,6),
                (3,7),
                (4,5), (4,6),
                (5,7),
                (6,7)
            ]
            for (i,j) in edges:
                cv2.line(vis, corners_2d[i], corners_2d[j], (0, 255, 255), 2)  # Желтые линии
            
            # Рисуем точки облака (каждая 50-я для производительности)
            for pt in points[::50]:
                px, py = self.project_to_image(pt)
                cv2.circle(vis, (px, py), 1, (255, 0, 0), -1)
        
        cv2.imshow("Grasp Detection (PCA + OBB)", vis)
        cv2.waitKey(1)