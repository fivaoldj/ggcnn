import sys
import rospy
import numpy as np
import torch
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
import matplotlib.pyplot as plt

def info_message(message):
    print("[INFO]" + message)

info_message("start")

height, width = 720, 1280  # Размеры карты глубины
fx, fy = 639.997649, 639.997649  # Фокальные расстояния камеры
cx, cy = 640, 360  # Координаты оптического центра (зависит от камеры)

def visualize_output_ggcnn(output):
    # Преобразуем тензор в numpy, отсоединив его от графа вычислений
    q_map = output[0].detach().squeeze().cpu().numpy()  # Карта качества захвата
    width_map = output[1].detach().squeeze().cpu().numpy()  # Карта ширины захвата
    angle_map = output[2].detach().squeeze().cpu().numpy()  # Карта углов
    offset_map = output[3].detach().squeeze().cpu().numpy()  # Карта смещения


    plt.figure(figsize=(8, 6))
    plt.imshow(q_map, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Capture Quality')
    plt.title('Capture Quality Map (q)')
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.imshow(width_map, cmap='plasma', interpolation='nearest')
    plt.colorbar(label='Grasp Width')
    plt.title('Grasp Width Map')
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.imshow(angle_map, cmap='twilight', interpolation='nearest')
    plt.colorbar(label='Grasp Angle (radians)')
    plt.title('Grasp Angle Map')
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.imshow(offset_map, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(label='Offset')
    plt.title('Offset Map')
    plt.show()

    # Найдем координаты максимума карты качества захвата
    max_idx = np.unravel_index(np.argmax(q_map), q_map.shape)
    max_x, max_y = max_idx

    # Визуализируем карту качества с точкой максимума
    plt.figure(figsize=(8, 6))
    plt.imshow(q_map, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Capture Quality')
    plt.scatter(max_y, max_x, color='red', marker='x', s=100)  # Красный крестик на максимуме
    plt.title('Capture Quality Map with Maximum Point')
    plt.show()


    # Применим гауссово размытие для сглаживания
    smoothed_q_map = gaussian_filter(q_map, sigma=2)

    # Визуализируем
    plt.figure(figsize=(8, 6))
    plt.imshow(smoothed_q_map, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Smoothed Capture Quality')
    plt.title('Smoothed Capture Quality Map')
    plt.show()

def pointcloud_to_depth_map(msg, height, width):
    """
    Преобразует сообщение PointCloud2 в Depth Map.
    
    :param msg: Сообщение PointCloud2
    :param height: Высота ожидаемой карты глубины
    :param width: Ширина ожидаемой карты глубины
    :return: Depth Map (numpy массив)
    """
    # Инициализация пустой depth map
    depth_map = np.full((height, width), np.nan, dtype=np.float32)

    # Преобразуем облако точек в читаемый массив
    for point in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
        x, y, z = point  # Координаты точки

        # Преобразуем координаты в пиксели (прямое отображение камеры)
        u = int((x / z) * fx + cx)  # Горизонтальная координата
        v = int((y / z) * fy + cy)  # Вертикальная координата

        if 0 <= u < width and 0 <= v < height:
            depth_map[v, u] = z  # Записываем глубину в карту

    # Заполняем NaN значения максимальным значением глубины
    depth_map = np.nan_to_num(depth_map, nan=np.max(depth_map))
    return depth_map

# Основной обработчик ROS
def callback(msg):
    device = torch.device("cuda")  # Используйте GPU

    torch.nn.Module.dump_patches = True
    model = torch.load('ggcnn_weights_cornell/ggcnn_epoch_23_cornell')

    depth_map = pointcloud_to_depth_map(msg, height, width)
    print("Depth Map создана с размерами:", depth_map.shape)

    # Теперь Depth Map можно подать в GGCNN
    depth_tensor = torch.tensor(depth_map).unsqueeze(0).unsqueeze(0).float()
    depth_tensor = depth_tensor.to(device)  # Переместите тензор на GPU
    output = model(depth_tensor)
    plt.imshow(depth_map, cmap='viridis')
    plt.colorbar(label='Глубина (м)')
    plt.title('Карта глубины')
    plt.show()

    depth_tensor = torch.tensor(depth_map).unsqueeze(0).unsqueeze(0).float().to(device)
    output = model(depth_tensor)
    visualize_output_ggcnn(output)

rospy.init_node('pointcloud_to_depth_map')
info_message("init ros node")
rospy.Subscriber('/camera_1_stereo/points2', PointCloud2, callback)
info_message("init subscriber")
rospy.spin()

# except rospy.ROSInterruptException:
#     pass
# except KeyboardInterrupt:
#     pass
