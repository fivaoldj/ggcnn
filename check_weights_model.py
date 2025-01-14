import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

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

device = torch.device("cuda")  # Используйте GPU

torch.nn.Module.dump_patches = True
model = torch.load('ggcnn_weights_cornell/ggcnn_epoch_23_cornell')
print(model)

# Загружаем глубинное изображение (например, raw numpy array)
raw_depth_image = np.random.rand(300, 300)  # Ваше сырое глубинное изображение
# Преобразуем в тензор PyTorch (добавляем необходимые измерения)
depth_tensor = torch.from_numpy(raw_depth_image).float().unsqueeze(0).unsqueeze(0)
depth_tensor = depth_tensor.to(device)  # Переместите тензор на GPU
output = model(depth_tensor)
visualize_output_ggcnn(output)
