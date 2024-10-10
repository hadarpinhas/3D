import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Step 1: Load and preprocess the images
def load_images(left_image_path, right_image_path):
    img_left = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
    img_right = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)

    # Ensure images are the same size
    assert img_left.shape == img_right.shape, "Images must be the same size"

    # Convert images to tensors and normalize
    tensor_left = torch.from_numpy(img_left).float().unsqueeze(0).unsqueeze(0) / 255.0
    tensor_right = torch.from_numpy(img_right).float().unsqueeze(0).unsqueeze(0) / 255.0

    return tensor_left, tensor_right

# Step 2: Define the Stereo Depth Estimation Model
class StereoDepthEstimationModel(nn.Module):
    def __init__(self, max_disparity=64):
        super(StereoDepthEstimationModel, self).__init__()
        self.max_disparity = max_disparity

        # Simple CNN feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.ReLU(inplace=True)
        )

    def forward(self, left, right):
        # Extract features
        feat_left = self.feature_extractor(left)
        feat_right = self.feature_extractor(right)

        # Initialize disparity map
        disparity_map = torch.zeros(left.size()[2:], dtype=torch.float32)

        # Compute disparity
        with torch.no_grad():
            # For each disparity level
            for d in range(self.max_disparity):
                # Shift the right feature map
                shifted_right = F.pad(feat_right, (d, 0, 0, 0))[:, :, :, :-d] if d > 0 else feat_right

                # Compute similarity (e.g., sum of absolute differences)
                sad = torch.mean(torch.abs(feat_left[:, :, :, :shifted_right.size(3)] - shifted_right), dim=1)

                # Update disparity map
                if d == 0:
                    cost_volume = sad
                else:
                    cost_volume = torch.cat((cost_volume, sad), dim=1)

            # Get disparity with minimum cost
            disparity_map = torch.argmin(cost_volume, dim=1).squeeze(0).cpu().numpy()

        return disparity_map

# Step 3: Compute the Disparity Map
def compute_disparity(model, left_image, right_image):
    model.eval()
    disparity_map = model(left_image, right_image)
    return disparity_map

# Step 4: Generate the 3D Point Cloud
def generate_point_cloud(disparity_map, focal_length, baseline):
    # Create coordinate grid
    h, w = disparity_map.shape
    Q = np.float32([
        [1, 0, 0, -w/2],
        [0, -1, 0, h/2],
        [0, 0, 0, -focal_length],
        [0, 0, 1/baseline, 0]
    ])
    points_3D = cv2.reprojectImageTo3D(disparity_map, Q)
    return points_3D

# Step 5: Visualize the Results
def visualize_disparity(disparity_map):
    plt.imshow(disparity_map, cmap='jet')
    plt.colorbar()
    plt.title('Disparity Map')
    plt.show()

def visualize_point_cloud(points_3D, colors):
    # Filter out points with infinite depth
    mask = np.isfinite(points_3D[:, :, 2])
    x = points_3D[:, :, 0][mask]
    y = points_3D[:, :, 1][mask]
    z = points_3D[:, :, 2][mask]
    rgb = colors[mask]

    # Sample points for visualization
    sample_indices = np.random.choice(len(x), size=10000, replace=False)
    x_sample = x[sample_indices]
    y_sample = y[sample_indices]
    z_sample = z[sample_indices]
    rgb_sample = rgb[sample_indices]

    # Plot using matplotlib
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(x_sample, y_sample, z_sample, c=rgb_sample/255.0, s=0.1)
    plt.title('3D Point Cloud')
    plt.show()

# Main function to run the steps
def main():
    # Replace with paths to your stereo images
    left_image_path = 'left_image.png'
    right_image_path = 'right_image.png'

    # Load images
    left_image, right_image = load_images(left_image_path, right_image_path)

    # Instantiate the model
    model = StereoDepthEstimationModel(max_disparity=64)

    # Compute disparity map
    disparity_map = compute_disparity(model, left_image, right_image)

    # Visualize disparity map
    visualize_disparity(disparity_map)

    # Generate 3D point cloud
    focal_length = 1.0  # Example value, replace with actual focal length
    baseline = 0.1      # Example value, replace with actual baseline distance
    points_3D = generate_point_cloud(disparity_map, focal_length, baseline)

    # Load color image for visualization
    img_left_color = cv2.imread(left_image_path, cv2.IMREAD_COLOR)
    visualize_point_cloud(points_3D, img_left_color)

if __name__ == '__main__':
    main()
