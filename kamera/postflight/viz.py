import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation


def visualize_quaternions(q1, q2, q3=None):
    """
    Visualizes three quaternions in 3D space by showing their corresponding rotated coordinate frames.

    Parameters:
    - q1: list or array of 4 elements representing the first quaternion [w, x, y, z]
    - q2: list or array of 4 elements representing the second quaternion [w, x, y, z]
    - q3: list or array of 4 elements representing the second quaternion [w, x, y, z]
    """

    def normalize(q):
        norm = np.linalg.norm(q)
        return q / norm if norm != 0 else q

    # Normalize the quaternions
    q1 = normalize(np.array(q1, dtype=np.float64))
    q2 = normalize(np.array(q2, dtype=np.float64))
    q3 = normalize(np.array(q3, dtype=np.float64))

    # Convert to rotation matrices
    R1 = Rotation.from_quat(q1).as_matrix()
    R2 = Rotation.from_quat(q2).as_matrix()
    # R3 = Rotation.from_quat(q3).as_matrix()

    # Original coordinate axes
    origin = np.zeros(3)
    axes = np.eye(3)  # 3x3 identity matrix

    # Apply rotations
    rotated_axes1 = R1 @ axes
    rotated_axes2 = R2 @ axes
    # rotated_axes3 = R3 @ axes

    # Plotting
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Function to plot axes
    def plot_axes(ax, origin, axes, colors, labels, linewidth=2, alpha=1.0):
        for i in range(3):
            ax.quiver(
                origin[0],
                origin[1],
                origin[2],
                axes[0, i],
                axes[1, i],
                axes[2, i],
                color=colors[i],
                label=labels[i] if origin is origin else "",
                linewidth=linewidth,
                arrow_length_ratio=0.1,
                alpha=alpha,
            )

    # Plot original axes
    plot_axes(
        ax, origin, axes, ["k", "k", "k"], ["X", "Y", "Z"], linewidth=1, alpha=0.3
    )

    # Plot first quaternion rotated axes
    plot_axes(
        ax,
        origin,
        rotated_axes1,
        ["r", "g", "b"],
        ["Q1 X", "Q1 Y", "Q1 Z"],
        linewidth=2,
        alpha=0.8,
    )

    # Plot second quaternion rotated axes
    plot_axes(
        ax,
        origin,
        rotated_axes2,
        ["c", "m", "y"],
        ["Q2 X", "Q2 Y", "Q2 Z"],
        linewidth=2,
        alpha=0.8,
    )

    # Plot third quaternion rotated axes
    # plot_axes(ax, origin, rotated_axes3, ["b", "m", "y"], ['Q3 X', 'Q3 Y', 'Q3 Z'], linewidth=2, alpha=0.8)

    # Setting the labels
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")

    # Setting the aspect ratio to be equal
    ax.set_box_aspect([1, 1, 1])  # Requires matplotlib >= 3.3.0

    # Setting the limits
    limit = 1.0
    ax.set_xlim([-limit, limit])
    ax.set_ylim([-limit, limit])
    ax.set_zlim([-limit, limit])

    # Adding a legend
    # To avoid duplicate labels in legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    plt.title("Visualization of Two Quaternions in 3D Space")
    plt.show()


# Example Usage
if __name__ == "__main__":
    # Define two quaternions
    # Quaternion q1 represents a rotation of 90 degrees around the X-axis
    q1 = [np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0]

    # Quaternion q2 represents a rotation of 90 degrees around the Y-axis
    q2 = [np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0]

    # Quaternion q3 represents a rotation of 90 degrees around the Z-axis
    q3 = [np.cos(np.pi / 4), np.sin(np.pi / 4), np.sin(np.pi / 4), 0]

    visualize_quaternions(q1, q2, q3)
