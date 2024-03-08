from PIL import Image
import matplotlib.pyplot as plt



def display_images(input_image_path, output_image_paths):
    """
    Display images in a single window, with the input image on the top row and
    up to 10 output images on the bottom row.

    Parameters:
    - input_image_path: Path to the input image.
    - output_image_paths: List of paths to the output images (max 10).
    """
    
    # Load the input image
    input_img = Image.open(input_image_path)
    
    # Determine the number of columns based on the number of output images
    num_output_images = len(output_image_paths)
    num_cols = max(1, num_output_images)
    
    # Initialize the plot with 2 rows and a dynamic number of columns
    fig, axes = plt.subplots(2, num_cols, figsize=(20, 10))  # Adjust the figsize as needed
    
    # Display the input image in the first row, centered
    if num_cols > 1:
        # If there are multiple columns, remove the axes for all but the first in the top row
        for ax in axes[0, 1:]:
            ax.axis('off')
        ax_input = axes[0, 0]
    else:
        # If there is only one column, use the single axis for the top row directly
        ax_input = axes[0]
        
    ax_input.imshow(input_img)
    ax_input.set_title('Input Image')
    ax_input.axis('off')
    
    # Display the output images in the second row
    if num_cols > 1:
        for i, img_path in enumerate(output_image_paths):
            output_img = Image.open(img_path)
            axes[1, i].imshow(output_img)
            axes[1, i].set_title(f'Output {i+1}')
            axes[1, i].axis('off')
        # Hide any unused axes in the second row
        for j in range(i + 1, num_cols):
            axes[1, j].axis('off')
    else:
        # If there's only one output, display it directly in the second row
        output_img = Image.open(output_image_paths[0])
        axes[1].imshow(output_img)
        axes[1].set_title('Output 1')
        axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
