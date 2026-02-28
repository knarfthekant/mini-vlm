from torchvision import transforms
from data.custom_transforms import DynamicResize, SplitImage, GlobalAndSplitImages

def get_image_processor(patch_size, max_side_len, resize_to_max_side_len=False):
    """
    Args:
        patch_size: Size of the image patches
        max_side_len: Maximum size of the long side of the image input
        resize_to_max_side_len: Whether to resize to the maximum side length
    Returns:
        Image processor
    """
    return transforms.Compose([
        DynamicResize(patch_size, max_side_len, resize_to_max_side_len),
        transforms.ToTensor(),
        GlobalAndSplitImages(patch_size)
    ])

def get_image_string(tokenizer, splitted_image_counts, image_token_length):
    """
    Generates an string of image tokens for the given tokenizer and image token length.
    Args:
        tokenizer: Tokenizer
        splitted_image_counts: List of tuples of (n_h, n_w) of splitted chunks for each image 
        image_token_length: Length of the image token
    Returns:
        List of tuples of (image_token, image_string)
    """
    image_string = ""
    for idx, (n_h, n_w) in enumerate(splitted_image_counts):
        # append image index token if there are multiple images
        if len(splitted_image_counts) > 1:
            image_string += f"<image: {idx}>"
        # append global token
        if hasattr(tokenizer, "global_image_token"):
            image_string += tokenizer.global_image_token
            image_string += tokenizer.image_token * image_token_length
            if n_h == 1 and n_w == 1:
                continue

        for i in range(n_h):
            for j in range(n_w):
                image_string += getattr(tokenizer, f'r{i+1}c{j+1}') # append spacial (row and column) tokens
                image_string += tokenizer.image_token * image_token_length # append visual embedding tokens
    return image_string