import cv2

def combine_masks(mask_files):
    combined_mask = None
    for mask_file in mask_files:
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        if combined_mask is None:
            combined_mask = mask
        else:
            combined_mask = cv2.bitwise_or(combined_mask, mask)
    return combined_mask

# usage
# mask_files = ["object1_mask.png", "object2_mask.png", "object3_mask.png"]
mask_files = ["1_mask.png", "3_mask.png", "4_mask.png", "5_mask.png", "7_mask.png", "15_mask.png", "17_mask.png", "21_mask.png", "22_mask.png", "24_mask.png", "26_mask.png", "27_mask.png"]
combined_mask = combine_masks(mask_files)
cv2.imwrite("combined_mask.png", combined_mask)
