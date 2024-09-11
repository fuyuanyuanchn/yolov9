
def UI_box(self, img, box, label='', color=(128, 128, 128), line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]

    try:
        if isinstance(box, np.ndarray):
            if box.ndim == 2 and box.shape[0] == 1:
                # If box is a 2D array with only one row, flatten it
                box = box.flatten()
            elif box.ndim > 2 or (box.ndim == 2 and box.shape[0] != 1):
                raise ValueError(f"Unexpected box shape: {box.shape}")

            c1 = tuple(map(int, box[:2]))
            c2 = tuple(map(int, box[2:4]))
        elif isinstance(box, (list, tuple)):
            c1 = (int(box[0]), int(box[1]))
            c2 = (int(box[2]), int(box[3]))
        elif isinstance(box, torch.Tensor):
            box = box.cpu().numpy()
            c1 = tuple(map(int, box[:2]))
            c2 = tuple(map(int, box[2:4]))
        else:
            raise TypeError(f"Unsupported box type: {type(box)}")

        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    except Exception as e:
        print(f"Error in UI_box: {e}")
        print(f"Box: {box}")

    return img