import cv2


def frames_for_video(video_path, resize_condition=True, output_size = (224*2,224*2)):
    """
    video_path: path to video
    return: list of video frames resized to `output_size`
    """
    result = []
    src = cv2.VideoCapture(str(video_path))
    video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)
    # print(f"Video length: {video_length}")
    start = 0
    src.set(cv2.CAP_PROP_POS_FRAMES, start)
    # ret is a boolean indicating whether read was successful, frame is the image itself
    success, frame = src.read()
    while success:
        if resize_condition:
            result.append(cv2.resize(frame, output_size))
        else:
            result.append(frame)
        success, frame = src.read()
    src.release()
    return result
