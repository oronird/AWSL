# from os import walk
import os
import cv2
from tqdm import tqdm
from stickman import image_pos_hands
from os import makedirs


def get_file_names_with_pattern(path):
  """
  path: path to folder where all files (assume only .jpeg files where original name lasts with number)
  return: list of full path to files matched patter
  """
  image_names = next(os.walk(path), (None, None, []))[2]
  image_names = [file_name for file_name in image_names if not (file_name[0]=='.')]
  image_names = [name for name in image_names
                if 'flipped' in name
                or 'shifted' in name
                or 'zoom' in name
                or 'rotated' in name
                or name[-6].isdigit()]
  return image_names


def stickman_copy_folder(dir, to_dir,
                         stop_list=list(),
                         stop_list_2=list(),
                         plotting_image_res=(224*2,224*2),
                         saving_image_res=(224,224),
                         ):
    """
    Copy dir -> to_dir
    """
    print(f'---> STICKMAN for {dir}')
    if not os.path.exists(to_dir):
        os.makedirs(to_dir)

    set_names = next(os.walk(dir))[1]
    for set_name in set_names:
        full_set_path = dir + set_name + '/'
        print(f' ---> {set_name}')
        if set_name in stop_list_2:
            print('  ', set_name, 'in stop list!')
            continue

        if not os.path.exists(to_dir + set_name + '/'):
            os.makedirs(to_dir + set_name + '/')

        label_names = next(os.walk(full_set_path))[1]
        print(full_set_path)
        print(label_names)
        for label_name in label_names:
            print(f'  --->  {set_name: <6}/ {label_name:<15}')
            if not (label_name in stop_list):
                if not os.path.exists(to_dir + set_name + '/' + label_name + '/'):
                    os.makedirs(to_dir + set_name + '/' + label_name + '/')
                full_label_path = full_set_path + label_name + '/'
                # print('   ',full_label_path)
                image_names = get_file_names_with_pattern(full_label_path)
                print('!!!')

                for image_name in tqdm(image_names):
                    # for image_name in image_names[:2]:
                    frame = cv2.imread(full_label_path + image_name)
                    # print(full_label_path + image_name)
                    frame_out = image_pos_hands(image=frame,
                                                min_detection_confidence=0.5,
                                                return_img=True,
                                                plot_result=False,
                                                output_image_res=plotting_image_res
                                                )
                    frame_out = cv2.resize(frame_out, saving_image_res, interpolation=cv2.INTER_AREA)
                    frame_out_path = to_dir + set_name + '/' + label_name + '/' + image_name
                    # plt.imshow(frame_out[:,:,::-1])
                    # plt.title(label_name)
                    # plt.show()
                    cv2.imwrite(frame_out_path, frame_out)
            else:
                print('    ', label_name, 'in stop list!')