from model_setup import *
os.chdir("../")

def clean_unlabeled_video(video_name, bucket_location, extension, upload_frames=False):
    folder_name = video_name
    yaml_val = {'Video': f'{video_name}.{extension}'}
    frames_folder = f'{video_name}_frames'
    split_video_frames(video_name, extension, folder_name, frames_folder)
    yaml_val.update(remove_blurry_images(frames_folder, 'png'))
    yaml_val.update(remove_duplicates(frames_folder, 'png'))
    yaml_val.update(detect_file(frames_folder, 'png'))
    remaining_frames = sorted(glob.glob(f'{frames_folder}/*.{extension}'))
    yaml_val.update({'Clean Frames': remaining_frames})
    f = open(f'{video_name}.yaml', "w")
    yaml.dump(yaml_val, f, default_flow_style=False)
    f.close()




def kitti_label_check(image_dir, label_dir, image_extension, label_extension):
    detect_list = []
    for image_file in sorted(glob.glob(f'{image_dir}/*.{image_extension}')):
        print(f'Detecting on Image: {image_file}', end='\r')
        output = {'Frame': image_file}
        detections = detect_objects(model, image_file)

        fol_len = len(image_dir)
        ext_len = len(image_extension) + 1
        label_file = f'{label_dir}{image_file[fol_len:-ext_len]}.{label_extension}'
        print(f'Reading GT File: {label_file}              ', end='\r')

        label = pd.read_csv(label_file, delim_whitespace=True, header=None)
        label.columns = ['Object', 'Trunc', 'Occ', 'Alpha', 'min_x', 'min_y', 'max_x', 'max_y', 'dim1', 'dim2', 'dim3',
                         'loc1', 'loc2', 'loc3', 'rot_y']

        label = label.drop(columns=['Alpha', 'dim1', 'dim2', 'dim3', 'loc1', 'loc2', 'loc3', 'rot_y'])

        gt_detections = label['Object'].tolist()
        gt_output = Counter(gt_detections)

        gt_detection_difference = detections['Objects Detected'] - gt_output
        gt_total_labels = sum(gt_output.values())

        ground_truth = {'GT Objects': dict(gt_output),
                        'Number of GT Labels': gt_total_labels,
                        'Difference between Detections and GT': dict(gt_detection_difference)}

        detections['Objects Detected'] = dict(detections['Objects Detected'])

        output.update(detections)
        output.update(ground_truth)

        detect_list.append(output)


    print('Label Check Completed')
    return {'Classification Information': detect_list}


def model_output(image_dir, label_file, image_extension, label_extension):
    model = load_model()


def kitti_sequence_label_check(image_dir, label_file, image_extension, label_extension):
    detect_list = []
    # Import Label File
    label_file = f'{label_file}.{label_extension}'
    print(f'Reading GT File: {label_file}              ', end='\r')
    labels = pd.read_csv(label_file, delim_whitespace=True, header=None)
    labels.columns = ['Frame', 'Track_ID', 'Object', 'Trunc', 'Occ', 'Alpha', 'min_x', 'min_y', 'max_x', 'max_y',
                      'dim1', 'dim2', 'dim3', 'loc1', 'loc2', 'loc3', 'rot_y']
    labels = labels.drop(columns=['Track_ID', 'Alpha', 'dim1', 'dim2', 'dim3', 'loc1', 'loc2', 'loc3', 'rot_y'])

    for image_file in sorted(glob.glob(f'{image_dir}/*.{image_extension}')):
        print(f'Detecting on Image: {image_file}', end='\r')
        output = {'Frame': image_file}
        detections = detect_objects(image_file)

        frame_num = int(re.search(r'\d+', os.path.basename(image_file)).group(0))
        frame_label = labels.loc[labels['Frame'] == frame_num]

        gt_detections = frame_label['Object'].tolist()
        gt_output = Counter(gt_detections)

        vehicles = gt_output['Car'] + gt_output['Van'] + gt_output['Truck']
        people = gt_output['Pedestrian'] + gt_output['Person_sitting'] + gt_output['Cyclist']
        urban_vehicles = gt_output['Tram']



        gt_detection_difference = detections['Objects Detected'] - gt_output
        gt_total_labels = sum(gt_output.values())

        ground_truth = {'GT Objects': dict(gt_output),
                        'Number of GT Labels': gt_total_labels,
                        'Difference between Detections and GT': dict(gt_detection_difference)}

        detections['Objects Detected'] = dict(detections['Objects Detected'])

        output.update(detections)
        output.update(ground_truth)

        detect_list.append(output)

    # sequence_classification = Counter(output['GT Classification']).most_common(1)

    print('Label Check Completed')
    return {'Classification Information': detect_list}




def main(project_folder, image_folder, image_extension, label_folder, label_extension, dataset_format):
    current_formats = ['KITTI', 'OCT', 'JAAD', 'KITTI_sequence']  # Current List of supported dataset formats
    if dataset_format not in current_formats:  # Check to make sure dataset_format is supported
        print(f'dataset_format dataset format not supported. Cancelling')
        return False
    yaml_val = {'Image Folder': f'{project_folder}/{image_folder}', 'Label Folder': f'{project_folder}/{label_folder}'}
    image_dir = f'{project_folder}/{image_folder}'
    label_dir = f'{project_folder}/{label_folder}'
    yaml_val.update(remove_blurry_images(image_dir, image_extension))
    yaml_val.update(remove_duplicates(image_dir, image_extension))
    if dataset_format == 'KITTI':
        yaml_val.update(kitti_label_check(image_dir, label_dir, image_extension, label_extension))
    elif dataset_format == 'KITTI_sequence':
        yaml_val.update(kitti_sequence_label_check(image_dir, label_dir, image_extension, label_extension))
    remaining_frames = sorted(glob.glob(f'{image_dir}/*.{image_extension}'))
    yaml_val.update({'Clean Frames': remaining_frames})

    image_folder_unslashed = image_folder.replace('/', '-')
    yaml_dir = f'{project_folder}_{image_folder_unslashed}.yaml'
    f = open(yaml_dir, "w")
    yaml.dump(yaml_val, f, default_flow_style=False)
    f.close()
    print(f'YAML File: {yaml_dir} created')
    print('Process Complete')
    return True


if __name__ == "__main__":
    main('kitti_data', 'images', 'png', 'labels', 'txt', 'KITTI')



