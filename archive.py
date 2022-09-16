# !pip install --upgrade nest_asyncio fastapi ffmpeg google-cloud-storage uvicorn python-multipart tensorflow-gpu scikit-image imutils wandb tensorflow_hub Pillow pyyaml


def clean_images_labeled(project_folder, image_folder, image_extension, label_folder, label_extension, dataset_format,
                         upload_frames=False):
    current_formats = ['KITTI', 'OCT', 'JAAD', 'KITTI_sequence']  # Current List of supported dataset formats
    if dataset_format not in current_formats:  # Check to make sure dataset_format is supported
        print(f'dataset_format dataset format not supported. Cancelling')
        return False
    yaml_val = {'Image Folder': f'{project_folder}/{image_folder}', 'Label Folder': f'{project_folder}/{label_folder}'}
    # download_images_and_labels(project_folder, image_folder, label_folder)
    image_dir = f'{project_folder}/{image_folder}'
    label_dir = f'{project_folder}/{label_folder}'
    yaml_val.update(remove_blurry_images(image_dir, image_extension))
    yaml_val.update(remove_duplicates(image_dir, image_extension))
    if dataset_format == 'JAAD':
        yaml_val.update(detect_file(image_dir, image_extension))
    elif dataset_format == 'KITTI':
        yaml_val.update(kitti_label_check(image_dir, label_dir, image_extension, label_extension))
    elif dataset_format == 'OCT':
        yaml_val.update(oct_label_check(image_dir, label_dir, image_extension, label_extension))
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
    # Upload yaml
    # blob = bucket.blob(yaml_dir)
    # blob.upload_from_filename(yaml_dir)
    # print(f'YAML File: {yaml_dir} uploaded to gs://{project_folder}/{yaml_dir}')
    # if upload_frames == True:
    #     upload_frames(project_folder, extension)
    # shutil.rmtree(project_folder)
    print('Process Complete')
    return True

def clean_images(folder_name, extension, upload_frames=False):
    folder_name = str(folder_name)
    yaml_val = {'Folder': folder_name}
    # download_images(folder_name)
    image_folder = f'{folder_name}/images'
    yaml_val.update(remove_blurry_images(image_folder, extension))
    yaml_val.update(remove_duplicates(image_folder, extension))
    yaml_val.update(detect_file(image_folder, extension))
    f = open(f'{folder_name}.yaml', "w")
    yaml.dump(yaml_val, f, default_flow_style=False)
    f.close()
    print(f'YAML File: {folder_name}.yaml created')
    # Upload yaml
    # blob = bucket.blob(f'{folder_name}.yaml')
    # blob.upload_from_filename(f'{folder_name}.yaml')
    # print(f'YAML File: {folder_name}.yaml uploaded to gs://{location}/{folder_name}.yaml')
    # if upload_frames == True:
    #     upload_frames(folder_name, extension)
    # shutil.rmtree(folder_name)


def download_images_and_labels(project_folder, image_folder, label_folder):
    print(f'Downloading project folder: {project_folder}.                       ')
    print("Checking if folder already downloaded", end="\r")
    img_dir = f'{project_folder}/{image_folder}'
    label_dir = f'{project_folder}/{label_folder}'
    download_images = True
    download_labels = True
    if download_images == True:
        os.makedirs(img_dir, exist_ok=True)
        print(f'Downloading Image Folder: {img_dir}                       ')
        blobs = storage_client.list_blobs(location, prefix=img_dir)
        for blob in blobs:
            print(f'Downloading: {blob.name}                       ', end='\r')
            try:
                blob.download_to_filename(blob.name)
            except:
                print(f'{blob.name} failed to download, skipping')
    if download_labels == True:
        os.makedirs(label_dir, exist_ok=True)
        print(f'Downloading Label Folder: {label_dir}                       ')
        blobs = storage_client.list_blobs(location, prefix=label_dir)
        for blob in blobs:
            print(f'Downloading: {blob.name}                       ', end='\r')
            try:
                blob.download_to_filename(blob.name)
            except:
                print(f'{blob.name} failed to download, skipping')
    print(f'Downloading project folder: {project_folder} complete.                       ')


def download_video(video_name, extension, origin_folder, dest_folder):
    print("Checking if video already downloaded", end="\r")
    vid_dir = f'{dest_folder}/{video_name}.{extension}'
    if os.path.exists(vid_dir):
        print(f'Video: {video_name}.{extension} already downloaded. Skipping Download.')
        return
    try:
        os.mkdir(str(dest_folder))
    except:
        print(f'Folder: {dest_folder} already exists.          ')
    def_location = f'{origin_folder}/{video_name}.{extension}'
    print(f'Downloading: {def_location}')
    blob = bucket.blob(def_location)
    def_destination = f'{dest_folder}/{video_name}.{extension}'
    blob.download_to_filename(def_destination)
    print('Download Complete')

def upload_frames(folder_name, extension):
    files = sorted(glob.glob(f'{folder_name}/*.{extension}'))
    # files=files[1:]

    print("Uploading Frames")
    for i in range(len(files)):
        print(files[i] + "             ", end="\r")
        blob = bucket.blob(folder_name + "/" + files[i])
        blob.upload_from_filename(folder_name + "/" + files[i])

    print("Done Uploading               ", end="\r")

def parallel_compare_images(i, files):
    image1 = cv2.imread(files[i])
    image2 = cv2.imread(files[i + 1])
    image_gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image_gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    try:
        diff, _ = compare_ssim(image_gray1, image_gray2, full=True)
    except:
        image_gray2 = cv2.resize(image_gray2, (image_gray1.shape[1], image_gray1.shape[0]),
                                 interpolation=cv2.INTER_AREA)
        diff, _ = compare_ssim(image_gray1, image_gray2, full=True)
    print(f'Similarity between {files[i]} and {files[i + 1]}: {diff}', end="\r")
    return diff