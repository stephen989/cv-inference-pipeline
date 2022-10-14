from model_setup import *
from tqdm import tqdm


def image_pipeline(opts):

    image_dir = opts.image_dir
    image_ext = opts.image_ext
    image_output_dir = opts.image_output_dir
    output_yaml = opts.yaml_output
    model = opts.model
    weights = opts.weights
    model_classes_file = opts.model_classes

    os.makedirs(image_output_dir, exist_ok=True)


    # feed through model one by one
    frames, outputs, image_names = image_preprocessing(image_dir, image_ext)
    outputs_dict = {"Directory": image_dir,
                    "Extension": image_ext,
                    "Output directory": image_output_dir,
                    "Model": model,
                    "Preprocessing output": outputs,
                    "Model Outputs": dict()}

    model = Model(model, weights, opts.model_version, model_classes_file)
    outputs_dict["Model classes"] = model.classes
    print("Feeding model")
    for image_name, frame in tqdm(zip(image_names, frames)):
        output = model(frame)
        outputs_dict["Model Outputs"][image_name] = output
    print("Complete")
    # write to yaml file
    with open(output_yaml, 'w') as y:
        yaml.dump(outputs_dict, y)

    create_output_images(output_yaml, frames,  image_output_dir)


def video_pipeline(opts):
    video = opts.video_input
    output_video = opts.video_output
    output_yaml = opts.yaml_output
    model = opts.model
    weights = opts.weights
    model_classes = opts.model_classes
    frames, outputs = video_prepocessing(video)
    outputs_dict = {"File": video,
                    "Output Video": output_video,
                    "Model": model,
                    "Preprocessing output": outputs,
                    "Model Outputs": dict()}

    model = Model(model, weights, opts.model_version, model_classes)
    outputs_dict["Model classes"] = model.classes

    # feed through model one by one
    print("Feeding model")
    for i, frame in enumerate(tqdm(frames)):
        output = model(frame)
        outputs_dict["Model Outputs"][i] = output
    print("Complete")
    # write to yaml file
    with open(output_yaml, 'w') as y:
        yaml.dump(outputs_dict, y)
    create_output_video(output_yaml, frames, output_video, video)






