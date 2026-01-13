from box import ConfigBox

def run_inferencing(trained_model: object, params: ConfigBox, image_path: str):
    results = trained_model(
        source=image_path,
        # img_size=params.img_size,
        conf=float(params.confidence_threshold),
        # show=True,
        # iou=params.iou_threshold,
        # device=params.device,
        save=params.save,
        save_txt=params.save_txt,
        save_conf=params.save_conf,
        project=params.project,
        name=params.name,
        exist_ok=params.exist_ok
    )