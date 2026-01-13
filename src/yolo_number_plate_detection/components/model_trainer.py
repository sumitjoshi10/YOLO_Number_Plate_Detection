from box import ConfigBox

def train_model(pretrained_model: object, params: ConfigBox):
    train_model = pretrained_model.train(
        data=params.data,
        epochs=params.epochs,
        imgsz=params.img_size,
        batch=params.batch_size,
        device=params.device,
        project=params.project, 
        name=params.name,
        workers=params.workers,
        mosaic=params.mosaic,
        amp=params.amp,
        exist_ok=params.exist_ok
    )
    

    return train_model
    
    