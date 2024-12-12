from src.sam.segment_anything.build_sam import *

def sam_model(model_size, checkpoint):
    return sam_model_registry[model_size](checkpoint=checkpoint)