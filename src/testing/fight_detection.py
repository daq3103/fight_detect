from models.model_3dcnn_r2plus1d import FightDetection3DCNN

class FightDetector():
    def __init__(self, model_weight_path = "model/best_weights.pt"):
        self.detector = FightDetection3DCNN()
        self.detector.load_state_dict(model_weight_path)