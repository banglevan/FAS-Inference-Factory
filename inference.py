import onnxruntime as ot
import numpy as np

PATH_TO_CHECKPOINT_ONNX = 'checkpoints/fas-best.onnx'

class LivenessDetection():
    def __init__(self) -> None:
        # to onnx: PATH_TO_CHECKPOINT_ONNX
        self.path_to_model = PATH_TO_CHECKPOINT_ONNX
        self.sess = self.load_model()
        self.threshold = 0.5

    def load_model(self):
        sess_options = ot.SessionOptions()
        sess_options.graph_optimization_level = ot.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ot.InferenceSession(self.path_to_model, sess_options=sess_options)
        return sess
    
    def pre_processing(self, face_image):
        # check requirement of needed align face or not
        # face_image --> normalize
        # --> reshape ...
        return face_tensors

    def infer_execute(self, face_tensor) -> list:
        """
        face numpy tensor --> [?, 3, 256, 256]
        when use single input ? = 1
        convert from opencv reader [b, g, r]
        return:
            outputs: list of [[prob to 0 label, prob to 1 label],...?]
                     numpy array fp32
                     label prob range from 0. ~ 1.
        """
        assert face_tensor.shape == 4
        assert face_tensor.shape[2] == face_tensor.shape[3] == 256
        outputs, x = self.sess.run(None, {'input.1': face_tensor})
        return outputs
    
    def post_process(self, outputs):
        outputs = np.array(outputs)
        pred_softmax = np.exp(outputs, axis=0) / np.sum(np.exp(outputs, axis=0))
        pred_label = np.argmax(pred_softmax, axis=0)
        pred_confd = np.max(pred_softmax, axis=0)
        return pred_label, pred_confd
    
    def run_on_images(self, images: list):
        tensor = self.pre_processing()
        outputs = self.infer_execute(tensor)
        results = self.post_process(outputs)
        return results
        
if __name__ == '__main__':
    obj_test = LivenessDetection()