import Vision
import CoreML

class ClassificationEngine {

    private var legacyModel: VNCoreMLModel?
    private let featureExtractor: PoseFeatureExtractor
    private let mirrorFeatureExtractor: PoseFeatureExtractor
    private let landmarkClassifier: PoseMLPClassifier?
    private let config: PoseFeatureConfig

    init(config: PoseFeatureConfig = .init()) {
        self.config = config
        featureExtractor = PoseFeatureExtractor(config: config)
        mirrorFeatureExtractor = PoseFeatureExtractor(config: config)

        if let profileURL = Bundle.main.url(forResource: "pose_mlp_profile", withExtension: "json") {
            landmarkClassifier = PoseMLPClassifier.fromJSON(url: profileURL)
        } else {
            landmarkClassifier = nil
        }

        if let coreMLModel = try? best(configuration: MLModelConfiguration()).model {
            legacyModel = try? VNCoreMLModel(for: coreMLModel)
        }
    }

    /// Primary path: classify from 3D landmarks + visibility.
    func classify(landmarks: [[Float]],
                  visibility: [Float]? = nil,
                  completion: @escaping (String, Double) -> Void) {

        guard let features = featureExtractor.extract(landmarks: landmarks, visibility: visibility) else {
            completion("unknown", 0)
            return
        }

        if let landmarkClassifier,
           let result = landmarkClassifier.predict(features: features) {
            if config.enableMirrorInference {
                let mirroredLandmarks = PoseFeatureExtractor.mirrorLandmarks(landmarks)
                if let mirroredFeatures = mirrorFeatureExtractor.extract(landmarks: mirroredLandmarks, visibility: visibility),
                   let mirroredResult = landmarkClassifier.predict(features: mirroredFeatures),
                   mirroredResult.confidence > result.confidence {
                    completion(mirroredResult.label, mirroredResult.confidence)
                    return
                }
            }
            completion(result.label, result.confidence)
            return
        }

        completion("unknown", 0)
    }

    /// Backward-compatible fallback for existing camera pipeline.
    func classify(buffer: CVPixelBuffer,
                  completion: @escaping (String, Double) -> Void) {

        guard let legacyModel else {
            completion("unknown", 0)
            return
        }

        let request = VNCoreMLRequest(model: legacyModel) { request, _ in
            if let results = request.results as? [VNClassificationObservation],
               let first = results.first {
                completion(first.identifier, Double(first.confidence))
            } else {
                completion("unknown", 0)
            }
        }

        let handler = VNImageRequestHandler(cvPixelBuffer: buffer)
        try? handler.perform([request])
    }
}
