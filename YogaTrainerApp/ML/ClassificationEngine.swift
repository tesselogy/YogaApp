import Vision
import CoreML
import Foundation

class ClassificationEngine {

    private var model: VNCoreMLModel?

    init() {
        model = Self.loadModel(named: "best")

        if model == nil {
            print("[ClassificationEngine] Classifier model missing: best.mlmodelc was not found in app bundle")
        } else {
            print("[ClassificationEngine] Classifier model loaded successfully")
        }
    }

    func classify(buffer: CVPixelBuffer,
                  completion: @escaping (String, Double) -> Void) {

        guard let model = model else {
            completion("model_missing", 0)
            return
        }

        let request = VNCoreMLRequest(model: model) { request, _ in
            if let results = request.results as? [VNClassificationObservation],
               let first = results.first {

                completion(first.identifier, Double(first.confidence))
            } else {
                print("[ClassificationEngine] No classification result, returning 'unknown'")
                completion("unknown", 0)
            }
        }

        let handler = VNImageRequestHandler(cvPixelBuffer: buffer)
        do {
            try handler.perform([request])
        } catch {
            print("[ClassificationEngine] Classification failed: \(error.localizedDescription)")
            completion("unknown", 0)
        }
    }

    private static func loadModel(named name: String) -> VNCoreMLModel? {
        guard let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc"),
              let model = try? MLModel(contentsOf: url) else {
            return nil
        }

        return try? VNCoreMLModel(for: model)
    }
}
