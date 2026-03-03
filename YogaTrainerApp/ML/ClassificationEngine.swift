import Vision
import CoreML

class ClassificationEngine {

    private var model: VNCoreMLModel?

    init() {
        model = Self.loadModel(named: "best")
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
                completion("unknown", 0)
            }
        }

        let handler = VNImageRequestHandler(cvPixelBuffer: buffer)
        try? handler.perform([request])
    }

    private static func loadModel(named name: String) -> VNCoreMLModel? {
        guard let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc"),
              let model = try? MLModel(contentsOf: url) else {
            return nil
        }

        return try? VNCoreMLModel(for: model)
    }
}
