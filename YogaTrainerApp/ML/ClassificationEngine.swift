import Vision
import CoreML

class ClassificationEngine {

    private var model: VNCoreMLModel?

    init() {
        if let coreMLModel = try? best(configuration: MLModelConfiguration()).model {
            model = try? VNCoreMLModel(for: coreMLModel)
        }
    }

    func classify(buffer: CVPixelBuffer,
                  completion: @escaping (String, Double) -> Void) {

        guard let model = model else { return }

        let request = VNCoreMLRequest(model: model) { request, _ in
            if let results = request.results as? [VNClassificationObservation],
               let first = results.first {

                completion(first.identifier, Double(first.confidence))
            }
        }

        let handler = VNImageRequestHandler(cvPixelBuffer: buffer)
        try? handler.perform([request])
    }
}