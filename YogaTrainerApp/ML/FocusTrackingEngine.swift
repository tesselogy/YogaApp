import Vision
import CoreML

class FocusTrackingEngine {

    private var detectionModel: VNCoreMLModel?
    private var trackingRequest: VNTrackObjectRequest?

    init() {
        detectionModel = Self.loadModel(named: "yolov8n")
    }

    func process(buffer: CVPixelBuffer,
                 completion: @escaping (VNDetectedObjectObservation?) -> Void) {

        if let trackingRequest = trackingRequest {
            let handler = VNImageRequestHandler(cvPixelBuffer: buffer)
            try? handler.perform([trackingRequest])

            if let result = trackingRequest.results?.first as? VNDetectedObjectObservation {
                completion(result)
                return
            }
        }

        detect(buffer: buffer, completion: completion)
    }

    private func detect(buffer: CVPixelBuffer,
                        completion: @escaping (VNDetectedObjectObservation?) -> Void) {

        guard let model = detectionModel else {
            completion(nil)
            return
        }

        let request = VNCoreMLRequest(model: model) { request, _ in
            let results = request.results as? [VNRecognizedObjectObservation] ?? []

            let persons = results.filter {
                $0.labels.first?.identifier == "person"
            }

            guard let best = persons.max(by: {
                ($0.boundingBox.width * $0.boundingBox.height) <
                ($1.boundingBox.width * $1.boundingBox.height)
            }) else {
                completion(nil)
                return
            }

            let tracking = VNTrackObjectRequest(detectedObjectObservation: best)
            tracking.trackingLevel = .accurate

            self.trackingRequest = tracking
            completion(best)
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
