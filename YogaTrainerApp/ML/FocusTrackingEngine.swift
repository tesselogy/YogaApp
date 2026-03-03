import Vision
import CoreML
import Foundation

class FocusTrackingEngine {

    private var detectionModel: VNCoreMLModel?
    private var trackingRequest: VNTrackObjectRequest?
    private var lastLogTime: Date = .distantPast

    init() {
        detectionModel = Self.loadModel(named: "yolov8n")

        if detectionModel == nil {
            debugLog("YOLO model missing: yolov8n.mlmodelc was not found in app bundle")
        } else {
            debugLog("YOLO model loaded successfully")
        }
    }

    func process(buffer: CVPixelBuffer,
                 completion: @escaping (VNDetectedObjectObservation?) -> Void) {

        if let trackingRequest = trackingRequest {
            let handler = VNImageRequestHandler(cvPixelBuffer: buffer)
            do {
                try handler.perform([trackingRequest])
            } catch {
                debugLog("Tracking request failed: \(error.localizedDescription)")
            }

            if let result = trackingRequest.results?.first as? VNDetectedObjectObservation {
                debugLog("Tracking person with confidence: \(String(format: "%.2f", result.confidence))")
                completion(result)
                return
            } else {
                debugLog("Tracking lost target, running detection again")
                self.trackingRequest = nil
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

            if results.isEmpty {
                self.debugLog("Detection returned 0 objects")
                completion(nil)
                return
            }

            let topLabels = results.prefix(3).compactMap { $0.labels.first?.identifier }.joined(separator: ",")
            self.debugLog("Detection objects: \(results.count), top labels: [\(topLabels)]")

            let persons = results.filter {
                $0.labels.first?.identifier.lowercased() == "person"
            }

            let candidates = persons.isEmpty ? results : persons

            guard let best = candidates.max(by: {
                ($0.boundingBox.width * $0.boundingBox.height) <
                ($1.boundingBox.width * $1.boundingBox.height)
            }) else {
                completion(nil)
                return
            }

            if persons.isEmpty {
                self.debugLog("No explicit 'person' label, using largest detected object")
            }

            let tracking = VNTrackObjectRequest(detectedObjectObservation: best)
            tracking.trackingLevel = .accurate

            self.trackingRequest = tracking
            completion(best)
        }

        request.imageCropAndScaleOption = .scaleFill

        let handler = VNImageRequestHandler(cvPixelBuffer: buffer)
        do {
            try handler.perform([request])
        } catch {
            debugLog("Detection request failed: \(error.localizedDescription)")
            completion(nil)
        }
    }

    private static func loadModel(named name: String) -> VNCoreMLModel? {
        guard let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc"),
              let model = try? MLModel(contentsOf: url) else {
            return nil
        }

        return try? VNCoreMLModel(for: model)
    }

    private func debugLog(_ message: String) {
        let now = Date()
        guard now.timeIntervalSince(lastLogTime) > 1 else { return }
        lastLogTime = now
        print("[FocusTrackingEngine] \(message)")
    }
}
