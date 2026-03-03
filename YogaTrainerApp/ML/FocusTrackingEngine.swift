import Vision
import CoreML
import Foundation

class FocusTrackingEngine {

    private var detectionModel: VNCoreMLModel?
    private var trackingRequest: VNTrackObjectRequest?
    private var lastLogTime: Date = .distantPast
    private var frameCounter = 0
    private let redetectInterval = 6

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

        frameCounter += 1
        let shouldRedetect = frameCounter % redetectInterval == 0

        if shouldRedetect {
            trackingRequest = nil
            debugLog("Periodic re-detection to avoid box shrink/drift")
        }

        if let trackingRequest = trackingRequest {
            let handler = VNImageRequestHandler(cvPixelBuffer: buffer)
            do {
                try handler.perform([trackingRequest])
            } catch {
                debugLog("Tracking request failed: \(error.localizedDescription)")
            }

            if let result = trackingRequest.results?.first as? VNDetectedObjectObservation,
               result.confidence > 0.25,
               (result.boundingBox.width * result.boundingBox.height) > 0.02 {
                debugLog("Tracking person with confidence: \(String(format: "%.2f", result.confidence))")
                completion(result)
                return
            } else {
                debugLog("Tracking lost/too small target, running detection again")
                self.trackingRequest = nil
            }
        }

        detect(buffer: buffer, completion: completion)
    }

    private func detect(buffer: CVPixelBuffer,
                        completion: @escaping (VNDetectedObjectObservation?) -> Void) {

        guard let model = detectionModel else {
            detectHumanFallback(buffer: buffer, completion: completion)
            return
        }

        let request = VNCoreMLRequest(model: model) { request, _ in
            let results = request.results as? [VNRecognizedObjectObservation] ?? []

            if results.isEmpty {
                self.debugLog("Detection returned 0 objects, trying Vision human fallback")
                self.detectHumanFallback(buffer: buffer, completion: completion)
                return
            }

            let persons = results.filter {
                $0.labels.first?.identifier.lowercased() == "person"
            }

            let candidates = persons.isEmpty ? results : persons

            guard let best = candidates.max(by: {
                ($0.boundingBox.width * $0.boundingBox.height) <
                ($1.boundingBox.width * $1.boundingBox.height)
            }) else {
                self.detectHumanFallback(buffer: buffer, completion: completion)
                return
            }

            if persons.isEmpty {
                self.debugLog("No explicit 'person' label, using largest detected object")
            }

            self.startTracking(with: best)
            completion(best)
        }

        request.imageCropAndScaleOption = .scaleFill

        let handler = VNImageRequestHandler(cvPixelBuffer: buffer)
        do {
            try handler.perform([request])
        } catch {
            debugLog("Detection request failed: \(error.localizedDescription), trying Vision human fallback")
            detectHumanFallback(buffer: buffer, completion: completion)
        }
    }

    private func detectHumanFallback(buffer: CVPixelBuffer,
                                     completion: @escaping (VNDetectedObjectObservation?) -> Void) {
        let request = VNDetectHumanRectanglesRequest { request, _ in
            let humans = request.results as? [VNHumanObservation] ?? []

            guard let best = humans.max(by: {
                ($0.boundingBox.width * $0.boundingBox.height) <
                ($1.boundingBox.width * $1.boundingBox.height)
            }) else {
                self.debugLog("Vision human fallback also found no person")
                completion(nil)
                return
            }

            self.debugLog("Vision human fallback detected person")
            let observation = VNDetectedObjectObservation(boundingBox: best.boundingBox)
            self.startTracking(with: observation)
            completion(observation)
        }

        let handler = VNImageRequestHandler(cvPixelBuffer: buffer)
        do {
            try handler.perform([request])
        } catch {
            debugLog("Vision human fallback failed: \(error.localizedDescription)")
            completion(nil)
        }
    }

    private func startTracking(with observation: VNDetectedObjectObservation) {
        let tracking = VNTrackObjectRequest(detectedObjectObservation: observation)
        tracking.trackingLevel = .accurate
        self.trackingRequest = tracking
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
