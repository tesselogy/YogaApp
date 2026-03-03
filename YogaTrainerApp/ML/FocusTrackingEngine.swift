import Vision
import CoreML
import Foundation

class FocusTrackingEngine {

    private var detectionModel: VNCoreMLModel?
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
        // Always redetect on every frame to avoid tracker box shrinking/drifting to chest.
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

            let expanded = self.expandForFullBody(best.boundingBox)
            completion(VNDetectedObjectObservation(boundingBox: expanded))
        }

        request.imageCropAndScaleOption = .scaleFit

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
            let expanded = self.expandForFullBody(best.boundingBox)
            completion(VNDetectedObjectObservation(boundingBox: expanded))
        }

        let handler = VNImageRequestHandler(cvPixelBuffer: buffer)
        do {
            try handler.perform([request])
        } catch {
            debugLog("Vision human fallback failed: \(error.localizedDescription)")
            completion(nil)
        }
    }

    private func expandForFullBody(_ bbox: CGRect) -> CGRect {
        let widthScale: CGFloat = 1.5
        let heightScale: CGFloat = 2.4

        let expandedWidth = min(1, bbox.width * widthScale)
        let expandedHeight = min(1, bbox.height * heightScale)

        // Shift center slightly down to include legs when detector is torso-biased.
        let centerX = bbox.midX
        let centerY = bbox.midY - bbox.height * 0.2

        let x = max(0, min(1 - expandedWidth, centerX - expandedWidth / 2))
        let y = max(0, min(1 - expandedHeight, centerY - expandedHeight / 2))

        return CGRect(x: x, y: y, width: expandedWidth, height: expandedHeight)
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
