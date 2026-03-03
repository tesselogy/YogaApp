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
            debugLog("YOLO model loaded successfully (used only as fallback)")
        }
    }

    func process(buffer: CVPixelBuffer,
                 completion: @escaping (VNDetectedObjectObservation?) -> Void) {
        // Primary detector: Vision human rectangles gives more stable full-body box.
        detectHuman(buffer: buffer) { humanObservation in
            if let humanObservation {
                completion(humanObservation)
                return
            }

            self.debugLog("Human rectangles found no person, trying YOLO fallback")
            self.detectWithYOLOFallback(buffer: buffer, completion: completion)
        }
    }

    private func detectHuman(buffer: CVPixelBuffer,
                             completion: @escaping (VNDetectedObjectObservation?) -> Void) {
        let request = VNDetectHumanRectanglesRequest { request, _ in
            let humans = request.results as? [VNHumanObservation] ?? []

            guard let best = humans.max(by: {
                ($0.boundingBox.width * $0.boundingBox.height) <
                ($1.boundingBox.width * $1.boundingBox.height)
            }) else {
                completion(nil)
                return
            }

            let expanded = self.expandForFullBody(best.boundingBox)
            completion(VNDetectedObjectObservation(boundingBox: expanded))
        }

        let handler = VNImageRequestHandler(cvPixelBuffer: buffer)
        do {
            try handler.perform([request])
        } catch {
            debugLog("Human rectangles failed: \(error.localizedDescription)")
            completion(nil)
        }
    }

    private func detectWithYOLOFallback(buffer: CVPixelBuffer,
                                        completion: @escaping (VNDetectedObjectObservation?) -> Void) {
        guard let model = detectionModel else {
            completion(nil)
            return
        }

        let request = VNCoreMLRequest(model: model) { request, _ in
            let results = request.results as? [VNRecognizedObjectObservation] ?? []

            guard !results.isEmpty else {
                self.debugLog("YOLO fallback found no objects")
                completion(nil)
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
                completion(nil)
                return
            }

            if persons.isEmpty {
                self.debugLog("YOLO fallback: no explicit 'person' label, using largest object")
            }

            let expanded = self.expandForFullBody(best.boundingBox)
            completion(VNDetectedObjectObservation(boundingBox: expanded))
        }

        request.imageCropAndScaleOption = .scaleFit

        let handler = VNImageRequestHandler(cvPixelBuffer: buffer)
        do {
            try handler.perform([request])
        } catch {
            debugLog("YOLO fallback failed: \(error.localizedDescription)")
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
