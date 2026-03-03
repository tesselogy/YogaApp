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
        detect(buffer: buffer, completion: completion)
    }

    private func detect(buffer: CVPixelBuffer,
                        completion: @escaping (VNDetectedObjectObservation?) -> Void) {

        guard let model = detectionModel else {
            debugLog("YOLO model unavailable")
            completion(nil)
            return
        }

        let request = VNCoreMLRequest(model: model) { request, _ in
            let results = request.results as? [VNRecognizedObjectObservation] ?? []

            let persons = results.filter {
                $0.labels.first?.identifier.lowercased() == "person"
            }

            guard let bestPerson = persons.max(by: { $0.confidence < $1.confidence }) else {
                self.debugLog("YOLO found no person")
                completion(nil)
                return
            }

            completion(VNDetectedObjectObservation(boundingBox: bestPerson.boundingBox))
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
