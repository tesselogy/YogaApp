import Vision
import CoreML
import Foundation

class ClassificationEngine {

    private var model: VNCoreMLModel?
    private var lastLogTime: Date = .distantPast

    init() {
        model = Self.loadModel(named: "best")

        if model == nil {
            print("[ClassificationEngine] Model missing: best.mlmodelc was not found in app bundle")
        } else {
            print("[ClassificationEngine] Model loaded successfully")
        }
    }

    func classify(buffer: CVPixelBuffer,
                  regionOfInterest: CGRect? = nil,
                  completion: @escaping (String, Double) -> Void) {

        guard let model = model else {
            completion("model_missing", 0)
            return
        }

        performInference(buffer: buffer, model: model, roi: regionOfInterest) { label, confidence in
            if label == "unknown", regionOfInterest != nil {
                self.debugLog("ROI inference returned unknown, retrying full-frame")
                self.performInference(buffer: buffer, model: model, roi: nil, completion: completion)
            } else {
                completion(label, confidence)
            }
        }
    }

    private func performInference(buffer: CVPixelBuffer,
                                  model: VNCoreMLModel,
                                  roi: CGRect?,
                                  completion: @escaping (String, Double) -> Void) {
        let request = VNCoreMLRequest(model: model) { request, _ in
            guard let results = request.results, !results.isEmpty else {
                self.debugLog("Inference returned empty results")
                completion("unknown", 0)
                return
            }

            if let classes = results as? [VNClassificationObservation],
               let first = classes.first {
                completion(first.identifier, Double(first.confidence))
                return
            }

            if let objects = results as? [VNRecognizedObjectObservation],
               let bestObject = objects.max(by: { $0.confidence < $1.confidence }),
               let bestLabel = bestObject.labels.first {
                self.debugLog("Model output is object-detection style, using top label '\(bestLabel.identifier)'")
                completion(bestLabel.identifier, Double(bestLabel.confidence))
                return
            }

            let outputType = String(describing: type(of: results[0]))
            self.debugLog("Unsupported Vision output type: \(outputType)")
            completion("unknown", 0)
        }

        request.imageCropAndScaleOption = .scaleFit

        if let roi {
            request.regionOfInterest = roi
        }

        let handler = VNImageRequestHandler(cvPixelBuffer: buffer)
        do {
            try handler.perform([request])
        } catch {
            print("[ClassificationEngine] Inference failed: \(error.localizedDescription)")
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

    private func debugLog(_ message: String) {
        let now = Date()
        guard now.timeIntervalSince(lastLogTime) > 1 else { return }
        lastLogTime = now
        print("[ClassificationEngine] \(message)")
    }
}
