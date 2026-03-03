import Vision
import CoreML
import Foundation

class ClassificationEngine {

    private var model: VNCoreMLModel?
    private var classLabels: [String] = []
    private var lastLogTime: Date = .distantPast

    init() {
        let loaded = Self.loadModel(named: "best")
        model = loaded.vnModel
        classLabels = loaded.classLabels

        if model == nil {
            print("[ClassificationEngine] Model missing: best.mlmodelc was not found in app bundle")
        } else {
            print("[ClassificationEngine] Model loaded successfully")
            if !classLabels.isEmpty {
                debugLog("Loaded \(classLabels.count) class labels from model metadata")
            }
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

            if let featureObs = results.first as? VNCoreMLFeatureValueObservation,
               let resolved = self.resolveFeatureValueObservation(featureObs) {
                completion(resolved.label, resolved.confidence)
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

    private func resolveFeatureValueObservation(_ observation: VNCoreMLFeatureValueObservation) -> (label: String, confidence: Double)? {
        let value = observation.featureValue

        if let multi = value.multiArrayValue,
           let (index, confidence) = Self.resolveBestClassIndex(from: multi, labelsCount: classLabels.count) {
            let label = index < classLabels.count ? classLabels[index] : "class_\(index)"
            debugLog("FeatureValue(MultiArray) resolved as \(label) @ \(String(format: "%.2f", confidence))")
            return (label, confidence)
        }

        if let dict = value.dictionaryValue as? [AnyHashable: NSNumber],
           let best = dict.max(by: { $0.value.doubleValue < $1.value.doubleValue }) {
            let rawLabel = String(describing: best.key)
            let label = Self.normalizeLabel(rawLabel, classLabels: classLabels)
            let confidence = best.value.doubleValue
            debugLog("FeatureValue(Dictionary) resolved as \(label) @ \(String(format: "%.2f", confidence))")
            return (label, confidence)
        }

        debugLog("FeatureValue type not supported: \(value.type)")
        return nil
    }

    private static func resolveBestClassIndex(from multi: MLMultiArray,
                                              labelsCount: Int) -> (index: Int, confidence: Double)? {
        let count = multi.count
        guard count > 0 else { return nil }

        // Standard classifier head: vector with one score per class.
        if labelsCount > 0, count == labelsCount {
            return argmax(from: multi)
        }

        // Detection-like tensor flattened: project into class space using max over anchors/cells.
        if labelsCount > 0, count > labelsCount {
            let byModulo = maxByModulo(multi: multi, labelsCount: labelsCount)
            let byBlocks = maxByBlocks(multi: multi, labelsCount: labelsCount)

            if let byModulo, let byBlocks {
                return byModulo.confidence >= byBlocks.confidence ? byModulo : byBlocks
            }
            return byModulo ?? byBlocks
        }

        // No labels metadata: avoid returning meaningless large tensor index.
        if count > 512 {
            return nil
        }

        return argmax(from: multi)
    }

    private static func maxByModulo(multi: MLMultiArray,
                                    labelsCount: Int) -> (index: Int, confidence: Double)? {
        guard labelsCount > 0 else { return nil }

        var bestIndex = 0
        var bestValue = -Double.greatestFiniteMagnitude

        for classIndex in 0 ..< labelsCount {
            var classBest = -Double.greatestFiniteMagnitude
            var i = classIndex
            while i < multi.count {
                let value = multi[i].doubleValue
                if value > classBest { classBest = value }
                i += labelsCount
            }

            if classBest > bestValue {
                bestValue = classBest
                bestIndex = classIndex
            }
        }

        guard bestValue > -Double.greatestFiniteMagnitude else { return nil }
        return (bestIndex, bestValue)
    }

    private static func maxByBlocks(multi: MLMultiArray,
                                    labelsCount: Int) -> (index: Int, confidence: Double)? {
        guard labelsCount > 0 else { return nil }

        let blockSize = multi.count / labelsCount
        guard blockSize > 0 else { return nil }

        var bestIndex = 0
        var bestValue = -Double.greatestFiniteMagnitude

        for classIndex in 0 ..< labelsCount {
            let start = classIndex * blockSize
            let end = min(start + blockSize, multi.count)
            guard start < end else { continue }

            var classBest = -Double.greatestFiniteMagnitude
            for i in start ..< end {
                let value = multi[i].doubleValue
                if value > classBest { classBest = value }
            }

            if classBest > bestValue {
                bestValue = classBest
                bestIndex = classIndex
            }
        }

        guard bestValue > -Double.greatestFiniteMagnitude else { return nil }
        return (bestIndex, bestValue)
    }

    private static func argmax(from multi: MLMultiArray) -> (index: Int, confidence: Double)? {
        let count = multi.count
        guard count > 0 else { return nil }

        var bestIndex = 0
        var bestValue = -Double.greatestFiniteMagnitude

        for i in 0 ..< count {
            let value = multi[i].doubleValue
            if value > bestValue {
                bestValue = value
                bestIndex = i
            }
        }

        return (bestIndex, bestValue)
    }

    private static func normalizeLabel(_ rawLabel: String, classLabels: [String]) -> String {
        if let index = Int(rawLabel), index >= 0, index < classLabels.count {
            return classLabels[index]
        }
        return rawLabel
    }

    private static func loadModel(named name: String) -> (vnModel: VNCoreMLModel?, classLabels: [String]) {
        guard let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc"),
              let rawModel = try? MLModel(contentsOf: url),
              let vnModel = try? VNCoreMLModel(for: rawModel) else {
            return (nil, [])
        }

        let labels = Self.extractClassLabels(from: rawModel)
        return (vnModel, labels)
    }

    private static func extractClassLabels(from model: MLModel) -> [String] {
        if let labels = model.modelDescription.classLabels {
            if let stringLabels = labels as? [String] {
                return stringLabels
            }

            if let intLabels = labels as? [Int] {
                return intLabels.map(String.init)
            }
        }

        // Some converted models keep class names in user-defined metadata.
        let meta = model.modelDescription.metadata[.creatorDefinedKey] as? [String: String] ?? [:]

        if let json = meta["classes"] ?? meta["class_labels"] ?? meta["names"] {
            if let data = json.data(using: .utf8) {
                if let arr = try? JSONSerialization.jsonObject(with: data) as? [String] {
                    return arr
                }

                if let dict = try? JSONSerialization.jsonObject(with: data) as? [String: String] {
                    return dict.keys
                        .compactMap(Int.init)
                        .sorted()
                        .compactMap { dict[String($0)] }
                }
            }

            let csv = json.split(separator: ",").map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            if !csv.isEmpty {
                return csv
            }
        }

        return []
    }

    private func debugLog(_ message: String) {
        let now = Date()
        guard now.timeIntervalSince(lastLogTime) > 1 else { return }
        lastLogTime = now
        print("[ClassificationEngine] \(message)")
    }
}
