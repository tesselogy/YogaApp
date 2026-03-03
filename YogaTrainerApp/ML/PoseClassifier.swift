import CoreML
import Foundation
import CoreVideo

struct ClassificationResult {
    let label: String
    let confidence: Float
}

final class PoseClassifier {
    private let model: MLModel
    private let inputName: String
    private let classLabels: [String]
    private let confidenceThreshold: Float

    init?(modelName: String = "best", confidenceThreshold: Float = 0.5) {
        guard let url = Bundle.main.url(forResource: modelName, withExtension: "mlmodelc"),
              let loadedModel = try? MLModel(contentsOf: url) else {
            return nil
        }

        guard let imageInput = loadedModel.modelDescription.inputDescriptionsByName.first(where: { $0.value.type == .image })?.key else {
            return nil
        }

        self.model = loadedModel
        self.inputName = imageInput
        self.confidenceThreshold = confidenceThreshold

        if let labels = loadedModel.modelDescription.classLabels as? [String] {
            self.classLabels = labels
        } else {
            self.classLabels = []
        }
    }

    func classify(cropBuffer: CVPixelBuffer) -> ClassificationResult {
        guard let provider = try? MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(pixelBuffer: cropBuffer)]),
              let prediction = try? model.prediction(from: provider) else {
            return ClassificationResult(label: "uncertain", confidence: 0)
        }

        if let probs = prediction.featureNames
            .compactMap({ prediction.featureValue(for: $0)?.dictionaryValue as? [String: NSNumber] })
            .first,
           let best = probs.max(by: { $0.value.floatValue < $1.value.floatValue }) {
            let conf = best.value.floatValue
            let label = conf >= confidenceThreshold ? best.key : "uncertain"
            return ClassificationResult(label: label, confidence: conf)
        }

        if let multi = prediction.featureNames.compactMap({ prediction.featureValue(for: $0)?.multiArrayValue }).first,
           multi.count > 0 {
            var bestIndex = 0
            var bestValue: Float = -Float.greatestFiniteMagnitude
            for i in 0..<multi.count {
                let v = multi[i].floatValue
                if v > bestValue {
                    bestValue = v
                    bestIndex = i
                }
            }

            let confidence = bestValue
            let rawLabel = bestIndex < classLabels.count ? classLabels[bestIndex] : "class_\(bestIndex)"
            let label = confidence >= confidenceThreshold ? rawLabel : "uncertain"
            return ClassificationResult(label: label, confidence: confidence)
        }

        return ClassificationResult(label: "uncertain", confidence: 0)
    }
}
