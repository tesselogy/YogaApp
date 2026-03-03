import CoreML
import Foundation
import CoreVideo

struct Detection {
    let bbox: CGRect // pixel xyxy stored as CGRect(x:x1,y:y1,width:x2-x1,height:y2-y1)
    let confidence: Float
    let classIndex: Int
}

final class YOLODetector {
    private let model: MLModel
    private let inputName: String
    private let outputNames: [String]

    private let confidenceThreshold: Float
    private let iouThreshold: Float
    private let personClassIndex: Int

    init?(modelName: String = "yolov8n",
          confidenceThreshold: Float = 0.25,
          iouThreshold: Float = 0.45,
          personClassIndex: Int = 0) {
        guard let url = Bundle.main.url(forResource: modelName, withExtension: "mlmodelc"),
              let loadedModel = try? MLModel(contentsOf: url) else {
            return nil
        }

        guard let imageInput = loadedModel.modelDescription.inputDescriptionsByName.first(where: { $0.value.type == .image })?.key else {
            return nil
        }

        self.model = loadedModel
        self.inputName = imageInput
        self.outputNames = Array(loadedModel.modelDescription.outputDescriptionsByName.keys)
        self.confidenceThreshold = confidenceThreshold
        self.iouThreshold = iouThreshold
        self.personClassIndex = personClassIndex
    }

    func detectPersons(in frame: CVPixelBuffer) -> [Detection] {
        guard let provider = try? MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(pixelBuffer: frame)]),
              let prediction = try? model.prediction(from: provider) else {
            return []
        }

        let frameWidth = CVPixelBufferGetWidth(frame)
        let frameHeight = CVPixelBufferGetHeight(frame)
        let all = decodeYOLOOutput(prediction: prediction, frameWidth: frameWidth, frameHeight: frameHeight)
            .filter { $0.classIndex == personClassIndex }

        return nonMaximumSuppression(all, iouThreshold: iouThreshold)
    }

    private func decodeYOLOOutput(prediction: MLFeatureProvider,
                                  frameWidth: Int,
                                  frameHeight: Int) -> [Detection] {
        let multiArrays: [MLMultiArray] = outputNames.compactMap { prediction.featureValue(for: $0)?.multiArrayValue }
        guard let tensor = multiArrays.max(by: { $0.count < $1.count }) else { return [] }

        let shape = tensor.shape.map { $0.intValue }
        let values = (0..<tensor.count).map { tensor[$0].floatValue }

        // Typical ultralytics export raw tensor: [1, 84, 8400] or [84, 8400]
        let channels: Int
        let anchors: Int

        if shape.count == 3 {
            channels = shape[1]
            anchors = shape[2]
        } else if shape.count == 2 {
            channels = shape[0]
            anchors = shape[1]
        } else {
            return []
        }

        guard channels > 5, anchors > 0 else { return [] }
        let classCount = channels - 4

        var detections: [Detection] = []
        detections.reserveCapacity(anchors)

        for anchor in 0..<anchors {
            let cx = values[0 * anchors + anchor]
            let cy = values[1 * anchors + anchor]
            let w = values[2 * anchors + anchor]
            let h = values[3 * anchors + anchor]

            var bestClass = 0
            var bestScore: Float = 0
            for c in 0..<classCount {
                let score = values[(4 + c) * anchors + anchor]
                if score > bestScore {
                    bestScore = score
                    bestClass = c
                }
            }

            guard bestScore >= confidenceThreshold else { continue }

            let x1 = max(0, min(Float(frameWidth - 1), cx - w / 2))
            let y1 = max(0, min(Float(frameHeight - 1), cy - h / 2))
            let x2 = max(0, min(Float(frameWidth - 1), cx + w / 2))
            let y2 = max(0, min(Float(frameHeight - 1), cy + h / 2))

            guard x2 > x1, y2 > y1 else { continue }

            let rect = CGRect(x: CGFloat(x1), y: CGFloat(y1), width: CGFloat(x2 - x1), height: CGFloat(y2 - y1))
            detections.append(Detection(bbox: rect, confidence: bestScore, classIndex: bestClass))
        }

        return detections
    }

    private func nonMaximumSuppression(_ detections: [Detection], iouThreshold: Float) -> [Detection] {
        let sorted = detections.sorted { $0.confidence > $1.confidence }
        var selected: [Detection] = []

        for det in sorted {
            var keep = true
            for chosen in selected {
                if iou(det.bbox, chosen.bbox) > iouThreshold {
                    keep = false
                    break
                }
            }
            if keep { selected.append(det) }
        }

        return selected
    }

    private func iou(_ a: CGRect, _ b: CGRect) -> Float {
        let inter = a.intersection(b)
        guard !inter.isNull else { return 0 }
        let interArea = Float(inter.width * inter.height)
        let unionArea = Float(a.width * a.height + b.width * b.height - inter.width * inter.height)
        return unionArea > 0 ? interArea / unionArea : 0
    }
}
