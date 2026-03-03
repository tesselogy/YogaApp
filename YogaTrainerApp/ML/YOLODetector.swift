import CoreML
import Foundation
import CoreVideo
import CoreImage

struct Detection {
    let bbox: CGRect // pixel xyxy stored as CGRect(x:x1,y:y1,width:x2-x1,height:y2-y1)
    let confidence: Float
    let classIndex: Int
}

private struct InputPreparation {
    let buffer: CVPixelBuffer
    let modelWidth: Int
    let modelHeight: Int
    let frameWidth: Int
    let frameHeight: Int
}

final class YOLODetector {
    static var lastInitError: String = ""

    private var lastDebugMessage: String = ""
    private let model: MLModel
    private let inputName: String
    private let outputNames: [String]

    private let confidenceThreshold: Float
    private let iouThreshold: Float
    private let personClassIndex: Int
    private let inputConstraint: MLImageConstraint?
    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])

    init?(modelName: String = "yolov8n",
          confidenceThreshold: Float = 0.25,
          iouThreshold: Float = 0.45,
          personClassIndex: Int = 0) {
        guard let url = Bundle.main.url(forResource: modelName, withExtension: "mlmodelc") else {
            Self.lastInitError = "model_not_found:\(modelName).mlmodelc"
            return nil
        }

        func loadModel(computeUnits: MLComputeUnits) throws -> MLModel {
            let config = MLModelConfiguration()
            config.computeUnits = computeUnits
            return try MLModel(contentsOf: url, configuration: config)
        }

        let loadedModel: MLModel
        do {
            loadedModel = try loadModel(computeUnits: .all)
        } catch {
            do {
                loadedModel = try loadModel(computeUnits: .cpuOnly)
                Self.lastInitError = "fallback_cpu_only_used after all_failed: \(error.localizedDescription)"
            } catch {
                Self.lastInitError = "model_load_failed: \(error.localizedDescription)"
                return nil
            }
        }

        guard let imageInput = loadedModel.modelDescription.inputDescriptionsByName.first(where: { $0.value.type == .image })?.key else {
            Self.lastInitError = "image_input_not_found"
            return nil
        }

        self.model = loadedModel
        self.inputName = imageInput
        self.outputNames = Array(loadedModel.modelDescription.outputDescriptionsByName.keys)
        self.confidenceThreshold = confidenceThreshold
        self.iouThreshold = iouThreshold
        self.personClassIndex = personClassIndex
        self.inputConstraint = loadedModel.modelDescription.inputDescriptionsByName[imageInput]?.imageConstraint
    }

    func detectPersons(in frame: CVPixelBuffer) -> [Detection] {
        guard let prepared = makeInputBufferIfNeeded(from: frame) else {
            let actualFormat = CVPixelBufferGetPixelFormatType(frame)
            let expectedSize = expectedModelSizeDescription()
            lastDebugMessage = "input_prepare_failed expected_fmt=\(pixelFormatName(inputConstraint?.pixelFormatType ?? 0)) actual_fmt=\(pixelFormatName(actualFormat)) expected_size=\(expectedSize) actual_size=\(CVPixelBufferGetWidth(frame))x\(CVPixelBufferGetHeight(frame))"
            return []
        }

        do {
            let provider = try MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(pixelBuffer: prepared.buffer)])
            let prediction = try model.prediction(from: provider)

            let all = decodeYOLOOutput(
                prediction: prediction,
                frameWidth: prepared.frameWidth,
                frameHeight: prepared.frameHeight,
                modelInputWidth: prepared.modelWidth,
                modelInputHeight: prepared.modelHeight
            )
            let persons = all.filter { $0.classIndex == personClassIndex }
            let nms = nonMaximumSuppression(persons, iouThreshold: iouThreshold)

            let shapeInfo = outputNames
                .compactMap { name -> String? in
                    guard let m = prediction.featureValue(for: name)?.multiArrayValue else { return nil }
                    return "\(name):\(m.shape.map{$0.intValue})"
                }
                .joined(separator: ",")

            let actualFormat = CVPixelBufferGetPixelFormatType(frame)
            let preparedFormat = CVPixelBufferGetPixelFormatType(prepared.buffer)
            lastDebugMessage = "fmt=\(pixelFormatName(actualFormat))->\(pixelFormatName(preparedFormat)) size=\(prepared.frameWidth)x\(prepared.frameHeight)->\(prepared.modelWidth)x\(prepared.modelHeight) outputs=[\(shapeInfo)] all=\(all.count) person=\(persons.count) nms=\(nms.count)"
            return nms
        } catch {
            let expected = pixelFormatName(inputConstraint?.pixelFormatType ?? 0)
            let expectedSize = expectedModelSizeDescription()
            lastDebugMessage = "prediction_failed expected_fmt=\(expected) expected_size=\(expectedSize) err=\(error.localizedDescription)"
            return []
        }
    }

    func debugMessage() -> String {
        lastDebugMessage
    }

    private func decodeYOLOOutput(prediction: MLFeatureProvider,
                                  frameWidth: Int,
                                  frameHeight: Int,
                                  modelInputWidth: Int,
                                  modelInputHeight: Int) -> [Detection] {
        let multiArrays: [MLMultiArray] = outputNames.compactMap { prediction.featureValue(for: $0)?.multiArrayValue }
        guard let tensor = multiArrays.max(by: { $0.count < $1.count }) else { return [] }

        let shape = tensor.shape.map { $0.intValue }
        let values = (0..<tensor.count).map { tensor[$0].floatValue }

        // Case A: [1,84,8400] or [84,8400]
        if (shape.count == 3 && shape[1] > 5) || (shape.count == 2 && shape[0] > 5) {
            let channels = shape.count == 3 ? shape[1] : shape[0]
            let anchors = shape.count == 3 ? shape[2] : shape[1]
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

                let normalized = max(abs(cx), abs(cy), abs(w), abs(h)) <= 2.0
                let mapX: Float = (modelInputWidth > 0 && modelInputWidth != frameWidth) ? Float(frameWidth) / Float(modelInputWidth) : 1
                let mapY: Float = (modelInputHeight > 0 && modelInputHeight != frameHeight) ? Float(frameHeight) / Float(modelInputHeight) : 1

                let sx: Float = normalized ? Float(frameWidth) : mapX
                let sy: Float = normalized ? Float(frameHeight) : mapY

                let x1 = max(0, min(Float(frameWidth - 1), (cx - w / 2) * sx))
                let y1 = max(0, min(Float(frameHeight - 1), (cy - h / 2) * sy))
                let x2 = max(0, min(Float(frameWidth - 1), (cx + w / 2) * sx))
                let y2 = max(0, min(Float(frameHeight - 1), (cy + h / 2) * sy))
                guard x2 > x1, y2 > y1 else { continue }

                detections.append(Detection(
                    bbox: CGRect(x: CGFloat(x1), y: CGFloat(y1), width: CGFloat(x2 - x1), height: CGFloat(y2 - y1)),
                    confidence: bestScore,
                    classIndex: bestClass
                ))
            }
            return detections
        }

        // Case B: [N,6] / [1,N,6] => x1,y1,x2,y2,conf,class
        let flat6 = shape.last == 6
        if flat6 {
            let rows = tensor.count / 6
            var detections: [Detection] = []
            detections.reserveCapacity(rows)
            for r in 0..<rows {
                let base = r * 6
                let x1raw = values[base]
                let y1raw = values[base + 1]
                let x2raw = values[base + 2]
                let y2raw = values[base + 3]
                let conf = values[base + 4]
                let cls = Int(values[base + 5])
                guard conf >= confidenceThreshold else { continue }

                let normalized = max(abs(x1raw), abs(y1raw), abs(x2raw), abs(y2raw)) <= 2.0
                let mapX: Float = (modelInputWidth > 0 && modelInputWidth != frameWidth) ? Float(frameWidth) / Float(modelInputWidth) : 1
                let mapY: Float = (modelInputHeight > 0 && modelInputHeight != frameHeight) ? Float(frameHeight) / Float(modelInputHeight) : 1
                let sx: Float = normalized ? Float(frameWidth) : mapX
                let sy: Float = normalized ? Float(frameHeight) : mapY

                let x1 = max(0, min(Float(frameWidth - 1), x1raw * sx))
                let y1 = max(0, min(Float(frameHeight - 1), y1raw * sy))
                let x2 = max(0, min(Float(frameWidth - 1), x2raw * sx))
                let y2 = max(0, min(Float(frameHeight - 1), y2raw * sy))
                guard x2 > x1, y2 > y1 else { continue }

                detections.append(Detection(
                    bbox: CGRect(x: CGFloat(x1), y: CGFloat(y1), width: CGFloat(x2 - x1), height: CGFloat(y2 - y1)),
                    confidence: conf,
                    classIndex: cls
                ))
            }
            return detections
        }

        lastDebugMessage = "unsupported_tensor_shape=\(shape)"
        return []
    }

    private func makeInputBufferIfNeeded(from frame: CVPixelBuffer) -> InputPreparation? {
        let frameWidth = CVPixelBufferGetWidth(frame)
        let frameHeight = CVPixelBufferGetHeight(frame)

        guard let inputConstraint else {
            return InputPreparation(buffer: frame, modelWidth: frameWidth, modelHeight: frameHeight, frameWidth: frameWidth, frameHeight: frameHeight)
        }

        let expectedFormat = inputConstraint.pixelFormatType
        let actualFormat = CVPixelBufferGetPixelFormatType(frame)

        let expectedWidth = inputConstraint.pixelsWide > 0 ? inputConstraint.pixelsWide : frameWidth
        let expectedHeight = inputConstraint.pixelsHigh > 0 ? inputConstraint.pixelsHigh : frameHeight

        let needsFormatConversion = expectedFormat != 0 && expectedFormat != actualFormat
        let needsResize = frameWidth != expectedWidth || frameHeight != expectedHeight

        if !needsFormatConversion && !needsResize {
            return InputPreparation(buffer: frame, modelWidth: frameWidth, modelHeight: frameHeight, frameWidth: frameWidth, frameHeight: frameHeight)
        }

        var converted: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true
        ]

        let outputFormat = expectedFormat == 0 ? actualFormat : expectedFormat

        guard CVPixelBufferCreate(kCFAllocatorDefault,
                                  expectedWidth,
                                  expectedHeight,
                                  outputFormat,
                                  attrs as CFDictionary,
                                  &converted) == kCVReturnSuccess,
              let converted else {
            return nil
        }

        let ciImage = CIImage(cvPixelBuffer: frame)
        let sourceRect = CGRect(x: 0, y: 0, width: frameWidth, height: frameHeight)
        let targetRect = CGRect(x: 0, y: 0, width: expectedWidth, height: expectedHeight)
        let scaleX = targetRect.width / sourceRect.width
        let scaleY = targetRect.height / sourceRect.height
        let resized = ciImage.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
        ciContext.render(resized, to: converted)

        return InputPreparation(buffer: converted,
                                modelWidth: expectedWidth,
                                modelHeight: expectedHeight,
                                frameWidth: frameWidth,
                                frameHeight: frameHeight)
    }

    private func expectedModelSizeDescription() -> String {
        guard let inputConstraint else { return "flexible" }
        if inputConstraint.pixelsWide > 0 && inputConstraint.pixelsHigh > 0 {
            return "\(inputConstraint.pixelsWide)x\(inputConstraint.pixelsHigh)"
        }
        return "flexible"
    }

    private func pixelFormatName(_ type: OSType) -> String {
        switch type {
        case kCVPixelFormatType_32BGRA: return "32BGRA"
        case kCVPixelFormatType_OneComponent8: return "OneComponent8"
        case kCVPixelFormatType_420YpCbCr8BiPlanarFullRange: return "420f"
        case kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange: return "420v"
        case 0: return "unspecified"
        default: return "\(type)"
        }
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
