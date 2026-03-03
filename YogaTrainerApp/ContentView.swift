import SwiftUI
import Vision
import Foundation

struct ContentView: View {

    @StateObject var camera = CameraManager()
    @StateObject var poseState = PoseState()

    let focusEngine = FocusTrackingEngine()
    let classifier = ClassificationEngine()

    @State var trackedPerson: DetectedPerson?
    @State var classificationROI: CGRect?
    @State var detectionStatus: String = "Waiting for person..."
    @State var isProcessingFrame: Bool = false

    private let classificationIntervalFrames = 3

    private func expandedROI(from bbox: CGRect) -> CGRect {
        let scale: CGFloat = 1.1
        let newWidth = min(1, bbox.width * scale)
        let newHeight = min(1, bbox.height * scale)
        let newX = max(0, min(1 - newWidth, bbox.midX - newWidth / 2))
        let newY = max(0, min(1 - newHeight, bbox.midY - newHeight / 2))
        return CGRect(x: newX, y: newY, width: newWidth, height: newHeight)
    }

    var body: some View {

        GeometryReader { geo in
            ZStack {

                if camera.currentBuffer != nil {
                    CameraPreview(pixelBuffer: camera.currentBuffer)
                        .scaledToFill()
                        .onChange(of: camera.currentBuffer) { _, newBuffer in
                            guard let buffer = newBuffer else { return }
                            guard !isProcessingFrame else { return }
                            isProcessingFrame = true

                            focusEngine.process(buffer: buffer) { result in
                                guard let active = result.active else {
                                    DispatchQueue.main.async {
                                        trackedPerson = nil
                                        classificationROI = nil
                                        detectionStatus = "No subject"
                                        poseState.resetForNoPerson()
                                        isProcessingFrame = false
                                    }
                                    return
                                }

                                let roi = expandedROI(from: active.observation.boundingBox)

                                if !poseState.shouldClassifyThisFrame(every: classificationIntervalFrames) {
                                    DispatchQueue.main.async {
                                        trackedPerson = active
                                        classificationROI = roi
                                        detectionStatus = "Focus ID: \(active.id)"
                                        isProcessingFrame = false
                                    }
                                    return
                                }

                                classifier.classify(buffer: buffer, regionOfInterest: roi) { label, confidence in
                                    DispatchQueue.main.async {
                                        trackedPerson = active
                                        classificationROI = roi
                                        detectionStatus = "Focus ID: \(active.id)"
                                        poseState.updateFromClassifier(label: label, confidence: confidence)
                                        isProcessingFrame = false
                                    }
                                }
                            }
                        }
                }

                if let trackedPerson {
                    DetectionBox(observation: trackedPerson.observation)
                        .stroke(boxColor(confidence: trackedPerson.confidence), lineWidth: boxWidth(confidence: trackedPerson.confidence))
                        .frame(
                            width: trackedPerson.observation.boundingBox.width * geo.size.width,
                            height: trackedPerson.observation.boundingBox.height * geo.size.height
                        )
                        .position(
                            x: trackedPerson.observation.boundingBox.midX * geo.size.width,
                            y: (1 - trackedPerson.observation.boundingBox.midY) * geo.size.height
                        )
                }

                if let classificationROI {
                    DetectionBox(observation: VNDetectedObjectObservation(boundingBox: classificationROI))
                        .stroke(.yellow, style: StrokeStyle(lineWidth: 3, dash: [10, 8]))
                        .frame(
                            width: classificationROI.width * geo.size.width,
                            height: classificationROI.height * geo.size.height
                        )
                        .position(
                            x: classificationROI.midX * geo.size.width,
                            y: (1 - classificationROI.midY) * geo.size.height
                        )
                }

                VStack {
                    Spacer()

                    Text(poseState.pose.uppercased())
                        .font(.system(size: 72, weight: .bold))
                        .minimumScaleFactor(0.5)
                        .foregroundColor(.white)
                        .shadow(color: .black.opacity(0.7), radius: 4)

                    if poseState.pose != "UNCERTAIN" && poseState.pose != "NO_PERSON" {
                        Text("HOLD: \(poseState.holdTime, specifier: "%.1f")s")
                            .font(.headline)
                            .foregroundColor(.white)

                        PoseProgressRing(progress: poseState.progress)
                            .frame(width: 90, height: 90)
                            .padding(.top, 6)
                    }

                    if let completion = poseState.completionWord {
                        Text(completion)
                            .font(.system(size: 40, weight: .heavy))
                            .foregroundColor(Color(red: 0.6, green: 0.96, blue: 0.8))
                            .scaleEffect(1.08)
                    }

                    Text(detectionStatus)
                        .font(.caption)
                        .foregroundColor(.white)
                        .padding(.top, 8)

                    Spacer()

                    Text("Press Q / Й to quit")
                        .font(.caption)
                        .foregroundColor(.white)
                        .padding(10)
                        .background(Color.black.opacity(0.45))
                        .cornerRadius(8)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.leading, 18)
                        .padding(.bottom, 24)
                }
            }
            .ignoresSafeArea()
            .background(.black)
        }
    }

    private func boxColor(confidence: Double) -> Color {
        if confidence < 0.45 { return Color(red: 0.7, green: 0.45, blue: 0.9) }
        if confidence < 0.7 { return Color(red: 0.95, green: 0.66, blue: 0.45) }
        return Color(red: 0.55, green: 0.93, blue: 0.75)
    }

    private func boxWidth(confidence: Double) -> CGFloat {
        CGFloat(2.0 + min(max(confidence, 0), 1) * 5.0)
    }
}

private struct DetectionBox: Shape {
    let observation: VNDetectedObjectObservation

    func path(in rect: CGRect) -> Path {
        Path(CGRect(origin: .zero, size: rect.size))
    }
}
