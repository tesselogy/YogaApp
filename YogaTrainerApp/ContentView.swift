import SwiftUI
import Vision
import Foundation

struct ContentView: View {

    @StateObject var camera = CameraManager()
    @StateObject var poseState = PoseState()

    let focusEngine = FocusTrackingEngine()
    let classifier = ClassificationEngine()

    @State var trackedObservation: VNDetectedObjectObservation?
    @State var classificationROI: CGRect?
    @State var detectionStatus: String = "Waiting for person..."

    private func expandedROI(from bbox: CGRect) -> CGRect {
        let scale: CGFloat = 1.35
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

                            focusEngine.process(buffer: buffer) { obs in
                                DispatchQueue.main.async {
                                    trackedObservation = obs
                                    detectionStatus = obs == nil ? "Person not detected" : "Person detected"
                                }

                                guard let obs else {
                                    DispatchQueue.main.async {
                                        classificationROI = nil
                                        poseState.update(newPose: "no_person")
                                        print("[ContentView] skip classify: no person detected")
                                    }
                                    return
                                }

                                let roi = expandedROI(from: obs.boundingBox)

                                DispatchQueue.main.async {
                                    classificationROI = roi
                                }

                                classifier.classify(buffer: buffer, regionOfInterest: roi) { label, confidence in
                                    DispatchQueue.main.async {
                                        poseState.update(newPose: label)
                                        print("[ContentView] pose=\(label) confidence=\(String(format: "%.2f", confidence)) detection=\(detectionStatus)")
                                    }
                                }
                            }
                        }
                }

                if let trackedObservation {
                    DetectionBox(observation: trackedObservation)
                        .stroke(.green, lineWidth: 4)
                        .frame(
                            width: trackedObservation.boundingBox.width * geo.size.width,
                            height: trackedObservation.boundingBox.height * geo.size.height
                        )
                        .position(
                            x: trackedObservation.boundingBox.midX * geo.size.width,
                            y: (1 - trackedObservation.boundingBox.midY) * geo.size.height
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
                    HStack {
                        Spacer()

                        if classificationROI != nil {
                            Text("Yellow dashed box = ROI for classify")
                                .font(.caption)
                                .padding(.horizontal, 10)
                                .padding(.vertical, 6)
                                .background(.black.opacity(0.45))
                                .foregroundColor(.yellow)
                                .cornerRadius(10)
                                .padding(.top, 30)
                                .padding(.trailing, 20)
                        }
                    }

                    Spacer()

                    Text(poseState.pose.uppercased())
                        .font(.system(size: 80, weight: .bold))
                        .foregroundStyle(
                            LinearGradient(
                                colors: [.purple, .cyan, .mint],
                                startPoint: .leading,
                                endPoint: .trailing
                            )
                        )
                        .shadow(radius: 15)

                    Text("Hold: \(poseState.holdTime, specifier: "%.1f")s")
                        .font(.title3)
                        .foregroundColor(.white.opacity(0.8))

                    Text(detectionStatus)
                        .font(.headline)
                        .padding(.horizontal, 16)
                        .padding(.vertical, 8)
                        .background(.black.opacity(0.45))
                        .foregroundColor(.white)
                        .cornerRadius(12)

                    Spacer()

                    Text("Press Q to quit")
                        .font(.caption)
                        .padding(12)
                        .background(.ultraThinMaterial)
                        .cornerRadius(18)
                        .padding(.bottom, 40)
                }
            }
            .ignoresSafeArea()
        }
    }
}

private struct DetectionBox: Shape {
    let observation: VNDetectedObjectObservation

    func path(in rect: CGRect) -> Path {
        Path(CGRect(origin: .zero, size: rect.size))
    }
}
