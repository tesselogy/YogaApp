import SwiftUI
import Foundation

struct ContentView: View {

    @StateObject var camera = CameraManager()
    @StateObject var poseState = PoseState()

    private let frameProcessor = FrameProcessor()

    @State var trackedBBox: CGRect?
    @State var detectionStatus: String = "Waiting for person..."
    @State var isProcessingFrame: Bool = false
    @State var debugLine: String = "-"

    var body: some View {
        GeometryReader { geo in
            ZStack {
                if camera.currentBuffer != nil {
                    CameraPreview(pixelBuffer: camera.currentBuffer)
                        .scaledToFill()
                        .onChange(of: camera.currentBuffer) { _, newBuffer in
                            guard let buffer = newBuffer else { return }
                            guard !isProcessingFrame else { return }
                            guard let frameProcessor else {
                                detectionStatus = "Models not loaded"
                                return
                            }

                            isProcessingFrame = true

                            DispatchQueue.global(qos: .userInitiated).async {
                                let output = frameProcessor.process(frame: buffer)
                                DispatchQueue.main.async {
                                    if let track = output.selectedTrack {
                                        trackedBBox = track.smoothedBBox
                                        detectionStatus = "Person detected (id: \(track.id))"
                                        poseState.update(newPose: output.label)
                                        debugLine = output.debugInfo
                                    } else {
                                        trackedBBox = nil
                                        detectionStatus = "Person not detected"
                                        poseState.update(newPose: "no_person")
                                        debugLine = output.debugInfo
                                    }

                                    print("[ContentView] pose=\(output.label) confidence=\(String(format: "%.2f", output.confidence)) detection=\(detectionStatus) debug=\(output.debugInfo)")
                                    isProcessingFrame = false
                                }
                            }
                        }
                }

                if let trackedBBox,
                   let buffer = camera.currentBuffer {
                    DetectionBox(rect: trackedBBox,
                                 frameWidth: CVPixelBufferGetWidth(buffer),
                                 frameHeight: CVPixelBufferGetHeight(buffer))
                        .stroke(.green, lineWidth: 4)
                        .frame(width: geo.size.width, height: geo.size.height)
                }

                VStack {
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

                    Text(debugLine)
                        .font(.caption2)
                        .lineLimit(4)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 8)
                        .background(.black.opacity(0.55))
                        .foregroundColor(.yellow)
                        .cornerRadius(10)

                    Spacer()
                }
            }
            .ignoresSafeArea()
        }
    }
}

private struct DetectionBox: Shape {
    let rect: CGRect
    let frameWidth: Int
    let frameHeight: Int

    func path(in drawRect: CGRect) -> Path {
        guard frameWidth > 0, frameHeight > 0 else { return Path() }
        let sx = drawRect.width / CGFloat(frameWidth)
        let sy = drawRect.height / CGFloat(frameHeight)

        let scaled = CGRect(
            x: rect.minX * sx,
            y: (CGFloat(frameHeight) - rect.maxY) * sy,
            width: rect.width * sx,
            height: rect.height * sy
        )

        var path = Path()
        path.addRect(scaled)
        return path
    }
}
