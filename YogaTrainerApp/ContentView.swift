import SwiftUI
import Vision

struct ContentView: View {

    @StateObject var camera = CameraManager()
    @StateObject var poseState = PoseState()

    let focusEngine = FocusTrackingEngine()
    let classifier = ClassificationEngine()

    @State var trackedObservation: VNDetectedObjectObservation?

    var body: some View {

        ZStack {

            if let buffer = camera.currentBuffer {
                Image(decorative: CIImage(cvPixelBuffer: buffer), scale: 1)
                    .resizable()
                    .scaledToFill()
                    .onAppear {
                        focusEngine.process(buffer: buffer) { obs in
                            trackedObservation = obs
                        }
                    }
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