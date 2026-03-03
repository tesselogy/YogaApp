import SwiftUI

struct PoseProgressRing: View {

    var progress: Double

    var body: some View {
        Circle()
            .trim(from: 0, to: min(progress / 10, 1))
            .stroke(
                AngularGradient(
                    gradient: Gradient(colors: [.purple, .cyan, .mint]),
                    center: .center
                ),
                style: StrokeStyle(lineWidth: 6, lineCap: .round)
            )
            .rotationEffect(.degrees(-90))
    }
}