import SwiftUI

struct PoseProgressRing: View {

    var progress: Double

    private var easedProgress: Double {
        let t = min(max(progress, 0), 1)
        return 1 - pow(1 - t, 3) // easeOutCubic
    }

    private var ringColor: Color {
        if progress < 0.5 {
            let t = progress / 0.5
            return Color(
                red: 0.60 + 0.28 * t,
                green: 0.46 + 0.24 * t,
                blue: 0.88 - 0.28 * t
            )
        }

        let t = (progress - 0.5) / 0.5
        return Color(
            red: 0.88 - 0.28 * t,
            green: 0.70 + 0.26 * t,
            blue: 0.60 + 0.20 * t
        )
    }

    var body: some View {
        Circle()
            .trim(from: 0, to: easedProgress)
            .stroke(ringColor, style: StrokeStyle(lineWidth: 10, lineCap: .round))
            .rotationEffect(.degrees(-90))
            .scaleEffect(progress >= 1 ? 1.2 : 1.0)
            .animation(.interpolatingSpring(stiffness: 160, damping: 14), value: progress >= 1)
    }
}
