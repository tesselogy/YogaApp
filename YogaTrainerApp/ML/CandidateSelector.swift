import Foundation
import CoreGraphics

final class CandidateSelector {
    func selectBest(from tracks: [Track], frameWidth: Int, frameHeight: Int) -> Track? {
        guard !tracks.isEmpty else { return nil }

        let center = CGPoint(x: CGFloat(frameWidth) / 2, y: CGFloat(frameHeight) / 2)
        let maxDistance = hypot(center.x, center.y)
        let frameArea = CGFloat(frameWidth * frameHeight)

        var best: (track: Track, score: CGFloat)?

        for track in tracks where track.missedFrames == 0 {
            let box = track.smoothedBBox
            let boxCenter = CGPoint(x: box.midX, y: box.midY)
            let distance = hypot(boxCenter.x - center.x, boxCenter.y - center.y)
            let centerScore = 1 - min(1, distance / maxDistance)
            let areaScore = min(1, (box.width * box.height) / frameArea)
            let score = 0.6 * centerScore + 0.4 * areaScore

            if best == nil || score > best!.score {
                best = (track, score)
            }
        }

        return best?.track
    }
}
