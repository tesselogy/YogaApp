import Foundation
import CoreGraphics

final class ByteTracker {
    private(set) var tracks: [Track] = []

    private let highThreshold: Float
    private let lowThreshold: Float
    private let matchIoUThreshold: Float
    private let maxMissedFrames: Int
    private let emaAlpha: CGFloat
    private var nextTrackID: Int = 1

    init(highThreshold: Float = 0.5,
         lowThreshold: Float = 0.1,
         matchIoUThreshold: Float = 0.3,
         maxMissedFrames: Int = 30,
         emaAlpha: CGFloat = 0.35) {
        self.highThreshold = highThreshold
        self.lowThreshold = lowThreshold
        self.matchIoUThreshold = matchIoUThreshold
        self.maxMissedFrames = maxMissedFrames
        self.emaAlpha = emaAlpha
    }

    func update(with detections: [Detection]) -> [Track] {
        let high = detections.filter { $0.confidence >= highThreshold }
        let low = detections.filter { $0.confidence >= lowThreshold && $0.confidence < highThreshold }

        var unmatchedTracks = Set(tracks.indices)
        var unmatchedHigh = Set(high.indices)

        // Stage 1: match high-confidence detections to tracks
        let highMatches = greedyMatch(trackIndices: Array(unmatchedTracks), detections: high, detectionIndices: Array(unmatchedHigh))
        for (ti, di) in highMatches {
            tracks[ti].update(bbox: high[di].bbox, confidence: high[di].confidence, emaAlpha: emaAlpha)
            unmatchedTracks.remove(ti)
            unmatchedHigh.remove(di)
        }

        // Stage 2: match unmatched tracks with low-confidence detections
        var unmatchedLow = Set(low.indices)
        let lowMatches = greedyMatch(trackIndices: Array(unmatchedTracks), detections: low, detectionIndices: Array(unmatchedLow))
        for (ti, di) in lowMatches {
            tracks[ti].update(bbox: low[di].bbox, confidence: low[di].confidence, emaAlpha: emaAlpha)
            unmatchedTracks.remove(ti)
            unmatchedLow.remove(di)
        }

        // mark unmatched tracks lost
        for ti in unmatchedTracks {
            tracks[ti].missedFrames += 1
            tracks[ti].age += 1
        }

        // create new tracks from unmatched high-confidence detections
        for di in unmatchedHigh {
            let det = high[di]
            let t = Track(id: nextTrackID, bbox: det.bbox, confidence: det.confidence)
            nextTrackID += 1
            tracks.append(t)
        }

        tracks.removeAll { $0.missedFrames > maxMissedFrames }
        return tracks
    }

    private func greedyMatch(trackIndices: [Int], detections: [Detection], detectionIndices: [Int]) -> [(Int, Int)] {
        guard !trackIndices.isEmpty, !detectionIndices.isEmpty else { return [] }

        var pairs: [(Int, Int, Float)] = []
        for ti in trackIndices {
            for di in detectionIndices {
                let i = iou(tracks[ti].smoothedBBox, detections[di].bbox)
                if i >= matchIoUThreshold {
                    pairs.append((ti, di, i))
                }
            }
        }

        pairs.sort { $0.2 > $1.2 }

        var usedTracks = Set<Int>()
        var usedDetections = Set<Int>()
        var matches: [(Int, Int)] = []

        for (ti, di, _) in pairs {
            guard !usedTracks.contains(ti), !usedDetections.contains(di) else { continue }
            usedTracks.insert(ti)
            usedDetections.insert(di)
            matches.append((ti, di))
        }

        return matches
    }

    private func iou(_ a: CGRect, _ b: CGRect) -> Float {
        let inter = a.intersection(b)
        guard !inter.isNull else { return 0 }
        let interArea = Float(inter.width * inter.height)
        let unionArea = Float(a.width * a.height + b.width * b.height - inter.width * inter.height)
        return unionArea > 0 ? interArea / unionArea : 0
    }
}
