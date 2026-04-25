import Foundation
import simd

struct PoseFeatureConfig {
    var includeXDistanceFeatures: Bool = true
    var visibilityThreshold: Float = 0.5
    var enableMirrorInference: Bool = true
}

struct PoseFeatureResult {
    let featureNames: [String]
    let rawFeatures: [Float]
    let weightedFeatures: [Float]
}

struct PoseNormalizationOutput {
    let metricPoints: [SIMD3<Float>]
    let localPoints: [SIMD3<Float>]
}

final class PoseNormalizer {

    private(set) var previousWorldPoints: [SIMD3<Float>]?

    func normalize(landmarks: [[Float]], visibility: [Float]?, config: PoseFeatureConfig) -> PoseNormalizationOutput? {
        guard landmarks.count >= 33 else { return nil }

        var points = landmarks.map { row -> SIMD3<Float> in
            let x = row.count > 0 ? row[0] : 0
            let y = row.count > 1 ? row[1] : 0
            let z = row.count > 2 ? row[2] : 0
            return SIMD3<Float>(x, y, z)
        }

        if let visibility {
            for i in 0..<min(points.count, visibility.count) where visibility[i] < config.visibilityThreshold {
                points[i] = SIMD3<Float>(.nan, .nan, .nan)
            }
        }

        if let previous = previousWorldPoints {
            for i in 0..<min(points.count, previous.count) where points[i].containsNaN {
                points[i] = previous[i]
            }
        }

        for i in points.indices where points[i].containsNaN {
            points[i] = .zero
        }

        previousWorldPoints = points

        let hipL = points[23]
        let hipR = points[24]
        let kneeL = points[25]
        let kneeR = points[26]
        let shL = points[11]
        let shR = points[12]

        let pelvis = 0.5 * (hipL + hipR)
        let centered = points.map { $0 - pelvis }

        var xAxis = normalizeOrFallback(hipR - hipL, fallback: SIMD3<Float>(1, 0, 0))

        let kneeMid = 0.5 * (kneeL + kneeR)
        let shoulderMid = 0.5 * (shL + shR)

        var ySeed = normalizeOrNil(pelvis - kneeMid)
        if ySeed == nil { ySeed = normalizeOrNil(shoulderMid - pelvis) }
        if ySeed == nil { ySeed = SIMD3<Float>(0, 1, 0) }

        var zAxis = normalizeOrFallback(simd_cross(xAxis, ySeed!), fallback: SIMD3<Float>(0, 0, 1))
        var yAxis = normalizeOrFallback(simd_cross(zAxis, xAxis), fallback: SIMD3<Float>(0, 1, 0))

        // Polar-decomposition based orthonormalization (SVD-like correction).
        var r = simd_float3x3(columns: (xAxis, yAxis, zAxis))
        for _ in 0..<2 {
            let invT = simd_transpose(simd_inverse(r))
            r = 0.5 * (r + invT)
        }

        if simd_determinant(r) < 0 {
            r.columns.2 *= -1
        }

        xAxis = normalizeOrFallback(r.columns.0, fallback: SIMD3<Float>(1, 0, 0))
        yAxis = normalizeOrFallback(r.columns.1, fallback: SIMD3<Float>(0, 1, 0))
        zAxis = normalizeOrFallback(r.columns.2, fallback: SIMD3<Float>(0, 0, 1))

        let rotation = simd_float3x3(columns: (xAxis, yAxis, zAxis))

        let centeredShoulderMid = 0.5 * (centered[11] + centered[12])
        var scale = simd_length(centeredShoulderMid)
        if scale < 1e-6 {
            scale = fallbackCoreScale(points: centered)
        }
        if scale < 1e-6 { scale = 1 }

        let metric = centered.map { $0 / scale }
        let local = metric.map { simd_transpose(rotation) * $0 }

        return PoseNormalizationOutput(metricPoints: metric, localPoints: local)
    }

    private func fallbackCoreScale(points: [SIMD3<Float>]) -> Float {
        let corePairs: [(Int, Int)] = [(11, 12), (23, 24), (11, 23), (12, 24)]
        let lengths = corePairs.map { simd_length(points[$0.0] - points[$0.1]) }.filter { $0 > 1e-6 }
        guard !lengths.isEmpty else { return 1 }
        return lengths.reduce(0, +) / Float(lengths.count)
    }

    private func normalizeOrNil(_ value: SIMD3<Float>) -> SIMD3<Float>? {
        let len = simd_length(value)
        guard len > 1e-6 else { return nil }
        return value / len
    }

    private func normalizeOrFallback(_ value: SIMD3<Float>, fallback: SIMD3<Float>) -> SIMD3<Float> {
        normalizeOrNil(value) ?? fallback
    }
}

final class PoseFeatureExtractor {

    private let config: PoseFeatureConfig
    private let normalizer = PoseNormalizer()
    private var previousMetricPoints: [SIMD3<Float>]?

    init(config: PoseFeatureConfig = .init()) {
        self.config = config
    }

    func extract(landmarks: [[Float]], visibility: [Float]?) -> PoseFeatureResult? {
        guard let normalized = normalizer.normalize(landmarks: landmarks, visibility: visibility, config: config) else {
            return nil
        }

        let pts = normalized.localPoints
        let gravity = SIMD3<Float>(0, -1, 0)
        let eps: Float = 1e-6

        let shoulderWidth = max(simd_length(pts[12] - pts[11]), eps)

        var names: [String] = [
            "knee_distance_norm",
            "foot_distance_norm"
        ]
        var values: [Float] = [
            simd_length(pts[25] - pts[26]) / shoulderWidth,
            simd_length(pts[31] - pts[32]) / shoulderWidth
        ]

        if config.includeXDistanceFeatures {
            names += ["KneeXDistanceNorm", "FootXDistanceNorm"]
            values += [
                abs(pts[25].x - pts[26].x) / shoulderWidth,
                abs(pts[31].x - pts[32].x) / shoulderWidth
            ]
        }

        let leftThigh = pts[25] - pts[23]
        let leftShin = pts[27] - pts[25]
        let rightThigh = pts[26] - pts[24]
        let rightShin = pts[28] - pts[26]
        let leftUpperArm = pts[13] - pts[11]
        let leftForearm = pts[15] - pts[13]
        let rightUpperArm = pts[14] - pts[12]
        let rightForearm = pts[16] - pts[14]

        names += [
            "left_thigh_angle_floor", "left_shin_angle_floor",
            "right_thigh_angle_floor", "right_shin_angle_floor",
            "left_upper_arm_angle_floor", "left_forearm_angle_floor",
            "right_upper_arm_angle_floor", "right_forearm_angle_floor"
        ]
        values += [
            angle(leftThigh, gravity), angle(leftShin, gravity),
            angle(rightThigh, gravity), angle(rightShin, gravity),
            angle(leftUpperArm, gravity), angle(leftForearm, gravity),
            angle(rightUpperArm, gravity), angle(rightForearm, gravity)
        ]

        names += [
            "left_knee_angle", "right_knee_angle",
            "left_elbow_angle", "right_elbow_angle",
            "left_hip_angle", "right_hip_angle",
            "left_shoulder_angle", "right_shoulder_angle"
        ]
        values += [
            jointAngle(pts[23], pts[25], pts[27]),
            jointAngle(pts[24], pts[26], pts[28]),
            jointAngle(pts[11], pts[13], pts[15]),
            jointAngle(pts[12], pts[14], pts[16]),
            jointAngle(pts[11], pts[23], pts[25]),
            jointAngle(pts[12], pts[24], pts[26]),
            jointAngle(pts[23], pts[11], pts[13]),
            jointAngle(pts[24], pts[12], pts[14])
        ]

        let pelvisMid = 0.5 * (pts[23] + pts[24])
        let shoulderMid = 0.5 * (pts[11] + pts[12])
        let torso = shoulderMid - pelvisMid

        names += ["torso_bend", "torso_angle_floor"]
        values += [angle(torso, -gravity), angle(torso, gravity)]

        let torsoAxis = normalizeOrFallback(torso, fallback: SIMD3<Float>(0, 1, 0))

        names += [
            "left_thigh_to_torso", "right_thigh_to_torso",
            "left_shin_to_torso", "right_shin_to_torso",
            "left_arm_to_torso", "right_arm_to_torso"
        ]
        values += [
            angle(leftThigh, torsoAxis), angle(rightThigh, torsoAxis),
            angle(leftShin, torsoAxis), angle(rightShin, torsoAxis),
            angle(leftUpperArm, torsoAxis), angle(rightUpperArm, torsoAxis)
        ]

        let expectedCount = config.includeXDistanceFeatures ? 28 : 26
        guard values.count == expectedCount, names.count == expectedCount else { return nil }

        var weights = Dictionary(uniqueKeysWithValues: names.map { ($0, Float(1.0)) })

        ["torso_bend", "torso_angle_floor"].forEach { weights[$0] = 1.8 }
        ["left_hip_angle", "right_hip_angle", "left_knee_angle", "right_knee_angle"].forEach { weights[$0] = 1.3 }
        ["left_thigh_angle_floor", "right_thigh_angle_floor", "left_shin_angle_floor", "right_shin_angle_floor"].forEach { weights[$0] = 1.2 }
        ["knee_distance_norm", "foot_distance_norm"].forEach { weights[$0] = 1.2 }
        if config.includeXDistanceFeatures {
            ["KneeXDistanceNorm", "FootXDistanceNorm"].forEach { weights[$0] = 1.15 }
        }

        let motionScale = motionScaleFromCurrentFrame(current: normalized.metricPoints)
        let weighted = zip(names, values).map { featureName, value in
            value * (weights[featureName] ?? 1) * (1 + motionScale)
        }

        previousMetricPoints = normalized.metricPoints

        return PoseFeatureResult(featureNames: names, rawFeatures: values, weightedFeatures: weighted)
    }

    private func motionScaleFromCurrentFrame(current: [SIMD3<Float>]) -> Float {
        guard let previousMetricPoints else { return 0 }
        let common = min(current.count, previousMetricPoints.count)
        guard common > 0 else { return 0 }

        var total: Float = 0
        for i in 0..<common {
            total += simd_length(current[i] - previousMetricPoints[i])
        }

        let meanJointSpeed = total / Float(common)
        return min(max(meanJointSpeed * 0.05, 0), 0.35)
    }

    private func angle(_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> Float {
        let na = normalizeOrFallback(a, fallback: SIMD3<Float>(0, 1, 0))
        let nb = normalizeOrFallback(b, fallback: SIMD3<Float>(0, 1, 0))
        let c = min(max(simd_dot(na, nb), -1), 1)
        return acos(c)
    }

    private func jointAngle(_ a: SIMD3<Float>, _ b: SIMD3<Float>, _ c: SIMD3<Float>) -> Float {
        angle(a - b, c - b)
    }

    private func normalizeOrFallback(_ value: SIMD3<Float>, fallback: SIMD3<Float>) -> SIMD3<Float> {
        let len = simd_length(value)
        if len <= 1e-6 { return fallback }
        return value / len
    }

    static let leftRightMirrorPairs: [(Int, Int)] = [
        (11, 12), (13, 14), (15, 16),
        (23, 24), (25, 26), (27, 28),
        (29, 30), (31, 32)
    ]

    static func mirrorLandmarks(_ landmarks: [[Float]]) -> [[Float]] {
        var mirrored = landmarks
        for i in mirrored.indices where mirrored[i].count >= 3 {
            mirrored[i][0] *= -1
        }

        for (l, r) in leftRightMirrorPairs where l < mirrored.count && r < mirrored.count {
            mirrored.swapAt(l, r)
        }

        return mirrored
    }
}

struct PoseMLPParameters: Decodable {
    let classNames: [String]
    let featureNames: [String]
    let mean: [Float]
    let std: [Float]
    let w1: [[Float]]
    let b1: [Float]
    let w2: [[Float]]
    let b2: [Float]
    let w3: [[Float]]
    let b3: [Float]
}

final class PoseMLPClassifier {

    private let params: PoseMLPParameters

    init(params: PoseMLPParameters) {
        self.params = params
    }

    static func fromJSON(url: URL) -> PoseMLPClassifier? {
        guard let data = try? Data(contentsOf: url),
              let params = try? JSONDecoder().decode(PoseMLPParameters.self, from: data) else {
            return nil
        }
        return PoseMLPClassifier(params: params)
    }

    func predict(features: PoseFeatureResult) -> (label: String, confidence: Double)? {
        guard features.weightedFeatures.count == params.featureNames.count,
              params.mean.count == params.featureNames.count,
              params.std.count == params.featureNames.count else {
            return nil
        }

        var normalized = [Float](repeating: 0, count: params.featureNames.count)
        for i in normalized.indices {
            normalized[i] = (features.weightedFeatures[i] - params.mean[i]) / (params.std[i] + 1e-6)
        }

        let l1 = relu(add(matVec(params.w1, normalized), params.b1))
        let l2 = relu(add(matVec(params.w2, l1), params.b2))
        let logits = add(matVec(params.w3, l2), params.b3)
        let probs = softmax(logits)

        guard let best = probs.enumerated().max(by: { $0.element < $1.element }) else {
            return nil
        }

        let label = best.offset < params.classNames.count ? params.classNames[best.offset] : "unknown"
        return (label, Double(best.element))
    }

    private func matVec(_ m: [[Float]], _ v: [Float]) -> [Float] {
        m.map { row in zip(row, v).map(*).reduce(0, +) }
    }

    private func add(_ lhs: [Float], _ rhs: [Float]) -> [Float] {
        zip(lhs, rhs).map(+)
    }

    private func relu(_ v: [Float]) -> [Float] {
        v.map { max($0, 0) }
    }

    private func softmax(_ logits: [Float]) -> [Float] {
        guard let maxLogit = logits.max() else { return [] }
        let expVals = logits.map { exp($0 - maxLogit) }
        let sum = expVals.reduce(0, +)
        if sum <= 1e-12 { return Array(repeating: 0, count: logits.count) }
        return expVals.map { $0 / sum }
    }
}
